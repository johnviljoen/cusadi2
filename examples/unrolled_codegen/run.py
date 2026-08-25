#!/usr/bin/env python
"""Benchmark the resurrected fully-unrolled level-parallel codegen against the
cusadi2 data-driven interpreter.

For the paper functions (a1_fk / a1_fkjac / h1_fk / h1_fkjac) the unrolled
kernel is emitted from the EXISTING plan headers in
``../paper_results/build/double`` — the exact LevelPlans the paper's
interpreter numbers used — so the two backends share slot assignment, level
schedule, block size and shared-memory footprint; only dispatch differs. The
bench binary validates the backends against each other elementwise and sweeps
batch size with kernel-only cudaEvent timing (one block per instance,
n_stages=1 — the paper's "lp" configuration). nvcc wall time on the unrolled
translation unit is recorded per function; the interpreter kernel compiles once
for all functions.

    python run.py --selftest            # small casadi function, fast gate
    python run.py --funcs a1_fk,h1_fk   # default: all four paper functions

Artifacts: ./build/<stem>_unrolled.cu, ./results/bench_<stem>.csv,
./results/compile_<stem>.json.

Env: CUSADI2_GPU (device index, default "1"), NVCC.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cusadi2
from cusadi2.kernel_template import write_kernel_source

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_BUILD = os.path.join(HERE, "..", "paper_results", "build", "double")
PKG = cusadi2.__path__[0]
BUILD = os.path.join(HERE, "build")
RESULTS = os.path.join(HERE, "results")

sys.path.insert(0, HERE)
from emit_unrolled import (parse_plan_header, write_unrolled_from_fn,
                           write_unrolled_source)

DEVICE = os.environ.get("CUSADI2_GPU", "1")
NVCC = os.environ.get("NVCC", "nvcc")

BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
N_WARMUP = 5
N_TIMED = 20
DTYPE = "double"

# paper function stems -> C prefix in the pre-generated plan headers.
# *_unr are the N=8 horizon-unrolled plans (the paper's "lp"/"batch" configs);
# ~8x the nodes, for the compile-time-scaling datapoints.
PAPER_PREFIX = {"a1_fk": "A1_FK", "a1_fkjac": "A1_FKJAC",
                "h1_fk": "H1_FK", "h1_fkjac": "H1_FKJAC",
                "a1_fk_unr": "A1_FK_UNR", "a1_fkjac_unr": "A1_FKJAC_UNR",
                "h1_fk_unr": "H1_FK_UNR", "h1_fkjac_unr": "H1_FKJAC_UNR"}


def gpu_arch():
    cc = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", DEVICE]
    ).decode().splitlines()[0].strip()
    return "sm_" + cc.replace(".", "")


def selftest_fn():
    """Small heterogeneous-op function: enough levels/ops to exercise dispatch,
    compiles in seconds."""
    import casadi as ca
    x = ca.SX.sym("x", 8)
    y = ca.SX.sym("y", 8)
    a = ca.sin(x) * y + ca.cos(y) * x
    b = ca.sqrt(x * x + y * y + 0.1) - ca.tanh(a)
    c = ca.exp(-b * b) + ca.fmax(a, b) + ca.atan2(a, b + 2.0)
    return ca.Function("selftest", [x, y], [ca.vertcat(a, b, c), a * b + c])


def timed_nvcc(cmd):
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - t0


def interp_kernel_obj(arch):
    """Compile the interpreter kernel once (function-independent)."""
    kernel_o = os.path.join(BUILD, "kernel.o")
    if os.path.exists(kernel_o):
        return kernel_o, None
    kernel_cu = os.path.join(BUILD, "kernel.cu")
    write_kernel_source(kernel_cu, dtype=DTYPE, tiled=False)
    s = timed_nvcc([NVCC, "-std=c++14", "-O3", f"-arch={arch}",
                    "-c", kernel_cu, "-o", kernel_o])
    print(f"[shared] interpreter kernel nvcc: {s:.1f}s (one-time, function-independent)")
    return kernel_o, s


def bench_one(stem, prefix, header_path, header_dir, arch, batches,
              stats, emit_s, n_instructions=None):
    """Compile the bench binary (interpreter + unrolled + harness) and run the
    sweep. The unrolled TU's nvcc time is the per-function compile datapoint."""
    kernel_o, interp_nvcc_s = interp_kernel_obj(arch)

    unr_cu = os.path.join(BUILD, f"{stem}_unrolled.cu")
    unr_o = os.path.join(BUILD, f"{stem}_unrolled.o")
    print(f"[{stem}] nvcc on unrolled TU ({stats['cu_lines']} lines) ...", flush=True)
    unr_nvcc_s = timed_nvcc([NVCC, "-std=c++14", "-O3", f"-arch={arch}",
                             "-c", unr_cu, "-o", unr_o])
    print(f"[{stem}] unrolled nvcc: {unr_nvcc_s:.1f}s")

    bench_o = os.path.join(BUILD, f"{stem}_bench.o")
    binpath = os.path.join(BUILD, f"{stem}_bench")
    subprocess.run([NVCC, "-std=c++14", "-O3", f"-arch={arch}",
                    f"-DCUSADI2_SCALAR={DTYPE}",
                    f"-DPLAN_HEADER=\"{os.path.basename(header_path)}\"",
                    f"-DPLAN={prefix}_plan", f"-DFUNC_NAME=\"{stem}\"",
                    f"-I{PKG}", f"-I{header_dir}",
                    "-c", os.path.join(HERE, "bench.cu"), "-o", bench_o], check=True)
    subprocess.run([NVCC, f"-arch={arch}", bench_o, unr_o, kernel_o,
                    "-o", binpath], check=True)

    rec = {"func": stem, "dtype": DTYPE, "n_instructions": n_instructions,
           "emit_s": emit_s, "nvcc_unrolled_s": unr_nvcc_s,
           "nvcc_interpreter_s": interp_nvcc_s, **stats}
    with open(os.path.join(RESULTS, f"compile_{stem}.json"), "w") as f:
        json.dump(rec, f, indent=2)

    csv = os.path.join(RESULTS, f"bench_{stem}.csv")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE)
    cmd = [binpath, str(N_WARMUP), str(N_TIMED)] + [str(b) for b in batches]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    out.check_returncode()
    with open(csv, "w") as f:
        f.write(out.stdout)
    print(f"[{stem}] wrote {csv}")


def run_selftest(arch, batches):
    from cusadi2.generate import generate_native_plan
    fn = selftest_fn()
    header = os.path.join(BUILD, "selftest_plan.h")
    generate_native_plan(fn, header, prefix="SELFTEST", dtype=DTYPE)
    t0 = time.perf_counter()
    _, stats = write_unrolled_from_fn(fn, os.path.join(BUILD, "selftest_unrolled.cu"),
                                      dtype=DTYPE)
    emit_s = time.perf_counter() - t0
    bench_one("selftest", "SELFTEST", header, BUILD, arch, batches,
              stats, emit_s, n_instructions=fn.n_instructions())


def run_paper(stem, arch, batches):
    prefix = PAPER_PREFIX[stem]
    header = os.path.join(PAPER_BUILD, f"{stem}_plan.h")
    assert os.path.exists(header), f"missing {header} — run paper_results first"
    plan, bs, shared_bytes = parse_plan_header(header, prefix)
    print(f"[{stem}] nodes={len(plan.node_ops)} levels={len(plan.level_offsets)} "
          f"block={bs} smem={shared_bytes / 1024:.1f}KB")
    t0 = time.perf_counter()
    _, stats = write_unrolled_source(plan, stem, bs, shared_bytes,
                                     os.path.join(BUILD, f"{stem}_unrolled.cu"),
                                     dtype=DTYPE)
    emit_s = time.perf_counter() - t0
    print(f"[{stem}] emitted {stats['cu_lines']} lines in {emit_s:.2f}s")
    bench_one(stem, prefix, header, PAPER_BUILD, arch, batches, stats, emit_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--funcs", default="a1_fk,h1_fk,a1_fkjac,h1_fkjac")
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    args = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    arch = gpu_arch()
    batches = [int(b) for b in args.batches.split(",")]
    if args.selftest:
        run_selftest(arch, batches)
        return
    for stem in args.funcs.split(","):
        assert stem in PAPER_PREFIX, f"unknown func {stem}"
        run_paper(stem, arch, batches)


if __name__ == "__main__":
    main()
