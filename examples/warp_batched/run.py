#!/usr/bin/env python
"""Benchmark the warp-batched (instances-across-lanes) interpreter against the
standard cusadi2 interpreter (nodes-across-lanes).

Both kernels are data-driven and consume the SAME uploaded plan arrays; the
paper functions use the pre-generated headers in ../paper_results/build/double.
Feasibility gate: the warp-batched work vector is peak_live_slots * 32 scalars,
checked against the device's per-block shared-memory limit (fp64 A1/H1 FK fit
on Ada; the frame Jacobians do not — that asymmetry is the result).

    python run.py --selftest
    python run.py                       # a1_fk,h1_fk with warps=4,8
    python run.py --funcs a1_fk --warps 2,4,8,16

Artifacts: ./results/bench_<stem>.csv (one interpreter row + one wb_w<W> row
per warp count per batch).

Env: CUSADI2_GPU (device index, default "1"), NVCC.
"""
import argparse
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
from wb_template import write_wb_kernel_source

DEVICE = os.environ.get("CUSADI2_GPU", "1")
NVCC = os.environ.get("NVCC", "nvcc")

BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
N_WARMUP = 5
N_TIMED = 20
DTYPE = "double"

PAPER_PREFIX = {"a1_fk": "A1_FK", "a1_fkjac": "A1_FKJAC",
                "h1_fk": "H1_FK", "h1_fkjac": "H1_FKJAC"}


def gpu_arch():
    cc = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", DEVICE]
    ).decode().splitlines()[0].strip()
    return "sm_" + cc.replace(".", "")


def max_block_smem():
    """Per-block dynamic shared-memory limit (bytes) of the selected device."""
    src = os.path.join(BUILD, "smem_probe.cu")
    binp = os.path.join(BUILD, "smem_probe")
    if not os.path.exists(binp):
        with open(src, "w") as f:
            f.write(r"""
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int v; cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    printf("%d\n", v); return 0;
}
""")
        subprocess.run([NVCC, src, "-o", binp], check=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE)
    return int(subprocess.check_output([binp], env=env).decode().strip())


def selftest_fn():
    import casadi as ca
    x = ca.SX.sym("x", 8)
    y = ca.SX.sym("y", 8)
    a = ca.sin(x) * y + ca.cos(y) * x
    b = ca.sqrt(x * x + y * y + 0.1) - ca.tanh(a)
    c = ca.exp(-b * b) + ca.fmax(a, b) + ca.atan2(a, b + 2.0)
    return ca.Function("selftest", [x, y], [ca.vertcat(a, b, c), a * b + c])


def objs(arch):
    """Compile both data-driven kernels once (function-independent)."""
    out = []
    for name, writer in [("kernel", lambda p: write_kernel_source(p, dtype=DTYPE, tiled=False)),
                         ("wb_kernel", lambda p: write_wb_kernel_source(p, dtype=DTYPE))]:
        o = os.path.join(BUILD, f"{name}.o")
        if not os.path.exists(o):
            cu = os.path.join(BUILD, f"{name}.cu")
            writer(cu)
            t0 = time.perf_counter()
            subprocess.run([NVCC, "-std=c++14", "-O3", f"-arch={arch}", "-c", cu, "-o", o],
                           check=True)
            print(f"[shared] {name} nvcc: {time.perf_counter() - t0:.1f}s (one-time)")
        out.append(o)
    return out


def bench_one(stem, prefix, header_path, header_dir, arch, batches, warps_list,
              shared_bytes, smem_cap):
    wb_smem = shared_bytes * 32
    print(f"[{stem}] smem: interp {shared_bytes}B -> wb {wb_smem}B "
          f"(cap {smem_cap}B)")
    if wb_smem > smem_cap:
        print(f"[{stem}] SKIP: warp-batched work vector exceeds per-block limit")
        return

    kernel_o, wb_o = objs(arch)
    bench_o = os.path.join(BUILD, f"{stem}_bench.o")
    binpath = os.path.join(BUILD, f"{stem}_bench")
    subprocess.run([NVCC, "-std=c++14", "-O3", f"-arch={arch}",
                    f"-DCUSADI2_SCALAR={DTYPE}",
                    f"-DPLAN_HEADER=\"{os.path.basename(header_path)}\"",
                    f"-DPLAN={prefix}_plan", f"-DFUNC_NAME=\"{stem}\"",
                    f"-I{PKG}", f"-I{header_dir}",
                    "-c", os.path.join(HERE, "bench_wb.cu"), "-o", bench_o], check=True)
    subprocess.run([NVCC, f"-arch={arch}", bench_o, wb_o, kernel_o,
                    "-o", binpath], check=True)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE)
    rows = []
    for wi, W in enumerate(warps_list):
        cmd = [binpath, str(N_WARMUP), str(N_TIMED), str(W)] + [str(b) for b in batches]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True)
        sys.stdout.write(out.stdout)
        sys.stderr.write(out.stderr)
        out.check_returncode()
        lines = out.stdout.strip().splitlines()
        # keep header + interpreter rows once; wb rows from every warp count
        rows += lines if wi == 0 else [l for l in lines[1:] if ",wb_w" in l]
    csv = os.path.join(RESULTS, f"bench_{stem}.csv")
    with open(csv, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[{stem}] wrote {csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--funcs", default="a1_fk,h1_fk")
    ap.add_argument("--warps", default="4,8")
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    args = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    arch = gpu_arch()
    smem_cap = max_block_smem()
    batches = [int(b) for b in args.batches.split(",")]
    warps_list = [int(w) for w in args.warps.split(",")]

    if args.selftest:
        from cusadi2.dag import extract_level_plan
        from cusadi2.generate import generate_native_plan
        fn = selftest_fn()
        header = os.path.join(BUILD, "selftest_plan.h")
        generate_native_plan(fn, header, prefix="SELFTEST", dtype=DTYPE)
        shared = extract_level_plan(fn).peak_live_slots * 8
        bench_one("selftest", "SELFTEST", header, BUILD, arch, batches, warps_list,
                  shared, smem_cap)
        return

    for stem in args.funcs.split(","):
        assert stem in PAPER_PREFIX, f"unknown func {stem}"
        header = os.path.join(PAPER_BUILD, f"{stem}_plan.h")
        assert os.path.exists(header), f"missing {header} — run paper_results first"
        import re
        text = open(header).read()
        m = re.search(r"StagePlanHost \w+ = \{\s*\{\s*\d+, \d+, [^\n]*\n\s*\d+, (\d+),", text)
        assert m, f"could not parse shared_bytes from {header}"
        bench_one(stem, PAPER_PREFIX[stem], header, PAPER_BUILD, arch, batches,
                  warps_list, int(m.group(1)), smem_cap)


if __name__ == "__main__":
    main()
