"""Fill the missing 2x2 corner: stage OFF, LP OFF.

Run *original* cusadi on the *unrolled-horizon* function (N stamps of
f_stage as one big CasADi function), at batch B. One block per batch
instance, one thread per block, sequential traversal of the full
unrolled DAG. This is what cusadi looked like before any of our work —
the deployable prior baseline at the same total work as our system.

Steps:
  1. Build f_unrolled = stamp(f_stage, N) symbolically.
  2. Save it as a .casadi fixture (so SequentialFn can load it).
  3. Codegen via generateCUDACodeDouble (writes codegen/<name>.cu).
  4. Append to CMakeLists, run cmake + make → measure NVCC compile time.
  5. Run benchmark at the same B values as the existing ablation.
  6. Persist results (compile time, per-B runtime) to JSON for the plot.

Why this is interesting:
  At horizon N=12 on A1 ABA Jacobian (40,969 instr/stage), the unrolled
  function has ~490k instructions in a single .cu — well past fn_1e5
  (103k) which already takes ~322 s to NVCC compile. Either compile
  time becomes a defense talking point (showing why the prior pipeline
  is impractical at MPC horizons) or, if it OOMs, the absence is itself
  the data point.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

# Make src importable when running as a script.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from validation._helpers import (load_casadi_fn, repo_path, SequentialFn,
                                  benchmark_kernel_only)
from validation._section_stage_robotics import build_unrolled_aba_jac
import casadi as ca


FN_NAME    = "a1_aba_jac_analytical"
N_STAGES   = 12
BATCHES    = [1, 8, 32, 128, 256, 1024, 4096]
TIMEOUT_S  = 7200    # 2 hours — be generous; NVCC at 490k instr is unknown territory
OUT_JSON   = repo_path("figures", "lp_ablation_unrolled_cusadi.json")


def _emit_progress(msg: str):
    print(f"[unrolled-cusadi] {msg}", flush=True)


def build_or_load_unrolled_function():
    """Build the unrolled function and persist it. Idempotent — if the
    .casadi file already exists we just load it."""
    f_stage = load_casadi_fn(FN_NAME)
    fn_name = f"{FN_NAME}_unrolled_N{N_STAGES}"
    out_path = repo_path("src", "casadi_functions", f"{fn_name}.casadi")
    if os.path.exists(out_path):
        _emit_progress(f"unrolled fixture already exists: {out_path}")
        f_unr = ca.Function.load(out_path)
    else:
        _emit_progress(f"stamping f_stage × {N_STAGES} symbolically ...")
        t0 = time.time()
        f_unr = build_unrolled_aba_jac(f_stage, N_STAGES)
        t_stamp = time.time() - t0
        _emit_progress(f"stamping done in {t_stamp:.2f}s, "
                       f"{f_unr.n_instructions():,} instructions")
        f_unr.save(out_path)
        _emit_progress(f"saved fixture: {out_path}")
    return f_unr


def codegen_cuda(f_unr) -> tuple[str, float]:
    """Generate the .cu file via original cusadi codegen. Returns the
    .cu path and codegen time."""
    from src.generateCUDACode import generateCUDACodeDouble
    cu_path = repo_path("codegen", f"{f_unr.name()}.cu")
    _emit_progress(f"generating CUDA code → {cu_path} ...")
    t0 = time.time()
    generateCUDACodeDouble(f_unr)
    t_codegen = time.time() - t0
    size_mb = os.path.getsize(cu_path) / (1024 * 1024)
    _emit_progress(f"codegen done in {t_codegen:.2f}s ({size_mb:.1f} MB .cu)")
    return cu_path, t_codegen


def ensure_cmake_target(fn_name: str):
    """Append a target for fn_name to CMakeLists.txt if not already present."""
    cmake_path = repo_path("CMakeLists.txt")
    with open(cmake_path) as f:
        content = f.read()
    if fn_name in content:
        _emit_progress(f"CMake target {fn_name} already present")
        return
    var = fn_name.upper() + "_SOURCE"
    addition = (f"\nset({var} codegen/{fn_name}.cu)\n"
                f"add_library({fn_name} SHARED ${{{var}}})\n")
    with open(cmake_path, "a") as f:
        f.write(addition)
    _emit_progress(f"appended CMake target for {fn_name}")


def nvcc_compile(fn_name: str) -> tuple[float, bool, str]:
    """Run cmake + make to NVCC-compile the new target. Returns
    (elapsed_s, success, log_tail)."""
    build_dir = repo_path("build")
    os.makedirs(build_dir, exist_ok=True)
    so_path = os.path.join(build_dir, f"lib{fn_name}.so")
    if os.path.exists(so_path):
        _emit_progress(f".so already exists: {so_path}  "
                       f"(remove to force rebuild)")
        return 0.0, True, "(cached)"
    _emit_progress(f"NVCC compile (timeout {TIMEOUT_S}s) ...")
    t0 = time.time()
    proc = subprocess.run(
        f"cd {build_dir} && cmake .. && make {fn_name} -j 2>&1",
        shell=True, capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    elapsed = time.time() - t0
    log_tail = (proc.stdout or "")[-1500:]
    success = (proc.returncode == 0) and os.path.exists(so_path)
    _emit_progress(f"NVCC done in {elapsed:.1f}s, "
                   f"returncode={proc.returncode}, success={success}")
    return elapsed, success, log_tail


def benchmark(f_unr, batches=BATCHES, n_warmup=3, n_timed=15) -> dict:
    """Run SequentialFn at each batch size, return median µs per B."""
    import torch  # noqa
    build_dir = repo_path("build")
    out: dict = {}
    _emit_progress("running benchmark ...")
    for B in batches:
        try:
            inputs = [torch.randn(B, f_unr.nnz_in(i), device="cuda",
                                   dtype=torch.double).contiguous()
                      for i in range(f_unr.n_in())]
            seq = SequentialFn(f_unr, B, build_dir)
            seq.set_inputs(inputs)
            us = benchmark_kernel_only(
                seq.evaluate, lambda: seq.eval_time, n_warmup, n_timed)
            out[str(B)] = us
            _emit_progress(f"  B={B:>5}: {us:>10.1f} µs")
            del seq
            torch.cuda.empty_cache()
        except Exception as e:
            _emit_progress(f"  B={B:>5}: FAILED ({type(e).__name__}: {e})")
            out[str(B)] = None
    return out


def main():
    f_unr = build_or_load_unrolled_function()
    cu_path, t_codegen = codegen_cuda(f_unr)
    ensure_cmake_target(f_unr.name())
    t_nvcc, success, log_tail = nvcc_compile(f_unr.name())

    if not success:
        _emit_progress(f"compile FAILED — log tail:\n{log_tail}")
        result = {
            "fn_name":     f_unr.name(),
            "n_stages":    N_STAGES,
            "n_instructions": f_unr.n_instructions(),
            "codegen_s":   t_codegen,
            "nvcc_s":      t_nvcc,
            "compile_success": False,
            "log_tail":    log_tail,
            "runtime_us":  {},
        }
    else:
        runtime = benchmark(f_unr)
        result = {
            "fn_name":     f_unr.name(),
            "n_stages":    N_STAGES,
            "n_instructions": f_unr.n_instructions(),
            "codegen_s":   t_codegen,
            "nvcc_s":      t_nvcc,
            "compile_success": True,
            "runtime_us":  runtime,
        }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    _emit_progress(f"results written to {OUT_JSON}")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
