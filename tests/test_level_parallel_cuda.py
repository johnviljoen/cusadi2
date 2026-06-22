"""
Phase 2 validation: data-driven level-parallel CUDA kernels vs CasADi CPU oracle.

End-to-end: write fixed kernel .cu → cmake → compile once → load .so →
upload LevelPlan data → validate per function.
"""

import argparse
import os
import sys
import time

import casadi
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src import CUSADI_ROOT_DIR, CUSADI_FUNCTION_DIR, CUSADI_BUILD_DIR, CUSADI_LP_KERNEL_DIR
from src.level_parallel.dag import extract_level_plan
from src.level_parallel.codegen import write_kernel_source
from src.level_parallel.function import CusadiLevelParallelFunction


KERNEL_LIB_NAME = "level_parallel_kernel"


def compile_kernel():
    """Compile the fixed kernel template once."""
    os.makedirs(CUSADI_LP_KERNEL_DIR, exist_ok=True)
    cu_path = os.path.join(CUSADI_LP_KERNEL_DIR, f"{KERNEL_LIB_NAME}.cu")
    write_kernel_source(cu_path)

    cmake_path = os.path.join(CUSADI_ROOT_DIR, "CMakeLists.txt")
    rel_cu = os.path.relpath(cu_path, CUSADI_ROOT_DIR)

    with open(cmake_path, "w") as f:
        f.write(f"""cmake_minimum_required(VERSION 3.15)
project(CusADi LANGUAGES CXX CUDA)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    include_directories(${{CUDA_INCLUDE_DIRS}})
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
set(CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS ${{CUDA_NVCC_FLAGS}}; -O3 --use_fast_math)
include_directories(${{CMAKE_CURRENT_SOURCE_DIR}})
set(LP_SOURCE {rel_cu})
add_library({KERNEL_LIB_NAME} SHARED ${{LP_SOURCE}})
target_link_libraries({KERNEL_LIB_NAME})
""")

    os.makedirs(CUSADI_BUILD_DIR, exist_ok=True)
    t0 = time.time()
    status = os.system(f"cd {CUSADI_BUILD_DIR} && cmake .. 2>&1 && make -j 2>&1")
    elapsed = time.time() - t0
    ok = status == 0
    print(f"  Kernel compile: {elapsed:.1f}s {'ok' if ok else 'FAIL'}")
    return ok


def validate_against_oracle(fn_casadi, num_instances=8, atol=1e-10, rtol=1e-10, seed=0):
    """Run the level-parallel kernel and compare against CasADi CPU oracle."""
    device = "cuda"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    inputs = [
        torch.randn(num_instances, fn_casadi.nnz_in(i),
                     generator=gen, device=device, dtype=torch.double).contiguous()
        for i in range(fn_casadi.n_in())
    ]

    lp_fn = CusadiLevelParallelFunction(fn_casadi, num_instances, lib_name=KERNEL_LIB_NAME)
    lp_fn.evaluate(inputs)
    torch.cuda.synchronize()

    gpu_outs = [
        lp_fn.outputs_sparse[i].detach().cpu().numpy().astype(np.float64)
        for i in range(fn_casadi.n_out())
    ]

    cpu_outs = []
    for i_out in range(fn_casadi.n_out()):
        cpu_outs.append(np.zeros((num_instances, fn_casadi.nnz_out(i_out)), dtype=np.float64))
    for n in range(num_instances):
        inputs_np = [
            inputs[i][n, :].detach().cpu().numpy().astype(np.float64)
            for i in range(fn_casadi.n_in())
        ]
        results = fn_casadi.call(inputs_np)
        for i_out in range(fn_casadi.n_out()):
            cpu_outs[i_out][n, :] = np.asarray(results[i_out].nonzeros(), dtype=np.float64)

    all_pass = True
    for i_out in range(fn_casadi.n_out()):
        abs_err = np.abs(cpu_outs[i_out] - gpu_outs[i_out])
        max_abs = float(abs_err.max()) if abs_err.size else 0.0
        denom = np.maximum(np.abs(cpu_outs[i_out]), np.abs(gpu_outs[i_out]))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(denom > 0, abs_err / denom, 0.0)
        max_rel = float(rel_err.max()) if rel_err.size else 0.0
        tol = atol + rtol * np.abs(cpu_outs[i_out])
        n_bad = int((abs_err > tol).sum())
        status = "ok" if n_bad == 0 else "FAIL"
        print(f"    out[{i_out}]: {status}  max|err|={max_abs:.2e}  max_rel={max_rel:.2e}  mismatches={n_bad}")
        if n_bad > 0:
            all_pass = False
            bad_coords = np.argwhere(abs_err > tol)
            for j in range(min(5, len(bad_coords))):
                inst, k = bad_coords[j]
                print(f"      inst={inst} k={k}  cpu={cpu_outs[i_out][inst,k]:+.15e}  "
                      f"gpu={gpu_outs[i_out][inst,k]:+.15e}  |err|={abs_err[inst,k]:.2e}")
    return all_pass


def test_function(name, atol=1e-10):
    path = os.path.join(CUSADI_FUNCTION_DIR, f"{name}.casadi")
    if not os.path.exists(path):
        print(f"SKIP {name}: file not found")
        return None

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    f = casadi.Function.load(path)

    t0 = time.time()
    plan = extract_level_plan(f)
    t_extract = time.time() - t0
    from src.level_parallel.codegen import pick_block_size
    block_size = pick_block_size(plan)
    print(f"  DAG: {plan.total_compute_ops} ops, {plan.n_compute_levels} levels, "
          f"block_size={block_size}, peak_live={plan.peak_live_slots}, "
          f"shared_mem={plan.shared_mem_bytes/1024:.1f}KB  ({t_extract:.2f}s)")

    print(f"  Validation (8 instances, atol={atol:.0e}):")
    passed = validate_against_oracle(f, num_instances=8, atol=atol)
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn", default="double_integrator",
                        help="Function name or 'all'")
    args = parser.parse_args()

    targets = {
        "double_integrator": 1e-14,
        "panda_aba_jac_analytical": 1e-10,
        "a1_aba_jac_analytical": 1e-8,
        "h1_aba_jac_analytical": 1e-8,
    }

    print("Compiling data-driven kernel (one-time)...")
    if not compile_kernel():
        sys.exit(1)

    if args.fn == "all":
        test_list = list(targets.items())
    elif args.fn in targets:
        test_list = [(args.fn, targets[args.fn])]
    else:
        test_list = [(args.fn, 1e-10)]

    results = {}
    for name, atol in test_list:
        results[name] = test_function(name, atol)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_pass = True
    for name, passed in results.items():
        if passed is None:
            print(f"  {name:>30s}: SKIP")
        elif passed:
            print(f"  {name:>30s}: PASS")
        else:
            print(f"  {name:>30s}: FAIL")
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
