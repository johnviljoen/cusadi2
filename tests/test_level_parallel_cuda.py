"""
Phase 2 validation: level-parallel CUDA kernels vs CasADi CPU oracle.

End-to-end: extract DAG → codegen .cu → cmake → compile → load .so → validate.

Uses the existing validation harness (src/validation.py) through
CusadiLevelParallelFunction which provides the same interface as CusadiFunction.
"""

import argparse
import os
import sys
import time

import casadi
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src import CUSADI_ROOT_DIR, CUSADI_FUNCTION_DIR, CUSADI_BUILD_DIR, CUSADI_CODEGEN_DIR
from src.level_parallel.dag import extract_level_plan, verify_level_plan_parallel
from src.level_parallel.codegen import generate_level_parallel_cuda, pick_block_size
from src.level_parallel.function import CusadiLevelParallelFunction
from src.generateCUDACode import generateCMakeLists


def compile_level_parallel(fn_name: str, cu_filepath: str):
    """Compile a single level-parallel .cu file via cmake."""
    cmake_filepath = os.path.join(CUSADI_ROOT_DIR, "CMakeLists.txt")
    lib_name = f"{fn_name}_lp"

    with open(cmake_filepath, "w") as f:
        f.write("cmake_minimum_required(VERSION 3.15)\n")
        f.write("project(CusADi LANGUAGES CXX CUDA)\n")
        f.write("""
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    message("CUDA found")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
set(CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 --use_fast_math)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

""")
        rel_cu = os.path.relpath(cu_filepath, CUSADI_ROOT_DIR)
        src_var = lib_name.upper() + "_SOURCE"
        f.write(f"set({src_var} {rel_cu})\n")
        f.write(f"add_library({lib_name} SHARED ${{{src_var}}})\n")
        f.write(f"target_link_libraries({lib_name})\n")

    os.makedirs(CUSADI_BUILD_DIR, exist_ok=True)
    status = os.system(f"cd {CUSADI_BUILD_DIR} && cmake .. 2>&1 && make -j 2>&1")
    if status != 0:
        print("COMPILATION FAILED")
        return False
    return True


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

    # GPU: level-parallel kernel
    lp_fn = CusadiLevelParallelFunction(fn_casadi, num_instances)
    lp_fn.evaluate(inputs)
    torch.cuda.synchronize()

    gpu_outs = [
        lp_fn.outputs_sparse[i].detach().cpu().numpy().astype(np.float64)
        for i in range(fn_casadi.n_out())
    ]

    # CPU: CasADi oracle
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

    # Compare
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

    # Phase 1: extract + verify in Python
    t0 = time.time()
    plan = extract_level_plan(f)
    t_extract = time.time() - t0
    block_size = pick_block_size(plan)
    print(f"  DAG: {plan.total_compute_ops} ops, {plan.n_compute_levels} levels, "
          f"block_size={block_size}, peak_live={plan.peak_live_slots}, "
          f"shared_mem={plan.shared_mem_bytes/1024:.1f}KB")

    # Phase 2: codegen
    os.makedirs(CUSADI_CODEGEN_DIR, exist_ok=True)
    cu_path = os.path.join(CUSADI_CODEGEN_DIR, f"{name}_lp.cu")
    t0 = time.time()
    generate_level_parallel_cuda(plan, cu_path, block_size, debug_mode=True)
    t_codegen = time.time() - t0
    cu_size = os.path.getsize(cu_path)
    print(f"  Codegen: {t_codegen:.2f}s, {cu_size/1024:.0f} KB .cu file")

    # Compile
    t0 = time.time()
    ok = compile_level_parallel(name, cu_path)
    t_compile = time.time() - t0
    print(f"  Compile: {t_compile:.1f}s {'ok' if ok else 'FAIL'}")
    if not ok:
        return False

    # Validate
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
