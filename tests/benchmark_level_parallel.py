"""
Phase 3 benchmarking: level-parallel kernel timing across batch sizes.

Runs level-parallel kernels first (fast — kernel already compiled).
Then optionally compiles and benchmarks sequential kernels for comparison.

Usage:
    python tests/benchmark_level_parallel.py                # LP only
    python tests/benchmark_level_parallel.py --sequential   # LP + sequential
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
from src.level_parallel.codegen import write_kernel_source, pick_block_size
from src.level_parallel.function import CusadiLevelParallelFunction
from src.CusadiFunction import CusadiFunction

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
N_WARMUP = 5
N_EVALS = 20

TARGETS = [
    "panda_aba_jac_analytical",
    "a1_aba_jac_analytical",
    "h1_aba_jac_analytical",
]


def ensure_lp_kernel_compiled():
    """Compile the data-driven kernel if not already built."""
    so_path = os.path.join(CUSADI_BUILD_DIR, "liblevel_parallel_kernel.so")
    if os.path.exists(so_path):
        return True

    os.makedirs(CUSADI_LP_KERNEL_DIR, exist_ok=True)
    cu_path = os.path.join(CUSADI_LP_KERNEL_DIR, "level_parallel_kernel.cu")
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
add_library(level_parallel_kernel SHARED ${{LP_SOURCE}})
""")

    os.makedirs(CUSADI_BUILD_DIR, exist_ok=True)
    status = os.system(f"cd {CUSADI_BUILD_DIR} && cmake .. 2>&1 && make -j 2>&1")
    return status == 0


def ensure_sequential_compiled(fn_names):
    """Compile sequential kernels for the given function names."""
    cmake_path = os.path.join(CUSADI_ROOT_DIR, "CMakeLists.txt")
    with open(cmake_path, "w") as f:
        f.write("""cmake_minimum_required(VERSION 3.15)
project(CusADi LANGUAGES CXX CUDA)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()
set(CMAKE_CXX_STANDARD 11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 --use_fast_math)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
""")
        for name in fn_names:
            cu_path = f"codegen/{name}.cu"
            if not os.path.exists(os.path.join(CUSADI_ROOT_DIR, cu_path)):
                print(f"  WARNING: {cu_path} not found, skipping {name}")
                continue
            var = name.upper() + "_SOURCE"
            f.write(f"set({var} {cu_path})\n")
            f.write(f"add_library({name} SHARED ${{{var}}})\n")

    os.makedirs(CUSADI_BUILD_DIR, exist_ok=True)
    status = os.system(f"cd {CUSADI_BUILD_DIR} && cmake .. 2>&1 && make -j 2>&1")
    return status == 0


def benchmark_lp(fn_casadi, batch_sizes):
    """Benchmark the level-parallel kernel across batch sizes."""
    results = {}
    for bs in batch_sizes:
        lp_fn = CusadiLevelParallelFunction(fn_casadi, bs, lib_name="level_parallel_kernel")
        inputs = [
            torch.randn(bs, fn_casadi.nnz_in(i), device="cuda", dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_in())
        ]

        for _ in range(N_WARMUP):
            lp_fn.evaluate(inputs)
        torch.cuda.synchronize()

        times = []
        for _ in range(N_EVALS):
            lp_fn.evaluate(inputs)
            times.append(lp_fn.eval_time)
        torch.cuda.synchronize()

        results[bs] = np.array(times)
    return results


def benchmark_seq(fn_casadi, batch_sizes):
    """Benchmark the sequential kernel across batch sizes."""
    so_path = os.path.join(CUSADI_BUILD_DIR, f"lib{fn_casadi.name()}.so")
    if not os.path.exists(so_path):
        return None

    results = {}
    for bs in batch_sizes:
        seq_fn = CusadiFunction(fn_casadi, bs)
        inputs = [
            torch.randn(bs, fn_casadi.nnz_in(i), device="cuda", dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_in())
        ]

        for _ in range(N_WARMUP):
            seq_fn.evaluate(inputs)
        torch.cuda.synchronize()

        times = []
        for _ in range(N_EVALS):
            seq_fn.evaluate(inputs)
            times.append(seq_fn.eval_time)
        torch.cuda.synchronize()

        results[bs] = np.array(times)
    return results


def print_results(name, lp_results, seq_results=None):
    print(f"\n{'='*90}")
    print(f"  {name}")
    print(f"{'='*90}")

    if seq_results:
        print(f"  {'batch':>6s}  {'LP median':>12s}  {'LP mean':>12s}  "
              f"{'SEQ median':>12s}  {'SEQ mean':>12s}  {'speedup':>8s}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
        for bs in sorted(lp_results.keys()):
            lp_med = np.median(lp_results[bs])
            lp_mean = np.mean(lp_results[bs])
            seq_med = np.median(seq_results[bs])
            seq_mean = np.mean(seq_results[bs])
            speedup = seq_med / lp_med if lp_med > 0 else float('inf')
            print(f"  {bs:>6d}  {lp_med:>12.6f}  {lp_mean:>12.6f}  "
                  f"{seq_med:>12.6f}  {seq_mean:>12.6f}  {speedup:>7.2f}x")
    else:
        print(f"  {'batch':>6s}  {'LP median (s)':>14s}  {'LP mean (s)':>14s}  "
              f"{'LP min (s)':>14s}  {'per-instance (us)':>18s}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*18}")
        for bs in sorted(lp_results.keys()):
            lp_med = np.median(lp_results[bs])
            lp_mean = np.mean(lp_results[bs])
            lp_min = np.min(lp_results[bs])
            per_inst = lp_med / bs * 1e6
            print(f"  {bs:>6d}  {lp_med:>14.6f}  {lp_mean:>14.6f}  "
                  f"{lp_min:>14.6f}  {per_inst:>18.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", action="store_true",
                        help="Also benchmark sequential kernels (requires compilation)")
    parser.add_argument("--fn", default="all",
                        help="Function name or 'all'")
    args = parser.parse_args()

    targets = TARGETS if args.fn == "all" else [args.fn]

    # Compile LP kernel
    print("Ensuring level-parallel kernel is compiled...")
    if not ensure_lp_kernel_compiled():
        print("FAILED to compile level-parallel kernel")
        sys.exit(1)

    # Run LP benchmarks first
    all_lp = {}
    for name in targets:
        path = os.path.join(CUSADI_FUNCTION_DIR, f"{name}.casadi")
        if not os.path.exists(path):
            print(f"SKIP {name}: not found")
            continue
        f = casadi.Function.load(path)
        plan = extract_level_plan(f)
        bs = pick_block_size(plan)
        print(f"\nBenchmarking LP: {name}  ({plan.total_compute_ops} ops, "
              f"{plan.n_compute_levels} levels, block={bs}, "
              f"smem={plan.shared_mem_bytes/1024:.1f}KB)")
        lp_results = benchmark_lp(f, BATCH_SIZES)
        all_lp[name] = (f, lp_results)
        print_results(name, lp_results)

    # Run sequential benchmarks if requested
    all_seq = {}
    if args.sequential:
        print(f"\n\n{'#'*90}")
        print("  Compiling sequential kernels (this may take a long time)...")
        print(f"{'#'*90}")
        if not ensure_sequential_compiled(targets):
            print("WARNING: sequential compilation failed")
        else:
            for name in targets:
                if name not in all_lp:
                    continue
                f = all_lp[name][0]
                so_path = os.path.join(CUSADI_BUILD_DIR, f"lib{name}.so")
                if not os.path.exists(so_path):
                    print(f"SKIP sequential {name}: .so not found")
                    continue
                print(f"\nBenchmarking SEQ: {name}")
                seq_results = benchmark_seq(f, BATCH_SIZES)
                if seq_results:
                    all_seq[name] = seq_results

    # Final summary
    print(f"\n\n{'#'*90}")
    print("  FINAL RESULTS")
    print(f"{'#'*90}")
    for name in targets:
        if name not in all_lp:
            continue
        _, lp_results = all_lp[name]
        seq_results = all_seq.get(name)
        print_results(name, lp_results, seq_results)


if __name__ == "__main__":
    main()
