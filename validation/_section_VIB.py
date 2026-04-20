"""Table III — Compilation time: sequential codegen+nvcc vs level-parallel."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time

from validation._helpers import load_casadi_fn, print_header, repo_path


BUILD_SEQ_DIR_NEW = repo_path("build_seq_compile_test")
FUNCTIONS = ["fn_1e1", "fn_1e4", "fn_1e5"]


def _time_sequential_compile(fn_name: str, timeout_s: int = 1800) -> float:
    """Full codegen + nvcc time for one sequential kernel. Isolated build dir
    so we truly measure compile, not cache."""
    from src.generateCUDACode import generateCUDACodeDouble

    f = load_casadi_fn(repo_path("src", "benchmark_functions", f"{fn_name}.casadi"))

    # 1. Remove any previous build artifacts for this fn
    shutil.rmtree(BUILD_SEQ_DIR_NEW, ignore_errors=True)
    os.makedirs(BUILD_SEQ_DIR_NEW, exist_ok=True)

    # 2. Codegen → codegen/{fn_name}.cu (freshly generated)
    cu_rel = f"codegen/{fn_name}.cu"
    cu_abs = repo_path(cu_rel)
    os.makedirs(repo_path("codegen"), exist_ok=True)

    t0 = time.time()
    generateCUDACodeDouble(f)       # writes codegen/{fn_name}.cu

    # 3. CMake + make in the isolated build dir
    cmake_path = os.path.join(BUILD_SEQ_DIR_NEW, "CMakeLists.txt")
    var = fn_name.upper() + "_SOURCE"
    with open(cmake_path, "w") as fcm:
        fcm.write(textwrap.dedent(f"""\
            cmake_minimum_required(VERSION 3.15)
            project(CusADi_VIB LANGUAGES CXX CUDA)
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES 89)
            endif()
            set(CMAKE_CXX_STANDARD 11)
            set(CUDA_NVCC_FLAGS ${{CUDA_NVCC_FLAGS}}; -O3 --use_fast_math)
            include_directories(${{CMAKE_CURRENT_SOURCE_DIR}}/..)
            set({var} ../{cu_rel})
            add_library({fn_name} SHARED ${{{var}}})
        """))
    proc = subprocess.run(
        f"cd {BUILD_SEQ_DIR_NEW} && cmake . 2>&1 && make -j 2>&1",
        shell=True, capture_output=True, text=True, timeout=timeout_s)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"    {fn_name}: compile FAILED:")
        print((proc.stdout or "")[-1500:])
        return float("nan")
    return elapsed


def _time_lp_compile() -> float:
    """Full cold compile of liblevel_parallel_kernel.so. We remove the .so,
    regenerate the .cu via write_kernel_source, then cmake+make."""
    from src.level_parallel.codegen import write_kernel_source
    cu_rel = "codegen/level_parallel_kernel.cu"
    cu_abs = repo_path(cu_rel)
    so_abs = repo_path("build", "liblevel_parallel_kernel.so")
    build_dir = repo_path("build_lp_compile_test")
    shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)

    t0 = time.time()
    # regen kernel source (idempotent — but we time it as part of the cost)
    write_kernel_source(cu_abs)
    cmake_path = os.path.join(build_dir, "CMakeLists.txt")
    with open(cmake_path, "w") as fcm:
        fcm.write(textwrap.dedent(f"""\
            cmake_minimum_required(VERSION 3.15)
            project(CusADi_LP LANGUAGES CXX CUDA)
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES 89)
            endif()
            set(CMAKE_CXX_STANDARD 11)
            set(CUDA_NVCC_FLAGS ${{CUDA_NVCC_FLAGS}}; -O3 --use_fast_math)
            include_directories(${{CMAKE_CURRENT_SOURCE_DIR}}/..)
            set(LP_SOURCE ../codegen/level_parallel_kernel.cu)
            add_library(level_parallel_kernel SHARED ${{LP_SOURCE}})
        """))
    proc = subprocess.run(
        f"cd {build_dir} && cmake . 2>&1 && make -j 2>&1",
        shell=True, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"    LP compile FAILED:")
        print((proc.stdout or "")[-1500:])
        return float("nan")
    return elapsed


def run_section_VIB(skip: bool = False) -> dict:
    print_header("SECTION VI-B / TABLE III : Compilation time (seq vs LP)")
    if skip:
        print("  SKIPPED (--skip-compilation). Paper values: 1.1, 10.8, 322.8 s seq; "
              "LP is 1.5 s one-off.")
        return {"skipped": True}

    # 1) LP one-off
    print("  Timing LP kernel (cold, write_kernel_source + nvcc)...")
    lp_s = _time_lp_compile()
    print(f"    LP kernel compiled in {lp_s:.1f} s (applies once, reused across all fns).")

    # 2) Sequential per-function
    print(f"  {'Function':<10} {'Seq (s)':>10} {'LP (s)':>8} {'Seq/LP':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    out = {"lp_kernel_compile_s": lp_s}
    for fn_name in FUNCTIONS:
        print(f"  {fn_name:<10} compiling sequential...", flush=True, end=" ")
        seq_s = _time_sequential_compile(fn_name)
        ratio = seq_s / lp_s if lp_s > 0 else float("nan")
        print(f"seq={seq_s:.1f} s")
        print(f"  {fn_name:<10} {seq_s:>10.1f} {lp_s:>8.1f} {ratio:>7.1f}x")
        out[fn_name] = {"seq_s": seq_s, "lp_s": lp_s, "speedup": ratio}
    return out
