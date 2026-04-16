"""
Benchmark: sequential vs level-parallel CusADi on the original paper's
five functions (fn_1e1 through fn_1e5).

Sequential kernels compile to build_sequential/ so both .so sets coexist.
LP kernel uses the existing build/liblevel_parallel_kernel.so.

Usage:
    CUDA_VISIBLE_DEVICES=5 conda run -n cusadi2 python benchmark/run_paper_comparison.py
"""

import argparse
import csv
import ctypes
import os
import subprocess
import sys
import textwrap
import time

import casadi
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src import (CUSADI_ROOT_DIR, CUSADI_BENCHMARK_DIR, CUSADI_BUILD_DIR,
                 CUSADI_CODEGEN_DIR)
from src.generateCUDACode import generateCUDACodeDouble
from src.level_parallel.dag import extract_level_plan, verify_level_plan_parallel
from src.level_parallel.codegen import write_kernel_source, pick_block_size
from src.level_parallel.function import CusadiLevelParallelFunction

FUNCTIONS = ["fn_1e1", "fn_1e2", "fn_1e3", "fn_1e4", "fn_1e5"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
               8192, 16384, 32768]
N_WARMUP = 10
N_EVALS = 100
SEQ_COMPILE_TIMEOUT = 1800  # 30 minutes per function
SEQ_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build_sequential")
RESULTS_DIR = os.path.join(CUSADI_ROOT_DIR, "benchmark", "results")

MAX_SHARED_BYTES = 101376  # L40S sm_89


def compile_sequential_single(fn_name, timeout_s=SEQ_COMPILE_TIMEOUT):
    """Compile one sequential kernel into build_sequential/. Returns (ok, seconds)."""
    os.makedirs(SEQ_BUILD_DIR, exist_ok=True)

    cu_rel = f"codegen/{fn_name}.cu"
    if not os.path.exists(os.path.join(CUSADI_ROOT_DIR, cu_rel)):
        return False, 0.0

    cmake_path = os.path.join(SEQ_BUILD_DIR, "CMakeLists.txt")
    var = fn_name.upper() + "_SOURCE"
    with open(cmake_path, "w") as f:
        f.write(textwrap.dedent(f"""\
            cmake_minimum_required(VERSION 3.15)
            project(CusADi_Seq LANGUAGES CXX CUDA)
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES 89)
            endif()
            set(CMAKE_CXX_STANDARD 11)
            set(CUDA_NVCC_FLAGS ${{CUDA_NVCC_FLAGS}}; -O3 --use_fast_math)
            include_directories(${{CMAKE_CURRENT_SOURCE_DIR}}/..)
            set({var} ../{cu_rel})
            add_library({fn_name} SHARED ${{{var}}})
        """))

    t0 = time.time()
    try:
        proc = subprocess.run(
            f"cd {SEQ_BUILD_DIR} && cmake . 2>&1 && make -j 2>&1",
            shell=True, capture_output=True, text=True, timeout=timeout_s)
        elapsed = time.time() - t0
        so_path = os.path.join(SEQ_BUILD_DIR, f"lib{fn_name}.so")
        if proc.returncode != 0 or not os.path.exists(so_path):
            print(f"  Compilation failed for {fn_name}:")
            print((proc.stdout or "")[-2000:])
            print((proc.stderr or "")[-2000:])
            return False, elapsed
        return True, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  TIMEOUT: {fn_name} sequential compilation exceeded {timeout_s}s")
        return False, elapsed


def ensure_lp_kernel_compiled():
    so_path = os.path.join(CUSADI_BUILD_DIR, "liblevel_parallel_kernel.so")
    if os.path.exists(so_path):
        return True
    os.makedirs(CUSADI_CODEGEN_DIR, exist_ok=True)
    cu_path = os.path.join(CUSADI_CODEGEN_DIR, "level_parallel_kernel.cu")
    write_kernel_source(cu_path)
    os.makedirs(CUSADI_BUILD_DIR, exist_ok=True)
    status = os.system(f"cd {CUSADI_BUILD_DIR} && cmake .. 2>&1 && make -j 2>&1")
    return status == 0


class SeqFunctionFromDir:
    """CusadiFunction-like wrapper loading .so from build_sequential/."""

    def __init__(self, fn_casadi, num_instances):
        self.fn_casadi = fn_casadi
        self.num_instances = num_instances

        lib_path = os.path.join(SEQ_BUILD_DIR, f"lib{fn_casadi.name()}.so")
        self._fn_library = ctypes.CDLL(lib_path)
        self._fn_library.evaluate.restype = ctypes.c_float

        n = num_instances
        self._input_tensors = [
            torch.zeros(n, fn_casadi.nnz_in(i), device="cuda",
                        dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_in())
        ]
        self._output_tensors = [
            torch.zeros(n, fn_casadi.nnz_out(i), device="cuda",
                        dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_out())
        ]
        self._work_tensor = torch.zeros(
            n, fn_casadi.sz_w(), device="cuda", dtype=torch.double).contiguous()

        self._input_ptrs = torch.zeros(
            fn_casadi.n_in(), device="cuda", dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(
            fn_casadi.n_out(), device="cuda", dtype=torch.int64).contiguous()
        for i in range(fn_casadi.n_in()):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(fn_casadi.n_out()):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self.eval_time = 0.0

    def evaluate(self, inputs):
        for i in range(self.fn_casadi.n_out()):
            self._output_tensors[i].zero_()
        self._work_tensor.zero_()

        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = inputs[i].data_ptr()

        fn_input = ctypes.cast(
            self._input_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_work = ctypes.cast(
            self._work_tensor.data_ptr(), ctypes.POINTER(ctypes.c_double))
        fn_output = ctypes.cast(
            self._output_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))

        self.eval_time = self._fn_library.evaluate(
            fn_input, fn_work, fn_output, self.num_instances)


def benchmark_kernel(fn_wrapper_factory, fn_casadi, batch_sizes):
    """Run benchmark across batch sizes. Returns {bs: median_seconds}."""
    results = {}
    for bs in batch_sizes:
        fn = fn_wrapper_factory(fn_casadi, bs)
        inputs = [
            torch.randn(bs, fn_casadi.nnz_in(i), device="cuda",
                        dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_in())
        ]

        for _ in range(N_WARMUP):
            fn.evaluate(inputs)
        torch.cuda.synchronize()

        times = []
        for _ in range(N_EVALS):
            fn.evaluate(inputs)
            times.append(fn.eval_time)
        torch.cuda.synchronize()

        results[bs] = float(np.median(times))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sequential", action="store_true")
    parser.add_argument("--skip-lp", action="store_true")
    parser.add_argument("--functions", nargs="+", default=FUNCTIONS)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    target_fns = args.functions
    batch_sizes = args.batch_sizes

    # Load CasADi functions
    casadi_fns = {}
    for name in target_fns:
        path = os.path.join(CUSADI_BENCHMARK_DIR, f"{name}.casadi")
        casadi_fns[name] = casadi.Function.load(path)
        print(f"Loaded {name}: {casadi_fns[name].n_instructions()} instructions")

    # --- DAG analysis ---
    print(f"\n{'='*90}")
    print("  DAG ANALYSIS")
    print(f"{'='*90}")

    dag_info = {}
    for name in target_fns:
        f = casadi_fns[name]
        plan = extract_level_plan(f)
        bs = pick_block_size(plan)
        smem = plan.shared_mem_bytes
        feasible = smem <= MAX_SHARED_BYTES

        dag_info[name] = {
            "instructions": f.n_instructions(),
            "compute_ops": plan.total_compute_ops,
            "levels": plan.n_compute_levels,
            "critical_path": plan.critical_path,
            "max_width": plan.max_width,
            "median_width": plan.median_width,
            "peak_live_slots": plan.peak_live_slots,
            "shared_mem_bytes": smem,
            "block_size": bs,
            "ceiling": plan.total_compute_ops / plan.critical_path,
            "lp_feasible": feasible,
        }

        status = "OK" if feasible else f"INFEASIBLE ({smem/1024:.0f}KB > {MAX_SHARED_BYTES/1024:.0f}KB)"
        print(f"\n  {name}: {f.n_instructions()} instrs, {plan.total_compute_ops} compute ops")
        print(f"    levels={plan.n_compute_levels}, crit_path={plan.critical_path}, "
              f"max_width={plan.max_width}, median_width={plan.median_width}")
        print(f"    peak_live={plan.peak_live_slots}, smem={smem/1024:.1f}KB, "
              f"block_size={bs}, ceiling={dag_info[name]['ceiling']:.1f}x")
        print(f"    LP status: {status}")

    # --- LP correctness validation ---
    lp_feasible = [n for n in target_fns if dag_info[n]["lp_feasible"]]

    print(f"\n{'='*90}")
    print("  LEVEL-PARALLEL CORRECTNESS VALIDATION")
    print(f"{'='*90}")

    for name in lp_feasible:
        f = casadi_fns[name]
        plan = extract_level_plan(f)
        ok = verify_level_plan_parallel(f, plan, n_tests=5)
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            dag_info[name]["lp_feasible"] = False

    for name in target_fns:
        if not dag_info[name]["lp_feasible"]:
            print(f"  {name}: SKIPPED (shared memory exceeds {MAX_SHARED_BYTES/1024:.0f}KB)")

    lp_feasible = [n for n in target_fns if dag_info[n]["lp_feasible"]]

    # --- Compile LP kernel ---
    if not args.skip_lp and lp_feasible:
        print(f"\n{'='*90}")
        print("  COMPILING LEVEL-PARALLEL KERNEL")
        print(f"{'='*90}")
        t0 = time.time()
        if not ensure_lp_kernel_compiled():
            print("FATAL: LP kernel compilation failed")
            sys.exit(1)
        lp_compile_time = time.time() - t0
        print(f"  LP kernel ready in {lp_compile_time:.1f}s (shared across all functions)")
    else:
        lp_compile_time = 0.0

    # --- Compile sequential kernels ---
    seq_compile_times = {}
    seq_compiled = set()

    if not args.skip_sequential:
        print(f"\n{'='*90}")
        print("  COMPILING SEQUENTIAL KERNELS")
        print(f"{'='*90}")
        for name in target_fns:
            f = casadi_fns[name]
            print(f"\n  Generating .cu for {name}...")
            t0 = time.time()
            generateCUDACodeDouble(f)
            gen_time = time.time() - t0
            print(f"    codegen: {gen_time:.1f}s")

            print(f"  Compiling {name} to build_sequential/...")
            ok, compile_time = compile_sequential_single(name)
            seq_compile_times[name] = compile_time
            if ok:
                seq_compiled.add(name)
                print(f"    compiled in {compile_time:.1f}s")
            else:
                if compile_time >= SEQ_COMPILE_TIMEOUT:
                    print(f"    TIMEOUT after {compile_time:.0f}s")
                else:
                    print(f"    FAILED after {compile_time:.1f}s")

    # --- Sequential benchmarks ---
    seq_results = {}
    if not args.skip_sequential and seq_compiled:
        print(f"\n{'='*90}")
        print("  SEQUENTIAL BENCHMARKS")
        print(f"{'='*90}")

        for name in target_fns:
            if name not in seq_compiled:
                print(f"\n  {name}: SKIPPED (not compiled)")
                continue
            f = casadi_fns[name]
            print(f"\n  Benchmarking SEQ: {name} ({f.n_instructions()} instrs)")

            results = benchmark_kernel(
                lambda fn, bs: SeqFunctionFromDir(fn, bs), f, batch_sizes)
            seq_results[name] = results

            for bs in batch_sizes:
                print(f"    batch={bs:>6d}  {results[bs]*1e6:>10.1f} us")

    # --- LP benchmarks ---
    lp_results = {}
    if not args.skip_lp and lp_feasible:
        print(f"\n{'='*90}")
        print("  LEVEL-PARALLEL BENCHMARKS")
        print(f"{'='*90}")

        for name in lp_feasible:
            f = casadi_fns[name]
            info = dag_info[name]
            print(f"\n  Benchmarking LP: {name} ({info['compute_ops']} ops, "
                  f"{info['levels']} levels, block={info['block_size']}, "
                  f"smem={info['shared_mem_bytes']/1024:.1f}KB)")

            results = benchmark_kernel(
                lambda fn, bs: CusadiLevelParallelFunction(
                    fn, bs, lib_name="level_parallel_kernel"),
                f, batch_sizes)
            lp_results[name] = results

            for bs in batch_sizes:
                print(f"    batch={bs:>6d}  {results[bs]*1e6:>10.1f} us")

    # --- Results table ---
    print(f"\n\n{'#'*110}")
    print("  COMPARISON TABLE")
    print(f"{'#'*110}")

    header = (f"{'Function':>8s} | {'Instrs':>7s} | {'CritPath':>8s} | "
              f"{'Ceiling':>7s} | {'Batch':>6s} | {'Sequential(us)':>14s} | "
              f"{'LevelPar(us)':>12s} | {'LP Speedup':>10s}")
    print(header)
    print("-" * len(header))

    csv_rows = []
    for name in target_fns:
        info = dag_info[name]
        for bs in batch_sizes:
            seq_t = seq_results.get(name, {}).get(bs)
            lp_t = lp_results.get(name, {}).get(bs)

            seq_str = f"{seq_t*1e6:.1f}" if seq_t is not None else "N/A"
            lp_str = f"{lp_t*1e6:.1f}" if lp_t is not None else "N/A"

            if seq_t is not None and lp_t is not None and lp_t > 0:
                speedup = seq_t / lp_t
                sp_str = f"{speedup:.2f}x"
            else:
                speedup = None
                sp_str = "N/A"

            print(f"{name:>8s} | {info['instructions']:>7d} | "
                  f"{info['critical_path']:>8d} | "
                  f"{info['ceiling']:>6.1f}x | {bs:>6d} | "
                  f"{seq_str:>14s} | {lp_str:>12s} | {sp_str:>10s}")

            csv_rows.append({
                "function": name,
                "instructions": info["instructions"],
                "compute_ops": info["compute_ops"],
                "critical_path": info["critical_path"],
                "ceiling": f"{info['ceiling']:.1f}",
                "max_width": info["max_width"],
                "median_width": info["median_width"],
                "shared_mem_kb": f"{info['shared_mem_bytes']/1024:.1f}",
                "block_size": info["block_size"],
                "batch_size": bs,
                "seq_median_us": f"{seq_t*1e6:.1f}" if seq_t is not None else "",
                "lp_median_us": f"{lp_t*1e6:.1f}" if lp_t is not None else "",
                "lp_speedup": f"{speedup:.2f}" if speedup is not None else "",
            })

    # --- Compilation time summary ---
    print(f"\n\n{'='*90}")
    print("  COMPILATION TIMES")
    print(f"{'='*90}")
    print(f"  LP kernel (fixed, all functions): {lp_compile_time:.1f}s")
    for name in target_fns:
        if name in seq_compile_times:
            t = seq_compile_times[name]
            status = ("OK" if name in seq_compiled
                      else "TIMEOUT" if t >= SEQ_COMPILE_TIMEOUT else "FAILED")
            print(f"  SEQ {name}: {t:.1f}s ({status})")

    # --- Save CSV ---
    csv_path = os.path.join(RESULTS_DIR, "paper_comparison.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Results saved to {csv_path}")

    compile_csv_path = os.path.join(RESULTS_DIR, "compilation_times.csv")
    with open(compile_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["function", "method", "compile_time_s", "status"])
        writer.writerow(["all", "level_parallel", f"{lp_compile_time:.1f}", "OK"])
        for name in target_fns:
            if name in seq_compile_times:
                t = seq_compile_times[name]
                status = ("OK" if name in seq_compiled
                          else "TIMEOUT" if t >= SEQ_COMPILE_TIMEOUT else "FAILED")
                writer.writerow([name, "sequential", f"{t:.1f}", status])
    print(f"  Compilation times saved to {compile_csv_path}")

    # --- Save markdown ---
    md_path = os.path.join(RESULTS_DIR, "paper_comparison.md")
    with open(md_path, "w") as f:
        f.write("# CusADi Paper Benchmark: Sequential vs Level-Parallel\n\n")
        f.write(f"GPU: L40S (sm_89), CUDA events timing, "
                f"median of {N_EVALS} evals after {N_WARMUP} warmup\n\n")

        f.write("## DAG Analysis\n\n")
        f.write("| Function | Instrs | Compute Ops | Levels | Crit Path | "
                "Max Width | Med Width | Shared Mem | Block Size | Ceiling | LP Feasible |\n")
        f.write("|----------|--------|-------------|--------|-----------|"
                "-----------|-----------|------------|------------|---------|-------------|\n")
        for name in target_fns:
            d = dag_info[name]
            f.write(f"| {name} | {d['instructions']} | {d['compute_ops']} | "
                    f"{d['levels']} | {d['critical_path']} | {d['max_width']} | "
                    f"{d['median_width']} | {d['shared_mem_bytes']/1024:.1f} KB | "
                    f"{d['block_size']} | {d['ceiling']:.1f}x | "
                    f"{'Yes' if d['lp_feasible'] else 'No'} |\n")

        f.write("\n## Compilation Times\n\n")
        f.write("| Function | Sequential (s) | Level-Parallel (s) |\n")
        f.write("|----------|---------------|--------------------|\n")
        for name in target_fns:
            seq_t = seq_compile_times.get(name)
            if seq_t is not None:
                sfx = ("" if name in seq_compiled
                       else " (TIMEOUT)" if seq_t >= SEQ_COMPILE_TIMEOUT
                       else " (FAILED)")
                seq_str = f"{seq_t:.1f}{sfx}"
            else:
                seq_str = "N/A"
            f.write(f"| {name} | {seq_str} | {lp_compile_time:.1f} (shared) |\n")

        f.write("\n## Performance Comparison\n\n")
        f.write("| Function | Instrs | Crit Path | Ceiling | Batch | "
                "Sequential (us) | Level-Parallel (us) | LP Speedup |\n")
        f.write("|----------|--------|-----------|---------|-------|"
                "-----------------|---------------------|------------|\n")
        for row in csv_rows:
            seq_str = row["seq_median_us"] if row["seq_median_us"] else "N/A"
            lp_str = row["lp_median_us"] if row["lp_median_us"] else "N/A"
            sp_str = (f"{float(row['lp_speedup']):.2f}x"
                      if row["lp_speedup"] else "N/A")
            f.write(f"| {row['function']} | {row['instructions']} | "
                    f"{row['critical_path']} | {row['ceiling']}x | "
                    f"{row['batch_size']} | {seq_str} | {lp_str} | {sp_str} |\n")

    print(f"  Markdown table saved to {md_path}")


if __name__ == "__main__":
    main()
