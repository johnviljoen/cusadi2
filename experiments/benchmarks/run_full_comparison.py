"""
Full benchmark: sequential vs level-parallel (original + tiled) on all five
paper functions. Uses pre-compiled .so files from build_sequential/ and build/.

Usage:
    CUDA_VISIBLE_DEVICES=5 conda run -n cusadi2 python benchmark/run_full_comparison.py
"""

import csv
import ctypes
import os
import sys
import time

import casadi
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cusadi import CUSADI_ROOT_DIR, CUSADI_FUNCTION_DIR, CUSADI_BUILD_DIR
from cusadi.level_parallel.dag import extract_level_plan, extract_tiled_level_plan
from cusadi.level_parallel.codegen import pick_block_size
from cusadi.pytorch.lp_function import (CusadiLevelParallelFunction,
                                         CusadiTiledLevelParallelFunction)

FUNCTIONS = ["fn_1e1", "fn_1e2", "fn_1e3", "fn_1e4", "fn_1e5"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
               8192, 16384, 32768]
N_WARMUP = 10
N_EVALS = 100
SEQ_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build", "sequential")
RESULTS_DIR = os.path.join(CUSADI_ROOT_DIR, "benchmark", "results")
MAX_SMEM_SLOTS = 12000  # 96 KB for L40S

# Compilation times from earlier runs (sequential .so already built)
SEQ_COMPILE_TIMES = {
    "fn_1e1": 1.1, "fn_1e2": 1.1, "fn_1e3": 1.6,
    "fn_1e4": 10.8, "fn_1e5": 322.8,
}


class SeqFunction:
    """CusadiFunction wrapper loading .so from build_sequential/."""
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
            for i in range(fn_casadi.n_in())]
        self._output_tensors = [
            torch.zeros(n, fn_casadi.nnz_out(i), device="cuda",
                        dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_out())]
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
        fn_input = ctypes.cast(self._input_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_work = ctypes.cast(self._work_tensor.data_ptr(), ctypes.POINTER(ctypes.c_double))
        fn_output = ctypes.cast(self._output_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))
        self.eval_time = self._fn_library.evaluate(
            fn_input, fn_work, fn_output, self.num_instances)


def benchmark(fn_factory, fn_casadi, batch_sizes):
    results = {}
    for bs in batch_sizes:
        try:
            fn = fn_factory(fn_casadi, bs)
            inputs = [torch.randn(bs, fn_casadi.nnz_in(i), device="cuda",
                                   dtype=torch.double).contiguous()
                      for i in range(fn_casadi.n_in())]
            for _ in range(N_WARMUP):
                fn.evaluate(inputs)
            torch.cuda.synchronize()
            times = []
            for _ in range(N_EVALS):
                fn.evaluate(inputs)
                times.append(fn.eval_time)
            torch.cuda.synchronize()
            results[bs] = float(np.median(times))
        except (RuntimeError, Exception) as e:
            print(f"    SKIP batch={bs}: {e}")
            # Reset CUDA state after error
            torch.cuda.synchronize()
            results[bs] = None
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load functions and plans
    casadi_fns = {}
    dag_info = {}
    tiled_plans = {}

    for name in FUNCTIONS:
        f = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, f"{name}.casadi"))
        casadi_fns[name] = f

        untiled = extract_level_plan(f)
        tiled = extract_tiled_level_plan(f, max_smem_slots=MAX_SMEM_SLOTS)
        tiled_plans[name] = tiled

        dag_info[name] = {
            "instructions": f.n_instructions(),
            "compute_ops": tiled.total_compute_ops,
            "levels": tiled.n_compute_levels,
            "critical_path": untiled.n_compute_levels,
            "max_width": tiled.max_width,
            "median_width": tiled.median_width,
            "untiled_peak": untiled.peak_live_slots,
            "tiled_peak": tiled.peak_live_slots,
            "n_chunks": tiled.n_chunks,
            "n_sub_batch_spills": len(tiled.sub_batch_spills),
            "spill_stride": tiled.spill_buffer_stride,
            "block_size": pick_block_size(tiled),
        }

    # --- DAG analysis ---
    print(f"\n{'='*100}")
    print("  DAG ANALYSIS")
    print(f"{'='*100}")
    for name in FUNCTIONS:
        d = dag_info[name]
        print(f"\n  {name}: {d['instructions']} instrs, {d['compute_ops']} ops, "
              f"{d['critical_path']} levels")
        print(f"    max_width={d['max_width']}, untiled_peak={d['untiled_peak']} "
              f"({d['untiled_peak']*8/1024:.1f}KB)")
        print(f"    tiled: {d['n_chunks']} chunks, peak={d['tiled_peak']} "
              f"({d['tiled_peak']*8/1024:.1f}KB), "
              f"sub_batch_spills={d['n_sub_batch_spills']}, "
              f"spill_buf={d['spill_stride']*8/1024:.0f}KB/inst")

    # --- Benchmark all three kernels per function ---
    seq_results = {}
    orig_lp_results = {}
    tiled_results = {}

    for name in FUNCTIONS:
        f = casadi_fns[name]
        plan = tiled_plans[name]
        d = dag_info[name]

        print(f"\n{'='*100}")
        print(f"  BENCHMARKING: {name} ({f.n_instructions()} instrs)")
        print(f"{'='*100}")

        # Sequential
        torch.cuda.empty_cache()
        print(f"\n  Sequential...")
        r = benchmark(lambda fn, bs: SeqFunction(fn, bs), f, BATCH_SIZES)
        seq_results[name] = r

        # Original LP (fn_1e1..fn_1e4 only — fn_1e5 exceeds smem)
        if name != "fn_1e5":
            torch.cuda.empty_cache()
            print(f"  Original LP...")
            r = benchmark(
                lambda fn, bs: CusadiLevelParallelFunction(
                    fn, bs, lib_name="level_parallel_kernel"), f, BATCH_SIZES)
            orig_lp_results[name] = r

        # Tiled LP — cap batch sizes by memory for large spill buffers
        torch.cuda.empty_cache()
        spill_bytes_per_inst = d["spill_stride"] * 8
        io_bytes_per_inst = (sum(f.nnz_in(i) for i in range(f.n_in())) +
                             sum(f.nnz_out(i) for i in range(f.n_out()))) * 8
        bytes_per_inst = spill_bytes_per_inst + io_bytes_per_inst
        max_batch = BATCH_SIZES[-1]
        if spill_bytes_per_inst > 0:
            free_mem = torch.cuda.mem_get_info()[0]
            budget = min(free_mem - 4 * 1024**3, 30 * 1024**3)
            max_batch = min(max_batch, int(budget / bytes_per_inst))
        tiled_batch_sizes = [bs for bs in BATCH_SIZES if bs <= max_batch]
        print(f"  Tiled LP ({d['n_chunks']} chunks, max_batch={max_batch})...")
        r = benchmark(
            lambda fn, bs, p=plan: CusadiTiledLevelParallelFunction(
                fn, bs, tiled_plan=p), f, tiled_batch_sizes)
        tiled_results[name] = r

        # Summary for this function
        for bs in [1, 256, 4096, 32768]:
            s = seq_results[name].get(bs)
            o = orig_lp_results.get(name, {}).get(bs)
            t = tiled_results[name].get(bs)
            parts = [f"batch={bs:>6d}"]
            parts.append(f"seq={s*1e6:>8.1f}" if s else "seq=     N/A")
            parts.append(f"olp={o*1e6:>8.1f}" if o else "olp=     N/A")
            parts.append(f"tlp={t*1e6:>8.1f}" if t else "tlp=     N/A")
            print(f"    {'  '.join(parts)}")
        torch.cuda.empty_cache()

    # --- Full comparison table ---
    print(f"\n\n{'#'*130}")
    print("  FULL COMPARISON TABLE")
    print(f"{'#'*130}")

    header = (f"{'Function':>8s} | {'Instrs':>7s} | {'Chunks':>6s} | {'Batch':>6s} | "
              f"{'Sequential':>12s} | {'Orig LP':>10s} | {'Tiled LP':>10s} | "
              f"{'TLP/SEQ':>8s} | {'TLP/OLP':>8s}")
    print(header)
    print("-" * len(header))

    csv_rows = []
    for name in FUNCTIONS:
        d = dag_info[name]
        for bs in BATCH_SIZES:
            seq_t = seq_results[name].get(bs)
            orig_t = orig_lp_results.get(name, {}).get(bs)
            tiled_t = tiled_results[name].get(bs)

            seq_us = f"{seq_t*1e6:.1f}" if seq_t else "N/A"
            orig_us = f"{orig_t*1e6:.1f}" if orig_t else "N/A"
            tiled_us = f"{tiled_t*1e6:.1f}" if tiled_t else "N/A"

            tlp_seq = f"{seq_t/tiled_t:.2f}x" if seq_t and tiled_t and tiled_t > 0 else "N/A"
            tlp_olp = f"{orig_t/tiled_t:.2f}x" if orig_t and tiled_t and tiled_t > 0 else "N/A"

            print(f"{name:>8s} | {d['instructions']:>7d} | {d['n_chunks']:>6d} | "
                  f"{bs:>6d} | {seq_us:>12s} | {orig_us:>10s} | {tiled_us:>10s} | "
                  f"{tlp_seq:>8s} | {tlp_olp:>8s}")

            csv_rows.append({
                "function": name, "instructions": d["instructions"],
                "compute_ops": d["compute_ops"],
                "critical_path": d["critical_path"],
                "max_width": d["max_width"],
                "untiled_peak": d["untiled_peak"],
                "n_chunks": d["n_chunks"],
                "n_sub_batch_spills": d["n_sub_batch_spills"],
                "spill_stride": d["spill_stride"],
                "block_size": d["block_size"],
                "batch_size": bs,
                "seq_us": f"{seq_t*1e6:.1f}" if seq_t else "",
                "orig_lp_us": f"{orig_t*1e6:.1f}" if orig_t else "",
                "tiled_lp_us": f"{tiled_t*1e6:.1f}" if tiled_t else "",
                "tiled_vs_seq": f"{seq_t/tiled_t:.2f}" if seq_t and tiled_t and tiled_t > 0 else "",
                "tiled_vs_orig_lp": f"{orig_t/tiled_t:.2f}" if orig_t and tiled_t and tiled_t > 0 else "",
            })

    # --- Compilation times ---
    print(f"\n\n{'='*100}")
    print("  COMPILATION TIMES")
    print(f"{'='*100}")
    print(f"  {'Function':>8s} | {'Sequential (s)':>15s} | {'LP Tiled (s)':>13s} | {'Speedup':>8s}")
    print(f"  {'-'*8} | {'-'*15} | {'-'*13} | {'-'*8}")
    lp_compile = 1.5  # approximate, fixed kernel
    for name in FUNCTIONS:
        st = SEQ_COMPILE_TIMES[name]
        sp = st / lp_compile
        print(f"  {name:>8s} | {st:>14.1f}s | {lp_compile:>12.1f}s | {sp:>7.0f}x")

    # --- Save CSV ---
    csv_path = os.path.join(RESULTS_DIR, "full_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  CSV saved to {csv_path}")

    # --- Save markdown ---
    md_path = os.path.join(RESULTS_DIR, "full_comparison.md")
    with open(md_path, "w") as f:
        f.write("# CusADi Paper Benchmark: Sequential vs Level-Parallel (with Tiling)\n\n")
        f.write(f"GPU: L40S (sm_89), CUDA events, median of {N_EVALS} evals "
                f"after {N_WARMUP} warmup, fp64\n\n")

        f.write("## DAG Analysis\n\n")
        f.write("| Function | Instrs | Ops | Levels | Max Width | "
                "Untiled Peak | Chunks | Sub-batch Spills | "
                "Tiled Peak | Spill Buf/Inst |\n")
        f.write("|----------|--------|-----|--------|-----------|"
                "-------------|--------|------------------|-"
                "-----------|----------------|\n")
        for name in FUNCTIONS:
            d = dag_info[name]
            f.write(f"| {name} | {d['instructions']} | {d['compute_ops']} | "
                    f"{d['critical_path']} | {d['max_width']} | "
                    f"{d['untiled_peak']} ({d['untiled_peak']*8/1024:.0f}KB) | "
                    f"{d['n_chunks']} | {d['n_sub_batch_spills']} | "
                    f"{d['tiled_peak']} ({d['tiled_peak']*8/1024:.0f}KB) | "
                    f"{d['spill_stride']*8/1024:.0f}KB |\n")

        f.write("\n## Compilation Times\n\n")
        f.write("| Function | Sequential | LP Tiled | Speedup |\n")
        f.write("|----------|-----------|----------|---------|\n")
        for name in FUNCTIONS:
            st = SEQ_COMPILE_TIMES[name]
            f.write(f"| {name} | {st:.1f}s | {lp_compile:.1f}s | {st/lp_compile:.0f}x |\n")

        f.write("\n## Performance (microseconds, median)\n\n")
        f.write("| Function | Batch | Sequential | Orig LP | Tiled LP | "
                "TLP vs SEQ | TLP vs OLP |\n")
        f.write("|----------|-------|------------|---------|----------|"
                "-----------|------------|\n")
        for row in csv_rows:
            orig = row["orig_lp_us"] if row["orig_lp_us"] else "N/A"
            tvs = f"{float(row['tiled_vs_seq']):.2f}x" if row["tiled_vs_seq"] else "N/A"
            tvo = f"{float(row['tiled_vs_orig_lp']):.2f}x" if row["tiled_vs_orig_lp"] else "N/A"
            f.write(f"| {row['function']} | {row['batch_size']} | "
                    f"{row['seq_us']} | {orig} | {row['tiled_lp_us']} | "
                    f"{tvs} | {tvo} |\n")

    print(f"  Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
