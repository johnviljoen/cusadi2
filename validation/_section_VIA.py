"""Table II — LP vs Sequential on paper benchmark functions fn_1e1..fn_1e5."""
from __future__ import annotations

import os

import numpy as np

from validation._helpers import (benchmark_kernel_only, build_lp_fn,
                                  load_casadi_fn, make_random_inputs,
                                  print_header, repo_path, SequentialFn,
                                  fmt_speedup)


FUNCTIONS = ["fn_1e1", "fn_1e2", "fn_1e3", "fn_1e4", "fn_1e5"]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SEQ_BUILD_DIR = repo_path("build_sequential")


def _load_fn_1e(fn_name: str):
    path = repo_path("src", "benchmark_functions", f"{fn_name}.casadi")
    return load_casadi_fn(path)


def _bench_lp_seq_at_batch(fn_casadi, batch: int, n_warmup: int, n_timed: int,
                           max_smem_slots: int = 12000):
    """Time LP (or tiled LP if it spills) and the precompiled sequential .so
    at this batch size. Returns (seq_us, lp_us, lp_mode)."""
    import torch
    inputs = make_random_inputs(fn_casadi, batch, seed=42 + batch)

    # Sequential .so — kernel-only time via cudaEventElapsedTime returned from .so
    seq = SequentialFn(fn_casadi, batch, SEQ_BUILD_DIR)
    seq.set_inputs(inputs)
    seq_us = benchmark_kernel_only(seq.evaluate, lambda: seq.eval_time,
                                    n_warmup, n_timed)

    # LP / tiled LP — same: kernel returns its own elapsed time
    lp, plan, lp_mode = build_lp_fn(fn_casadi, batch, max_smem_slots=max_smem_slots)
    lp_us = benchmark_kernel_only(lambda: lp.evaluate(inputs),
                                   lambda: lp.eval_time, n_warmup, n_timed)

    # Drop the LP instance so the next batch re-allocates the right N
    del lp, seq
    torch.cuda.empty_cache()
    return seq_us, lp_us, lp_mode


def run_section_VIA(n_warmup: int = 10, n_timed: int = 100) -> dict:
    print_header("SECTION VI-A / TABLE II : fn_1e1..fn_1e5 LP vs Sequential")
    print(f"  {'Function':<10} {'Instrs':>8} {'BestSpeedup':>12} "
          f"{'@Batch':>8} {'Batch=1':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*8} {'-'*10}")

    out: dict = {}
    for fn_name in FUNCTIONS:
        try:
            f = _load_fn_1e(fn_name)
        except FileNotFoundError as e:
            print(f"  {fn_name:<10} MISSING: {e}")
            out[fn_name] = {"error": "file_missing"}
            continue

        sweep = {}
        for batch in BATCH_SIZES:
            try:
                seq_us, lp_us, lp_mode = _bench_lp_seq_at_batch(
                    f, batch, n_warmup, n_timed)
                speedup = seq_us / lp_us if lp_us > 0 else float("nan")
                sweep[batch] = {"seq_us": seq_us, "lp_us": lp_us,
                                "speedup": speedup, "lp_mode": lp_mode}
            except Exception as e:
                sweep[batch] = {"error": str(e)}

        valid = [(b, r["speedup"]) for b, r in sweep.items()
                 if "speedup" in r and np.isfinite(r["speedup"])]
        if valid:
            best_batch, best_speedup = max(valid, key=lambda x: x[1])
        else:
            best_batch, best_speedup = None, float("nan")
        batch1_speedup = sweep.get(1, {}).get("speedup", float("nan"))

        print(f"  {fn_name:<10} {f.n_instructions():>8,} "
              f"{fmt_speedup(best_speedup):>12} {best_batch if best_batch else 'N/A':>8} "
              f"{fmt_speedup(batch1_speedup):>10}")

        out[fn_name] = {
            "instructions": int(f.n_instructions()),
            "best_speedup": float(best_speedup),
            "best_batch": int(best_batch) if best_batch else None,
            "batch1_speedup": float(batch1_speedup),
            "sweep": sweep,
        }
    return out
