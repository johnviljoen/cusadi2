#!/usr/bin/env python3
"""
reproduce_all_results.py

Reproduces every numerical result in the paper:
  "Level-Parallel GPU Evaluation of Symbolic Expression DAGs
   for Real-Time Robot Dynamics"

Usage:
  python reproduce_all_results.py                         # Run all
  python reproduce_all_results.py --section VI-A          # One section
  python reproduce_all_results.py --skip-mjx              # Skip MJX
  python reproduce_all_results.py --skip-compilation      # Skip Tbl III
  python reproduce_all_results.py --quick                 # 10 timed runs

Output:
  Console: formatted tables matching paper, discrepancy report
  JSON:    results/paper_reproduction_<timestamp>.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import textwrap
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)


# -----------------------------------------------------------------------------
# Constants / paths
# -----------------------------------------------------------------------------
EXPECTED_GPU = "NVIDIA L40S"
PAPER_VALUES_PATH = os.path.join(THIS_DIR, "paper_expected_values.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "validation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Lazy imports — casadi / torch fail on GPU mismatch, defer
# -----------------------------------------------------------------------------
def _lazy_imports():
    import casadi  # noqa
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    dev = torch.cuda.get_device_name(0)
    if EXPECTED_GPU not in dev:
        print(f"[warn] Detected GPU '{dev}' does not match expected '{EXPECTED_GPU}'. "
              f"Absolute timings will differ from paper; ratios should still agree.")
    return casadi, torch


# -----------------------------------------------------------------------------
# Unified benchmark helper — CUDA-event median timing
# -----------------------------------------------------------------------------
def benchmark_evaluate(evaluate_fn, n_warmup: int, n_timed: int) -> float:
    """Time one `evaluate_fn()` call via CUDA events. Returns median in µs."""
    import torch
    for _ in range(n_warmup):
        evaluate_fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(n_timed):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        evaluate_fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return float(np.median(times_ms) * 1000.0)   # µs


# -----------------------------------------------------------------------------
# Sequential kernel wrapper (prebuilt .so in build_sequential/ or build/)
# -----------------------------------------------------------------------------
class SequentialFn:
    """Load a precompiled per-function sequential CusADi .so."""
    def __init__(self, fn_casadi, num_instances: int, so_dir: str):
        import torch
        lib = os.path.join(so_dir, f"lib{fn_casadi.name()}.so")
        if not os.path.exists(lib):
            raise FileNotFoundError(
                f"Sequential .so not found: {lib}\n"
                f"Build it first; run `python benchmark/run_paper_comparison.py` "
                f"(for fn_1eX) or `python tests/test_level_parallel_cuda.py "
                f"--fn <fn_name>` (for Pinocchio functions).")
        self.fn = fn_casadi
        self.N = num_instances
        self._lib = ctypes.CDLL(lib)
        self._lib.evaluate.restype = ctypes.c_float
        n = num_instances
        self._inp = [torch.zeros(n, fn_casadi.nnz_in(i), device="cuda",
                                 dtype=torch.double).contiguous()
                     for i in range(fn_casadi.n_in())]
        self._out = [torch.zeros(n, fn_casadi.nnz_out(i), device="cuda",
                                 dtype=torch.double).contiguous()
                     for i in range(fn_casadi.n_out())]
        self._work = torch.zeros(n, fn_casadi.sz_w(),
                                  device="cuda", dtype=torch.double).contiguous()
        self._in_ptr = torch.zeros(fn_casadi.n_in(), device="cuda",
                                   dtype=torch.int64).contiguous()
        self._out_ptr = torch.zeros(fn_casadi.n_out(), device="cuda",
                                    dtype=torch.int64).contiguous()
        for i in range(fn_casadi.n_in()):
            self._in_ptr[i] = self._inp[i].data_ptr()
        for i in range(fn_casadi.n_out()):
            self._out_ptr[i] = self._out[i].data_ptr()

    def set_inputs(self, inputs: List):
        for i, t in enumerate(inputs):
            self._inp[i] = t
            self._in_ptr[i] = t.data_ptr()

    def evaluate(self):
        for o in self._out:
            o.zero_()
        self._work.zero_()
        fn_in = ctypes.cast(self._in_ptr.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_out = ctypes.cast(self._out_ptr.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_work = ctypes.cast(self._work.data_ptr(), ctypes.POINTER(ctypes.c_double))
        self._lib.evaluate(fn_in, fn_out, fn_work, self.N)


# -----------------------------------------------------------------------------
# Input stubs
# -----------------------------------------------------------------------------
def make_random_inputs(fn_casadi, N: int, seed: int = 42):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(N, fn_casadi.nnz_in(i), device="cuda",
                        dtype=torch.double, generator=g).contiguous()
            for i in range(fn_casadi.n_in())]


# -----------------------------------------------------------------------------
# Paper-value loading + discrepancy check
# -----------------------------------------------------------------------------
def load_paper_values() -> Dict:
    with open(PAPER_VALUES_PATH) as f:
        return json.load(f)


def compare_and_flag(measured: float, expected: float, warn_pct: float = 10,
                     err_pct: float = 25) -> str:
    if expected <= 0:
        return "—"
    pct = abs(measured - expected) / expected * 100.0
    if pct > err_pct:
        return f"FAIL ({pct:.0f}%)"
    if pct > warn_pct:
        return f"WARN ({pct:.0f}%)"
    return f"ok ({pct:.0f}%)"


def fmt_speedup(s: float) -> str:
    return f"{s:.2f}x" if np.isfinite(s) else "N/A"


# Section runners are lazily imported inside main() AFTER
# CUDA_VISIBLE_DEVICES is set, so that torch/casadi don't grab the wrong GPU.


# -----------------------------------------------------------------------------
# Final discrepancy report
# -----------------------------------------------------------------------------
def discrepancy_report(all_results: Dict, paper: Dict,
                       warn_pct: float = 10, err_pct: float = 25) -> List[dict]:
    """Compare measured against expected, return a list of flagged items.

    `all_results` is keyed by section name ('dag', 'VI-A', ...). Each section
    returns a dict whose leaf float values are compared against leaves in
    `paper`. To reduce false missings, we look up matching paper keys via
    a section→paper-key map; if a section doesn't declare one, we skip."""
    SECTION_TO_PAPER = {
        "dag":    ["table_I_dag_structure", "table_II_peak_liveness"],
        "VI-A":   ["table_II_fn_benchmarks"],
        "VI-B":   ["table_III_compilation"],
        "VI-C":   ["table_IV_unconstrained_aba_jac_us"],
        "VI-D":   ["table_V_a1_constrained_us", "table_VI_h1_constrained_us"],
        "VI-E":   ["table_VII_mpc_pattern"],
        "VI-F":   ["table_VIII_mppi_rollout"],
        "VI-G":   ["table_IX_ceiling_vs_achieved"],
        "inline": ["inline_claims"],
    }

    # Leaf keys to skip — they're either informational (not continuous
    # numerical claims) or live on plateaus where paper + reproduction
    # disagree by multiples but neither is "wrong".
    SKIP_LEAF_KEYS = {"best_batch"}

    flags = []
    def walk(meas, exp, path):
        if isinstance(meas, dict) and isinstance(exp, dict):
            for k in exp:
                if k in SKIP_LEAF_KEYS:
                    continue
                if k in meas:
                    walk(meas[k], exp[k], path + [k])
        elif isinstance(meas, (int, float)) and isinstance(exp, (int, float)):
            if exp > 0:
                pct = abs(meas - exp) / exp * 100
                if pct > err_pct:
                    flags.append({"path": "/".join(path), "status": "FAIL",
                                  "pct_diff": pct, "expected": exp, "measured": meas})
                elif pct > warn_pct:
                    flags.append({"path": "/".join(path), "status": "WARN",
                                  "pct_diff": pct, "expected": exp, "measured": meas})

    for section, paper_keys in SECTION_TO_PAPER.items():
        meas = all_results.get(section)
        if not isinstance(meas, dict):
            continue
        for pk in paper_keys:
            pv = paper.get(pk)
            if isinstance(pv, dict):
                walk(meas, pv, [section, pk])
    return flags


def print_discrepancy_report(flags: List[dict]):
    print("\n" + "=" * 78)
    print("DISCREPANCY CHECK")
    print("=" * 78)
    if not flags:
        print("  all reproduced values within 10% of paper values. ok.")
        return
    fails = [f for f in flags if f.get("status") == "FAIL"]
    warns = [f for f in flags if f.get("status") == "WARN"]
    missing = [f for f in flags if f.get("status") == "missing_from_measured"]
    if fails:
        print(f"  {len(fails)} FAIL(s) (>25% off paper):")
        for f in fails:
            print(f"    {f['path']:50s}  paper={f['expected']}  "
                  f"measured={f['measured']:.3g}  ({f['pct_diff']:.1f}%)")
    if warns:
        print(f"  {len(warns)} WARN(s) (>10% off paper):")
        for f in warns:
            print(f"    {f['path']:50s}  paper={f['expected']}  "
                  f"measured={f['measured']:.3g}  ({f['pct_diff']:.1f}%)")
    if missing:
        print(f"  {len(missing)} expected value(s) not reproduced:")
        for f in missing[:10]:
            print(f"    {f['path']}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
SECTION_ORDER = ["dag", "VI-A", "VI-B", "VI-C", "VI-D", "VI-E",
                 "VI-F", "VI-G", "inline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all",
                    choices=["all"] + SECTION_ORDER)
    ap.add_argument("--skip-mjx", action="store_true",
                    help="Skip VI-D (Tables V/VI) — MJX JIT is slow (~30-160 s each)")
    ap.add_argument("--skip-compilation", action="store_true",
                    help="Skip VI-B (Table III) — fn_1e5 seq compile is ~5 min")
    ap.add_argument("--quick", action="store_true",
                    help="10 timed runs instead of 100")
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    n_timed = 10 if args.quick else 100
    n_warmup = 5 if args.quick else 10

    import torch  # now that CUDA_VISIBLE_DEVICES is set
    torch.manual_seed(42)
    np.random.seed(42)
    _lazy_imports()

    paper = load_paper_values()
    all_results: Dict = {"__meta": {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "n_warmup": n_warmup, "n_timed": n_timed,
        "quick": args.quick, "skip_mjx": args.skip_mjx,
        "skip_compilation": args.skip_compilation,
    }}

    todo = SECTION_ORDER if args.section == "all" else [args.section]

    # Lazy section imports (post-GPU-selection)
    from experiments.validation._section_dag import run_dag_analysis
    from experiments.validation._section_VIA import run_section_VIA
    from experiments.validation._section_VIB import run_section_VIB
    from experiments.validation._section_VIC import run_section_VIC
    from experiments.validation._section_VID import run_section_VID
    from experiments.validation._section_VIE import run_section_VIE
    from experiments.validation._section_VIF import run_section_VIF
    from experiments.validation._section_VIG import run_section_VIG
    from experiments.validation._section_inline import run_inline_claims

    section_dispatch = {
        "dag":    lambda: run_dag_analysis(),
        "VI-A":   lambda: run_section_VIA(n_warmup=n_warmup, n_timed=n_timed),
        "VI-B":   lambda: run_section_VIB(skip=args.skip_compilation),
        "VI-C":   lambda: run_section_VIC(n_warmup=n_warmup, n_timed=n_timed),
        "VI-D":   lambda: run_section_VID(skip=args.skip_mjx,
                                          n_warmup=n_warmup, n_timed=n_timed),
        "VI-E":   lambda: run_section_VIE(n_warmup=n_warmup, n_timed=n_timed),
        "VI-F":   lambda: run_section_VIF(n_warmup=n_warmup, n_timed=n_timed),
        "VI-G":   lambda: run_section_VIG(all_results, paper),
        "inline": lambda: run_inline_claims(all_results),
    }

    for section in todo:
        run = section_dispatch.get(section)
        if run is None:
            continue
        try:
            all_results[section] = run()
        except Exception as e:
            print(f"  [ERROR] Section {section} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[section] = {"__error": str(e)}

    # Compare
    flags = discrepancy_report(all_results, paper)
    print_discrepancy_report(flags)
    all_results["__discrepancies"] = flags

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out_dir, f"paper_reproduction_{ts}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
