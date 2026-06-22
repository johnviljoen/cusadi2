"""Table IX — DAG ceiling vs achieved LP speedup at batch=1.

Ceiling = n_ops / critical_path  (theoretical upper bound).
Achieved = sequential_us(batch=1) / lp_us(batch=1)  (measured).

Depends on VI-A and VI-C having run — pulls their batch=1 numbers.
"""
from __future__ import annotations

from experiments.validation._helpers import load_casadi_fn, print_header, repo_path


TARGETS = [
    # (paper-key,  casadi-fn-name,  source-section-in-measured-results)
    ("fn_1e3", "fn_1e3",                    "VI-A"),
    ("fn_1e4", "fn_1e4",                    "VI-A"),
    ("h1_aba", "h1_aba_jac_analytical",     "VI-C"),
]


def _compute_ceiling(fn_name: str) -> float:
    import os
    from cusadi.level_parallel.dag import extract_level_plan
    # All generated functions (robot fixtures + fn_1eX) live in build/casadi_functions
    candidates = [
        repo_path("build", "casadi_functions", f"{fn_name}.casadi"),
    ]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        return float("nan")
    f = load_casadi_fn(path)
    plan = extract_level_plan(f)
    total_ops = sum(lv.width for lv in plan.compute_levels)
    critical_path = max(1, plan.n_compute_levels)
    return total_ops / critical_path


def _get_achieved_batch1(measured: dict, key: str, source: str) -> float:
    """Pull the batch=1 speedup from earlier sections' results."""
    import math
    if source == "VI-A":
        sweep = measured.get("VI-A", {}).get(key, {}).get("sweep", {})
        row = sweep.get(1, {}) or sweep.get("1", {})
        sp = row.get("speedup", float("nan"))
    elif source == "VI-C":
        robot = "h1" if key.startswith("h1") else "a1"
        per_robot = measured.get("VI-C", {}).get(robot, {})
        lp = per_robot.get("lp", {}).get(1, {}).get("median_us", float("nan"))
        seq = per_robot.get("seq", {}).get(1, {}).get("median_us", float("nan"))
        sp = seq / lp if lp > 0 else float("nan")
    else:
        sp = float("nan")
    return float(sp) if isinstance(sp, (int, float)) and math.isfinite(sp) else float("nan")


def run_section_VIG(measured: dict, paper: dict) -> dict:
    print_header("SECTION VI-G / TABLE IX : DAG ceiling vs achieved (batch=1)")
    print(f"  {'Function':<10} {'Ceiling':>10} {'Achieved':>12} {'Fraction':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    out = {}
    for key, fn_name, source in TARGETS:
        ceiling = _compute_ceiling(fn_name)
        achieved = _get_achieved_batch1(measured, key, source)
        frac_pct = achieved / ceiling * 100.0 if ceiling > 0 else float("nan")
        print(f"  {key:<10} {ceiling:>9.1f}x {achieved:>11.2f}x {frac_pct:>9.1f}%")
        out[key] = {
            "ceiling": ceiling,
            "achieved_batch1": achieved,
            "fraction_pct": frac_pct,
        }
    return out
