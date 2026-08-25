"""Tables I + II — DAG structure and peak liveness.

All values come from `extract_level_plan()` / `extract_tiled_level_plan()`,
which is pure Python (no GPU). Runs in seconds.
"""
from __future__ import annotations

import numpy as np

from experiments.validation._helpers import load_casadi_fn, print_header, fmt_vs_paper


DAG_TARGETS = [
    # (label,           casadi_fn_name,                 contact_desc)
    ("panda",            "panda_aba_jac_analytical",     "—"),
    ("a1_unconstrained", "a1_aba_jac_analytical",        "—"),
    ("a1_constrained",   "a1_constrained_aba_jac",       "4×3D"),
    ("h1_unconstrained", "h1_aba_jac_analytical",        "—"),
    ("h1_constrained",   "h1_constrained_aba_jac",       "2×6D"),
]


def run_dag_analysis() -> dict:
    from cusadi.level_parallel.dag import extract_level_plan, extract_tiled_level_plan

    print_header("TABLE I + II : DAG structure and peak liveness")
    print(f"  {'Model':<20} {'Contact':<8} {'Instrs':>10} {'CritPath':>10} "
          f"{'MeanW':>8} {'Ceiling':>10} {'PeakLive':>10} {'Smem(KB)':>10} {'Fits?':<6}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    results = {}
    for label, fn_name, contact in DAG_TARGETS:
        try:
            f = load_casadi_fn(fn_name)
        except FileNotFoundError as e:
            print(f"  {label:<20} MISSING: {e}")
            results[label] = {"error": "file_missing", "fn_name": fn_name}
            continue

        plan = extract_level_plan(f)
        widths = np.array([lv.width for lv in plan.compute_levels])
        total_ops = int(widths.sum())
        n_levels = int(plan.n_compute_levels)
        mean_width = float(total_ops / max(n_levels, 1))
        ceiling = mean_width  # for this paper ceiling == mean width
        peak_live = int(plan.peak_live_slots)
        smem_kb = peak_live * 8 / 1024
        fits = "yes" if peak_live * 8 <= 96 * 1024 else "tiled"

        tiled_info = None
        if fits == "tiled":
            # Build tiled plan to also surface chunk metadata
            try:
                tp = extract_tiled_level_plan(f, max_smem_slots=12000)
                tiled_info = {
                    "n_chunks": int(getattr(tp, "n_chunks", 0)),
                    "peak_chunk_kb": float(getattr(tp, "peak_chunk_live_slots",
                                                    peak_live) * 8 / 1024),
                }
            except Exception as e:
                tiled_info = {"error": str(e)}

        print(f"  {label:<20} {contact:<8} {f.n_instructions():>10,} "
              f"{n_levels:>10} {mean_width:>8.0f} {ceiling:>9.0f}x "
              f"{peak_live:>10,} {smem_kb:>10.1f} {fits:<6}")

        # Keep paper-schema fields at top level for the discrepancy walker
        # (paper table I keys: instructions, critical_path, mean_width, ceiling
        #  paper table II keys: nodes, peak_live, shared_mem_kb, fits).
        results[label] = {
            "fn_name": fn_name,
            "contact": contact,
            "instructions": int(f.n_instructions()),
            "nodes": int(f.n_instructions()),
            "dag_ops": total_ops,
            "critical_path": n_levels,
            "mean_width": mean_width,
            "ceiling": mean_width,
            "peak_live": peak_live,
            "shared_mem_kb": smem_kb,
            "fits_untiled": peak_live * 8 <= 96 * 1024,
            "median_width": float(np.median(widths)),
            "max_width": int(widths.max()),
            "tiled": tiled_info,
        }

    return results
