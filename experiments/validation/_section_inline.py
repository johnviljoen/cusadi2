"""Inline paper claims:
  1) Analytical vs naive Jacobian: 3.1-4.7x fewer instructions on Panda/A1/H1.
  2) fn_1e5 tiled plan: chunks, sub-batches, peak smem/chunk.
  3) Single-sync kernel overhead reduction ~8% (skipped if 2-sync absent).
  4) MJX contact overhead 17x at A1 batch=128.
  5) LP contact overhead 6x at A1 batch=128.
  6) SRBM MPC DAG: ~497K instructions, ~6187 levels, median width ~7.
"""
from __future__ import annotations

import os

from experiments.validation._helpers import load_casadi_fn, print_header, repo_path


def _load_or_none(name_or_path: str):
    import casadi
    candidates = [
        name_or_path,
        repo_path("build", "casadi_functions", f"{name_or_path}.casadi"),
        repo_path("build", "casadi_functions", f"{name_or_path}.casadi"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return casadi.Function.load(c)
    return None


def run_inline_claims(measured: dict) -> dict:
    print_header("INLINE CLAIMS")
    out: dict = {}

    # 1) Analytical vs naive (tracked in repo as _jac vs _jac_analytical pairs)
    print("\n  [claim 1] Analytical vs naive Jacobian (instructions)")
    ratios = {}
    for robot in ("panda", "a1", "h1"):
        naive = _load_or_none(f"{robot}_aba_jac")              # ca.jacobian(aba)
        ana   = _load_or_none(f"{robot}_aba_jac_analytical")   # Pinocchio analytical
        if naive is None or ana is None:
            print(f"    {robot}: MISSING (one of {robot}_aba_jac[_analytical].casadi)")
            continue
        r = naive.n_instructions() / ana.n_instructions()
        ratios[robot] = {
            "naive_n_instr": int(naive.n_instructions()),
            "analytical_n_instr": int(ana.n_instructions()),
            "ratio": r,
        }
        print(f"    {robot:<6}: naive={naive.n_instructions():>8,}  "
              f"analytical={ana.n_instructions():>8,}  ratio={r:.2f}x")
    lo, hi = 3.1, 4.7
    vals = [r["ratio"] for r in ratios.values()]
    if vals:
        in_range = all(lo <= v <= hi for v in vals)
        print(f"    expected range: [{lo}, {hi}]  measured range: "
              f"[{min(vals):.2f}, {max(vals):.2f}]  "
              f"{'ok' if in_range else 'WARN'}")
    out["analytical_vs_naive"] = ratios

    # 2) fn_1e5 tiling
    print("\n  [claim 2] fn_1e5 tiled plan")
    from cusadi.level_parallel.dag import extract_tiled_level_plan
    f_1e5 = _load_or_none("fn_1e5")
    if f_1e5 is None:
        print("    fn_1e5 not found, skipping")
        out["fn_1e5_tiling"] = {"error": "missing"}
    else:
        tp = extract_tiled_level_plan(f_1e5, max_smem_slots=12000)
        # Introspect tiled plan for chunk stats
        n_chunks = int(getattr(tp, "n_chunks", 0))
        peak_kb = float(getattr(tp, "peak_chunk_live_slots",
                                 getattr(tp, "peak_live_slots", 0)) * 8 / 1024)
        sub_batches = int(getattr(tp, "n_sub_batches", 0))
        print(f"    chunks={n_chunks}  sub_batches={sub_batches}  peak_chunk={peak_kb:.1f} KB")
        out["fn_1e5_tiling"] = {
            "chunks": n_chunks,
            "sub_batches": sub_batches,
            "peak_chunk_kb": peak_kb,
        }

    # 3) Single-sync — skip per brief (2-sync kernel not retained)
    print("\n  [claim 3] Single-sync kernel (8% overhead reduction)")
    print("    SKIPPED: 2-sync kernel no longer in repo; claim from earlier benchmark.")
    out["single_sync"] = {"skipped": True, "reason": "2-sync variant not retained"}

    # 4) MJX contact overhead at A1 batch=128
    print("\n  [claim 4] MJX contact overhead at A1 batch=128 (expect 17x)")
    mjx = (measured.get("VI-D", {}).get("mjx", {}).get("a1", {}))
    nc = (mjx.get("nocontact", {}).get("results", {}) or {}).get("128", {}) or {}
    yc = (mjx.get("contact",   {}).get("results", {}) or {}).get("128", {}) or {}
    nc_us = nc.get("median_s", 0) * 1e6
    yc_us = yc.get("median_s", 0) * 1e6
    if nc_us > 0 and yc_us > 0:
        ratio = yc_us / nc_us
        print(f"    MJX no-contact={nc_us:,.0f} us   with-contact={yc_us:,.0f} us   "
              f"ratio={ratio:.1f}x")
        out["mjx_contact_overhead_a1_b128"] = ratio
    else:
        print("    VI-D MJX results unavailable (likely --skip-mjx).")
        out["mjx_contact_overhead_a1_b128"] = None

    # 5) LP contact overhead at A1 batch=128
    print("\n  [claim 5] LP contact overhead at A1 batch=128 (expect 6x)")
    lp_un = (measured.get("VI-C", {}).get("a1", {}).get("lp", {}).get(128, {})
             .get("median_us", float("nan")))
    lp_con = (measured.get("VI-D", {}).get("lp_constrained", {}).get("a1", {})
              .get("128", {}).get("median_us", float("nan")))
    if lp_un > 0 and lp_con > 0:
        r = lp_con / lp_un
        print(f"    LP unconstrained={lp_un:,.0f} us   constrained={lp_con:,.0f} us   "
              f"ratio={r:.1f}x")
        out["lp_contact_overhead_a1_b128"] = r
    else:
        print("    VI-C or VI-D LP batch=128 not available yet.")
        out["lp_contact_overhead_a1_b128"] = None

    # 6) SRBM MPC DAG
    print("\n  [claim 6] SRBM MPC DAG (expect ~497K instr, ~6187 levels, median width ~7)")
    out["srbm_mpc"] = _profile_srbm_mpc()

    return out


def _profile_srbm_mpc() -> dict:
    """Run the SRBM MPC profile in-process. Uses the harness in
    exploration/srbm_mpc_profile.py but returns the T=10 row only."""
    import sys
    import numpy as np
    sys.path.insert(0, repo_path("exploration"))
    try:
        from srbm_mpc_profile import build_one_sqp_iteration
    except ImportError as e:
        print(f"    SRBM profile module missing: {e}")
        return {"error": "missing"}
    from cusadi.level_parallel.dag import extract_level_plan
    # T=20 matches the paper's "497K, 6187 levels" ballpark; T=10 from earlier
    # harness gave 9298 levels. The paper's SRBM likely uses a different spec.
    # Report T=5, 8, 10 for context.
    rows = {}
    for T in (5, 8, 10):
        f = build_one_sqp_iteration(T)
        plan = extract_level_plan(f)
        widths = [lv.width for lv in plan.compute_levels]
        row = {
            "T": T,
            "n_instructions": int(f.n_instructions()),
            "levels": int(plan.n_compute_levels),
            "median_width": float(np.median(widths)),
            "mean_width": float(sum(widths) / max(1, len(widths))),
            "peak_live_slots": int(plan.peak_live_slots),
        }
        rows[f"T{T}"] = row
        print(f"    T={T}:  n_instr={row['n_instructions']:,}  "
              f"levels={row['levels']}  median_width={row['median_width']:.1f}")
    return rows
