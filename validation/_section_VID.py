"""Tables V / VI — MJX comparison (Constrained ABA Jacobian).

This section is the slowest: each (robot, contact_on/off, batch) MJX
configuration requires a JIT compile (30-160 s). Gated by --skip-mjx.

Strategy:
 1. Run our LP-constrained benchmark (fast, no JIT) for A1 and H1.
 2. For MJX: spawn the existing `bench_mjx_jacobian.py` in subprocesses
    (it has to run in a separate process per fp32/fp64/contact config),
    using smaller N_EVALS when --quick is set. Cache results in the
    reproduction JSON file.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import json

from validation._helpers import (benchmark_kernel_only, build_lp_fn,
                                  load_casadi_fn, make_random_inputs,
                                  print_header, repo_path)

ROBOTS = ["a1", "h1"]
BATCHES = [1, 8, 32, 128, 256]


def _bench_lp_constrained(robot: str, n_warmup: int, n_timed: int) -> dict:
    import torch
    fn_name = f"{robot}_constrained_aba_jac"
    try:
        f = load_casadi_fn(fn_name)
    except FileNotFoundError as e:
        print(f"  LP {robot}: MISSING ({e})")
        return {"error": "file_missing"}
    out = {}
    for bs in BATCHES:
        inputs = make_random_inputs(f, bs, seed=42)
        lp, _plan, mode = build_lp_fn(f, bs, max_smem_slots=12000)
        us = benchmark_kernel_only(lambda: lp.evaluate(inputs),
                                    lambda: lp.eval_time,
                                    n_warmup, n_timed)
        out[str(bs)] = {"median_us": us, "mode": mode}
        del lp
        torch.cuda.empty_cache()
    return out


def _run_mjx_subprocess(robot: str, with_contact: bool, out_path: str,
                       batches, n_warmup: int, n_evals: int,
                       timeout_s: int = 3600) -> dict:
    """Spawn bench_mjx_jacobian.py in a subprocess; load JSON output."""
    bench = repo_path("benchmark", "mjx-comparison", "bench_mjx_jacobian.py")
    cmd = [sys.executable, bench,
           "--robot", robot, "--dtype", "fp64",
           "--out", out_path,
           "--n_warmup", str(n_warmup),
           "--n_evals", str(n_evals),
           "--batch_sizes", *[str(b) for b in batches]]
    if with_contact:
        cmd.append("--with_contact")
    print(f"    [mjx] subprocess: {robot} contact={with_contact}  "
          f"batches={batches} n_evals={n_evals} ...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"    [mjx] FAILED ({elapsed:.0f} s)")
        print((r.stdout or "")[-800:])
        print((r.stderr or "")[-800:])
        return {"error": "subprocess_failed",
                "stdout_tail": (r.stdout or "")[-800:],
                "stderr_tail": (r.stderr or "")[-800:]}
    if not os.path.exists(out_path):
        return {"error": "no_output_file"}
    with open(out_path) as f:
        data = json.load(f)
    print(f"    [mjx] done ({elapsed:.0f} s)")
    return data


def run_section_VID(skip: bool = False, n_warmup: int = 10, n_timed: int = 100) -> dict:
    print_header("SECTION VI-D / TABLES V + VI : MJX Comparison (constrained)")

    result: dict = {"lp_constrained": {}, "mjx": {}, "table": {}}

    # Always run LP-constrained (fast)
    print("  [1/3] LP constrained ABA Jacobian (A1, H1)...")
    for robot in ROBOTS:
        result["lp_constrained"][robot] = _bench_lp_constrained(robot, n_warmup, n_timed)

    if skip:
        print("  MJX SKIPPED (--skip-mjx). Paper values are in paper_expected_values.json.")
        return result

    # MJX subprocess runs. Use smaller n_evals in quick mode (passed via outer).
    mjx_warmup = max(2, n_warmup // 2)
    mjx_evals = max(5, n_timed // 10)     # MJX median stabilizes fast; reduce overhead
    mjx_dir = repo_path("validation", "results", "mjx")
    os.makedirs(mjx_dir, exist_ok=True)

    for robot in ROBOTS:
        result["mjx"][robot] = {}
        for contact in (False, True):
            tag = "contact" if contact else "nocontact"
            out_path = os.path.join(mjx_dir, f"{robot}_{tag}_fp64.json")
            result["mjx"][robot][tag] = _run_mjx_subprocess(
                robot, contact, out_path, BATCHES, mjx_warmup, mjx_evals)

    # Stitch a table row per (robot, batch)
    for robot in ROBOTS:
        per = {}
        lp_side = result["lp_constrained"].get(robot, {})
        mjx_no = (result["mjx"].get(robot, {}).get("nocontact", {}) or {}).get("results", {})
        mjx_yes = (result["mjx"].get(robot, {}).get("contact", {}) or {}).get("results", {})
        for bs in BATCHES:
            lp_us = lp_side.get(str(bs), {}).get("median_us", float("nan"))
            mjx_nc_us = (mjx_no.get(str(bs), {}) or {}).get("median_s", float("nan")) * 1e6
            mjx_c_us = (mjx_yes.get(str(bs), {}) or {}).get("median_s", float("nan")) * 1e6
            ratio = mjx_c_us / lp_us if lp_us > 0 else float("nan")
            per[str(bs)] = {
                "lp_constrained_us": lp_us,
                "mjx_no_contact_us": mjx_nc_us,
                "mjx_with_contact_us": mjx_c_us,
                "ratio_mjx_contact_over_lp": ratio,
            }
        result["table"][robot] = per
        # Pretty print
        print(f"\n  {robot.upper()} constrained (LP vs MJX):")
        print(f"    {'Batch':>6} {'LP us':>12} {'MJX no-contact us':>20} "
              f"{'MJX contact us':>17} {'MJX/LP':>10}")
        for bs in BATCHES:
            r = per[str(bs)]
            print(f"    {bs:>6} {r['lp_constrained_us']:>12,.0f} "
                  f"{r['mjx_no_contact_us']:>20,.0f} "
                  f"{r['mjx_with_contact_us']:>17,.0f} "
                  f"{r['ratio_mjx_contact_over_lp']:>9.1f}x")
    return result
