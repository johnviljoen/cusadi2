"""
Join MJX / LP / sequential / mujoco-warp benchmark JSONs into a comparison table.

File naming convention the driver (run_all.sh) uses:
  <robot>_mjx_<dtype>.json            MJX Jacobian sweep
  <robot>_lp_fp64.json                LP Jacobian
  <robot>_seq_fp64.json               Sequential CusADi Jacobian
  <robot>_fwd_<backend>[_<dtype>].json  Forward dynamics

Outputs a Markdown table to stdout.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any


# robot, backend, dtype, is_fwd
def classify(path: str, d: dict) -> tuple[str, str, str, bool]:
    name = os.path.basename(path)
    robot = d.get("robot", "?")
    if "_fwd_" in name:
        is_fwd = True
        # pattern: <robot>_fwd_<backend>[_<dtype>].json
        stem = name.replace(".json", "").split("_fwd_", 1)[1]
        parts = stem.split("_")
        # backend can be level_parallel, sequential, mjx, mujoco_warp
        if parts[:2] == ["level", "parallel"]:
            backend = "level_parallel"
            dtype = "fp64"
        elif parts[0] == "mjx":
            backend = "mjx"
            dtype = parts[1] if len(parts) > 1 else d.get("dtype", "fp32")
        elif parts[:2] == ["mujoco", "warp"] or parts[0] in ("mjw",):
            backend = "mujoco_warp"
            dtype = "fp32"
        elif parts[0] == "sequential" or parts[:1] == ["seq"]:
            backend = "sequential"
            dtype = "fp64"
        else:
            backend = d.get("backend", parts[0])
            dtype = d.get("dtype", "fp64")
    else:
        is_fwd = False
        # <robot>_<backend>_<dtype>.json
        stem = name.replace(".json", "")
        parts = stem.split("_")
        # parts[0] = robot, parts[1] = backend, parts[2] = dtype
        if len(parts) >= 3:
            backend_raw = parts[1]
            dtype = parts[2]
            if backend_raw == "lp":
                backend = "level_parallel"
            elif backend_raw == "seq":
                backend = "sequential"
            elif backend_raw == "mjx":
                backend = "mjx"
            else:
                backend = backend_raw
        else:
            backend = d.get("backend", d.get("method", "?"))
            dtype = d.get("dtype", "?")

    return (robot, backend, dtype, is_fwd)


def load_all(results_dir: str) -> list[tuple[tuple, dict]]:
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(p) == "combined.json":
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"[compare] skip {p}: {e}")
            continue
        key = classify(p, d)
        d["__path__"] = p
        out.append((key, d))
    return out


def get_median(d: dict, batch: int) -> float | None:
    r = d.get("results", {})
    k = batch if batch in r else str(batch)
    if k not in r:
        return None
    return r[k].get("median_s")


def get_first_call(d: dict, batch: int) -> float | None:
    r = d.get("results", {})
    k = batch if batch in r else str(batch)
    return r.get(k, {}).get("jit_compile_plus_run_s")


def format_us(s: float | None) -> str:
    if s is None:
        return "—"
    v = s * 1e6
    if v >= 10000:
        return f"{v:,.0f}"
    elif v >= 100:
        return f"{v:,.1f}"
    else:
        return f"{v:.1f}"


def build_table(buckets: dict[tuple, dict], batches: list[int], cols: list[tuple]) -> list[str]:
    """cols: list of (header, backend, dtype) where a lookup is buckets.get((backend, dtype), None)."""
    lines = []
    header = ["batch"] + [c[0] for c in cols]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")
    # Median time table
    for bs in batches:
        row = [str(bs)]
        for hdr, back, dt in cols:
            d = buckets.get((back, dt))
            row.append(format_us(get_median(d, bs) if d is not None else None))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_ratio_table(buckets: dict[tuple, dict], batches: list[int],
                      base: tuple, competitors: list[tuple]) -> list[str]:
    """Ratio (competitor / base), so >1 means competitor is slower than base (LP)."""
    lines = []
    base_d = buckets.get(base)
    if base_d is None:
        return ["(no base data — skipping ratio table)"]
    header = ["batch"] + [f"{c[0]} / {base[0]}-{base[1]}" for c in competitors]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")
    for bs in batches:
        row = [str(bs)]
        base_v = get_median(base_d, bs)
        for hdr, back, dt in [(c[0] + "-" + c[1], c[0], c[1]) for c in competitors]:
            d = buckets.get((back, dt))
            v = get_median(d, bs) if d is not None else None
            if v is None or base_v is None or base_v == 0:
                row.append("—")
            else:
                row.append(f"{v / base_v:.1f}x")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="benchmark/mjx-comparison/results")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128, 256, 512])
    args = ap.parse_args()

    loaded = load_all(args.results_dir)
    print(f"<!-- Loaded {len(loaded)} JSON files from {args.results_dir} -->\n")

    for robot in ("a1", "h1"):
        # JACOBIAN section ------------------------------------------------
        jac_buckets = {
            (b, dt): d for (r, b, dt, is_fwd), d in loaded
            if r == robot and not is_fwd
        }
        if jac_buckets:
            print(f"\n## {robot.upper()} — ABA Jacobian (ddq/dq, ddq/dv, ddq/dtau)\n")
            print("Median time per evaluation in µs.\n")
            cols = [
                ("LP (fp64)", "level_parallel", "fp64"),
                ("Seq (fp64)", "sequential", "fp64"),
                ("MJX (fp32)", "mjx", "fp32"),
                ("MJX (fp64)", "mjx", "fp64"),
            ]
            for line in build_table(jac_buckets, args.batches, cols):
                print(line)

            print(f"\n**Ratio to LP fp64** (>1 means the backend is slower than LP):\n")
            ratio_cols = [
                ("seq", "fp64"), ("mjx", "fp32"), ("mjx", "fp64"),
            ]
            ratio_cols_hdr = [
                ("Seq", "fp64"), ("MJX", "fp32"), ("MJX", "fp64"),
            ]
            for line in build_ratio_table(jac_buckets, args.batches,
                                          ("level_parallel", "fp64"),
                                          ratio_cols_hdr):
                print(line)

            # JIT / compile
            print(f"\n**First-call compile-and-run time** (s; one-off per batch size for MJX):\n")
            mjx_b = jac_buckets.get(("mjx", "fp64")) or jac_buckets.get(("mjx", "fp32"))
            if mjx_b:
                print("| batch | MJX first-call (s) |")
                print("|---:|---:|")
                for bs in args.batches:
                    v = get_first_call(mjx_b, bs)
                    print(f"| {bs} | {v:.1f} |" if v is not None else f"| {bs} | — |")
            print("\nLP data-driven kernel compiles once in ~1.5 s and is reused across all functions. "
                  "Sequential CusADi kernels each require a one-off NVCC compile (20–60 s).")

        # FORWARD DYNAMICS section ---------------------------------------
        fwd_buckets = {
            (b, dt): d for (r, b, dt, is_fwd), d in loaded
            if r == robot and is_fwd
        }
        if fwd_buckets:
            print(f"\n## {robot.upper()} — Forward dynamics only (ABA, no Jacobian)\n")
            cols = [
                ("LP (fp64)", "level_parallel", "fp64"),
                ("Seq (fp64)", "sequential", "fp64"),
                ("MJX (fp32)", "mjx", "fp32"),
                ("MJX (fp64)", "mjx", "fp64"),
                ("mujoco-warp (fp32)", "mujoco_warp", "fp32"),
            ]
            for line in build_table(fwd_buckets, args.batches, cols):
                print(line)


if __name__ == "__main__":
    main()
