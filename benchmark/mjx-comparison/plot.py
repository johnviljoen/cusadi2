"""
Plot LP vs Seq vs MJX (fp32 / fp64) for each robot, both on a shared log axis.

Outputs PNG files under benchmark/mjx-comparison/results/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sweep(path: str) -> tuple[list[int], list[float]]:
    with open(path) as f:
        d = json.load(f)
    r = d["results"]
    bs = sorted(int(k) for k in r.keys())
    med = [r[str(b)]["median_s"] * 1e6 for b in bs]
    return bs, med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="benchmark/mjx-comparison/results")
    ap.add_argument("--out_dir", default="benchmark/mjx-comparison/results")
    args = ap.parse_args()

    for robot in ("a1", "h1"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        any_plotted = False
        curves = [
            (f"{robot}_lp_fp64.json",   "LP (fp64)",         "C0", "o-"),
            (f"{robot}_seq_fp64.json",  "Sequential (fp64)", "C1", "s-"),
            (f"{robot}_mjx_fp32.json",  "MJX (fp32)",        "C2", "^-"),
            (f"{robot}_mjx_fp64.json",  "MJX (fp64)",        "C3", "v-"),
        ]
        for fname, label, color, style in curves:
            p = os.path.join(args.results_dir, fname)
            if not os.path.exists(p):
                continue
            bs, med = load_sweep(p)
            ax.plot(bs, med, style, color=color, label=label, linewidth=1.5, markersize=5)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Median time per evaluation (µs)")
        ax.set_title(f"{robot.upper()} ABA Jacobian: LP vs Sequential CusADi vs MJX (L40S, fp64 / fp32)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        out = os.path.join(args.out_dir, f"{robot}_jacobian_comparison.png")
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")

    # Forward dynamics plot
    for robot in ("a1", "h1"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        any_plotted = False
        curves = [
            (f"{robot}_fwd_lp.json",       "LP (fp64)",          "C0", "o-"),
            (f"{robot}_fwd_seq.json",      "Sequential (fp64)",  "C1", "s-"),
            (f"{robot}_fwd_mjx_fp32.json", "MJX (fp32)",         "C2", "^-"),
            (f"{robot}_fwd_mjx_fp64.json", "MJX (fp64)",         "C3", "v-"),
            (f"{robot}_fwd_mjw.json",      "mujoco-warp (fp32)", "C4", "D-"),
        ]
        for fname, label, color, style in curves:
            p = os.path.join(args.results_dir, fname)
            if not os.path.exists(p):
                continue
            bs, med = load_sweep(p)
            ax.plot(bs, med, style, color=color, label=label, linewidth=1.5, markersize=5)
            any_plotted = True
        if not any_plotted:
            plt.close(fig)
            continue
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Median time per evaluation (µs)")
        ax.set_title(f"{robot.upper()} Forward dynamics: LP vs Sequential CusADi vs MJX vs mujoco-warp (L40S)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        out = os.path.join(args.out_dir, f"{robot}_forward_dynamics_comparison.png")
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
