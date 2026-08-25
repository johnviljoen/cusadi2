"""Defense slide A — main 2×2 ablation: {staging on/off} × {LP on/off}.

Three measurable curves at fixed total work (B × n_stages thread groups):

    stage + LP    (ours)              ON / ON
    original cusadi (deployable)      ON / OFF      (= stage axis flattened
                                                       into batch axis;
                                                       per-stage AOT kernel)
    unrolled + LP                     OFF / ON      (LP on the stamped DAG;
                                                       grid = (B,) only)

The 4th corner — stage OFF, LP OFF — is the original cusadi codegen
applied to the unrolled .cu. We attempted it (figures/lp_ablation_unrolled_cusadi.json)
and NVCC did not finish in 2 hours of wall time on a 35 MB / 490k-instruction
single-function .cu, so the corner is shown as ✗ instead of a curve.

Usage:
    CUDA_VISIBLE_DEVICES=1 python figures/plot_lp_ablation_main.py [--data PATH]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_data(path):
    with open(path) as f:
        return json.load(f)


def make_main_figure(data, out_dir):
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":        14,
        "axes.titlesize":   16,
        "axes.labelsize":   15,
        "xtick.labelsize":  13,
        "ytick.labelsize":  13,
        "legend.fontsize":  13,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333333",
        "grid.color":       "#cccccc",
        "grid.linestyle":   ":",
    })

    BLUE  = "#2563eb"
    GREY  = "#6b7280"
    GREEN = "#16a34a"

    fig, ax = plt.subplots(figsize=(11, 7.0))

    B = np.array(data["B"])
    lp = np.array(data["stage_LP"], dtype=float)
    orig_raw = data["original"]
    unr_raw = data["unrolled_LP"]
    orig = np.array([np.nan if v is None else v for v in orig_raw], dtype=float)
    unr = np.array([np.nan if v is None else v for v in unr_raw], dtype=float)
    has_orig = np.isfinite(orig).any()
    has_unr = np.isfinite(unr).any()

    # Curves — only the three 2×2 corners that are physically measurable.
    ax.plot(B, lp, color=BLUE, marker="o", linewidth=3.0, markersize=11,
            label="ours       (staging ON, LP ON)", zorder=6)

    if has_orig:
        ax.plot(B, orig, color=GREY, marker="^", linewidth=2.6, markersize=10,
                label="no LP     (staging ON, LP OFF — original cusadi)",
                zorder=5)

    if has_unr:
        mask = np.isfinite(unr)
        ax.plot(B[mask], unr[mask], color=GREEN, marker="D",
                linewidth=2.4, markersize=9,
                label="no staging (staging OFF, LP ON — unrolled DAG)",
                zorder=4)

    # Speed-up ratio annotations (ours vs original cusadi only — the
    # comparison the audience cares about).
    if has_orig:
        gain = orig / lp
        for i, b in enumerate(B):
            if not np.isfinite(gain[i]) or gain[i] < 1.05:
                continue
            ax.plot([b, b], [lp[i], orig[i]], color="#bbbbbb",
                    linewidth=1.0, zorder=1.6)
            gm = np.sqrt(lp[i] * orig[i])
            ax.text(b * 1.20, gm, f"{gain[i]:.1f}×",
                    fontsize=11.5, color="#222222", weight="bold",
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                              edgecolor="#bbbbbb", lw=0.8))

    # 4th 2×2 corner: stage OFF + LP OFF — un-compileable.
    # Disclosed as a footer below the figure; cleaner than an in-axes ✗.
    finite_max = np.nanmax([np.nanmax(lp),
                            np.nanmax(orig) if has_orig else np.nan,
                            np.nanmax(unr)  if has_unr  else np.nan])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size  B  (problem instances)")
    ax.set_ylabel("kernel time (µs, lower is better)")
    ax.set_title(
        f"a1 ABA Jacobian  ·  N = {data['n_stages']}  ·  "
        f"per-stage = {data['peak_live']*8/1024:.0f} KB smem  ·  "
        f"chunk_size = {data['chunk_size']}",
        pad=12)

    ax.set_xlim(0.7, B[-1] * 4.5)
    ax.set_ylim(np.nanmin(lp) * 0.4, finite_max * 12.0)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{int(y):,}" if y < 1e4 else f"{int(y/1000):,}k"))
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)

    leg = ax.legend(loc="upper left", framealpha=0.97,
                    edgecolor="#cccccc", fontsize=12)
    leg.get_frame().set_linewidth(0.8)

    # Greyed italicised footer — discloses the un-measurable 4th 2×2 corner
    # in the same style as the MJX comparison footers.
    fig.text(0.5, -0.04,
             "Missing 2×2 corner — staging OFF + LP OFF (original cusadi on the 490k-instruction "
             "unrolled .cu): NVCC compile exceeded 2 hours of wall time at MPC horizon N = 30. "
             "The prior pipeline cannot reach this regime.",
             ha="center", va="top", fontsize=10, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "lp_ablation_main.png")
    pdf = os.path.join(out_dir, "lp_ablation_main.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(here, "lp_ablation_N30.json"))
    ap.add_argument("--out_dir", default=here)
    args = ap.parse_args()
    data = load_data(args.data)
    make_main_figure(data, args.out_dir)


if __name__ == "__main__":
    main()
