"""Defense slide B (supplementary) — within-kernel LP ablation.

Two curves, same chunk kernel, only the in-block thread count changes:

    stage + LP        block_size = pick_block_size(plan)   (= our system)
    chunk @ bs = 1    block_size forced to 1               (LP off inside the chunk)

This isolates *within-block* parallelism from cross-block parallelism.
The chunk kernel always reserves the per-stage smem allocation
(34 KB at this function), so block_size=1 still pays the smem
allocation despite using only one thread → SM occupancy collapses
from ~2-3 resident blocks per SM to itself, on top of giving up the
threading factor. Combined ~38× slowdown.

Usage:
    CUDA_VISIBLE_DEVICES=1 python figures/plot_lp_ablation_supp.py [--data PATH]
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


def make_supp_figure(data, out_dir):
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":        14,
        "axes.titlesize":   16,
        "axes.labelsize":   15,
        "xtick.labelsize":  13,
        "ytick.labelsize":  13,
        "legend.fontsize":  12.5,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333333",
        "grid.color":       "#cccccc",
        "grid.linestyle":   ":",
    })

    BLUE = "#2563eb"
    PINK = "#ec4899"

    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    B = np.array(data["B"])
    lp = np.array(data["stage_LP"], dtype=float)
    bs1 = np.array(data["chunk_bs1"], dtype=float)
    block_size = data["block_size"]
    per_stage_kb = data["peak_live"] * 8 / 1024

    ax.plot(B, lp, color=BLUE, marker="o", linewidth=3.0, markersize=11,
            label=f"stage + LP    (block_size = {block_size})",
            zorder=4)
    ax.plot(B, bs1, color=PINK, marker="s", linewidth=2.6, markersize=10,
            linestyle="--",
            label="chunk @ bs = 1   (LP forced off, same smem)",
            zorder=3)

    # Speedup ratios between the two curves.
    gain = bs1 / lp
    for i, b in enumerate(B):
        if not np.isfinite(gain[i]) or gain[i] < 1.05:
            continue
        ax.plot([b, b], [lp[i], bs1[i]], color="#bbbbbb",
                linewidth=1.0, zorder=1.6)
        gm = np.sqrt(lp[i] * bs1[i])
        ax.text(b * 1.20, gm, f"{gain[i]:.0f}×",
                fontsize=11.5, color="#222222", weight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="#bbbbbb", lw=0.8))

    # Single explanation box, right side, in a region with no data.
    ax.text(0.985, 0.04,
            "Forcing block_size = 1 inside the chunk kernel\n"
            f"keeps the {per_stage_kb:.0f} KB per-stage smem allocation,\n"
            "so SMs still cap residency at ~3 blocks/SM.\n"
            "Result: idle threads + lost SM occupancy.\n"
            "Not a deployable configuration — this is a\n"
            "controlled within-kernel ablation of LP only.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10.5, color="#831843",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fdf2f8",
                      edgecolor=PINK, lw=1.0))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size  B  (problem instances)")
    ax.set_ylabel("kernel time (µs, lower is better)")
    ax.set_title(
        f"within-kernel LP ablation  ·  a1 ABA Jacobian  ·  "
        f"N = {data['n_stages']}",
        pad=12)

    ax.set_xlim(0.7, B[-1] * 1.8)
    ax.set_ylim(np.nanmin(lp) * 0.5, np.nanmax(bs1) * 2.4)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{int(y):,}" if y < 1e4 else f"{int(y/1000):,}k"))
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)

    leg = ax.legend(loc="upper left", framealpha=0.97,
                    edgecolor="#cccccc", fontsize=12)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "lp_ablation_supp.png")
    pdf = os.path.join(out_dir, "lp_ablation_supp.pdf")
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
    make_supp_figure(data, args.out_dir)


if __name__ == "__main__":
    main()
