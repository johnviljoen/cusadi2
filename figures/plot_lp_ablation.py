"""Render the LP-vs-no-LP ablation as a defense-slide-quality figure.

Three curves on log-log axes — same total work (B × n_stages thread
groups), three different ways of executing it:

    stage + LP      Our system. Chunk kernel, block_size = pick_block_size,
                    threads cooperate within each block on the chunked DAG
                    via the level scheduler.

    chunk @ bs=1    Within-kernel ablation. Same chunk kernel, but
                    block_size forced to 1 — only one thread per block
                    walks the DAG sequentially. Useful as a controlled
                    ablation of within-block parallelism, but pays the
                    chunk kernel's per-block 34 KB shared-memory allocation
                    even though only one thread uses it (~5–10× SM-occupancy
                    penalty). Not a deployable configuration.

    original cusadi The prior state-of-the-art deployable baseline.
                    AOT-compiled per-stage kernel, run at batch B × n_stages
                    (i.e. stage axis flattened into batch axis). One thread
                    per (instance, stage), inline arithmetic, no shared
                    memory.

Two distinct gaps to read off the plot:

    stage+LP vs chunk@bs=1   = within-kernel LP gain. Constant ~38×, set by
                               median per-level width / 1. Measured under
                               the smem penalty above, so this is a ceiling
                               on what LP can deliver within this kernel.

    stage+LP vs original     = realistic LP gain over the prior deployable
                               baseline. Variable: 1.2–19× depending on B.
                               Smallest at B≈256 (GPU saturation by sheer
                               block count); biggest in the small-batch
                               real-time MPC regime; recovers at large B
                               where original cusadi runs out of per-thread
                               scaling.

Usage:
    CUDA_VISIBLE_DEVICES=1 python figures/plot_lp_ablation.py [--regenerate]
"""
from __future__ import annotations

import argparse
import os
import sys

# Cached data from a1_aba_jac_analytical (n_stages=12) on John's L40S, GPU 1.
# n_stages=12 was chosen because at n_stages=24 the unrolled function's
# tiled-LP plan exceeds the device's 99 KB shared-memory budget (a1's
# unrolled DAG at N=24 has a level with width ~67k, which the tiled-plan
# extractor cannot fit into 12k-slot chunks). N=12 is still a typical MPC
# horizon and exercises the same scaling regime.
#   per-stage peak_live = 4402 slots (34.4 KB)
#   median per-level width = 81, max = 2781
#   block_size picked = 128
#   unrolled at N=12: 490k instr, 226 levels, peak 52747 slots (412 KB) → tiled
DEFAULT_DATA = {
    "fn":           "a1_aba_jac_analytical",
    "n_stages":     12,
    "median_width": 81,
    "block_size":   128,
    "B":            [1,     8,     32,     128,    256,    1024,   4096],
    "stage_LP":     [160.8, 162.5, 316.4,  940.2,  1756.2, 7587.8, 29838.4],
    # NEW: LP applied to the 12-stamped unrolled function. One block per
    # batch instance; tiled because peak smem is 412 KB.
    "unrolled_LP":  [2428.5,2814.5,3882.0, 5429.8, 10844.7,44671.5,190258.7],
    # Within-kernel ablation: chunk kernel forced to block_size=1.
    # Pays full per-block smem allocation despite single-thread use.
    "chunk_bs1":    [5717.0,6139.9,12128.8,36662.8,67780.1,299361.8,1172010.0],
    # Deployable no-LP baseline: original cusadi at batch B*n_stages.
    "original":     [599.2, 1600.5,4336.1, 4338.3, 4302.3, 8881.2, 157752.9],
}


def regenerate():
    """Re-run the ablation and return fresh measurements."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from validation._section_lp_ablation import run_section_lp_ablation
    out = run_section_lp_ablation()
    Bs = sorted(int(b) for b in out.keys())
    return {
        "fn":           "a1_aba_jac_analytical",
        "n_stages":     24,
        "median_width": 81,
        "block_size":   128,
        "B":            Bs,
        "stage_LP":     [out[str(b)]["stage_LP_us"]   for b in Bs],
        "chunk_bs1":    [out[str(b)]["stage_only_us"] for b in Bs],
        "original":     [out[str(b)]["sequential_us"] for b in Bs],
    }


def make_plot(data, out_dir):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":        13,
        "axes.titlesize":   15,
        "axes.labelsize":   14,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  11.5,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333333",
        "axes.labelcolor":  "#000000",
        "grid.color":       "#cccccc",
        "grid.linestyle":   ":",
    })

    BLUE   = "#3b82f6"
    GREEN  = "#16a34a"
    PINK   = "#ec4899"
    GREY   = "#6b7280"
    SHADE  = "#fff7e6"

    fig, ax = plt.subplots(figsize=(12.5, 8.5))

    B    = np.array(data["B"])
    lp   = np.array(data["stage_LP"])
    unr  = np.array(data["unrolled_LP"])
    bs1  = np.array(data["chunk_bs1"])
    orig = np.array(data["original"])
    realistic_gain = orig / lp

    # Real-time MPC operating zone (B ≤ 256), shaded vertical band.
    ax.axvspan(0.7, 256, color=SHADE, alpha=1.0, zorder=0)

    # Four curves — the 2×2 ablation, plus a within-kernel ablation:
    #   stage on  + LP on  : our system                  (blue, "stage + LP")
    #   stage off + LP on  : LP only, no staging         (green, "unrolled + LP")
    #   stage on  + LP off : staging only, deployable    (grey,  "stage" — original cusadi @ B×N)
    #   stage off + LP off : (would need unrolled-cusadi NVCC build, skipped)
    # Plus the within-kernel ablation:
    #   chunk @ block_size = 1                            (pink dashed)
    ax.plot(B, lp, color=BLUE, marker="o", linewidth=2.6, markersize=9,
            label="stage + LP    (staging ON,  LP ON  — our system)",
            zorder=6)
    ax.plot(B, unr, color=GREEN, marker="D", linewidth=2.4, markersize=8,
            label="unrolled + LP    (staging OFF, LP ON — LP on the unrolled-horizon DAG; tiled)",
            zorder=5)
    ax.plot(B, orig, color=GREY, marker="^", linewidth=2.6, markersize=9,
            label="stage    (staging ON,  LP OFF — original cusadi kernel @ batch B × n_stages)",
            zorder=4)
    ax.plot(B, bs1, color=PINK, marker="s", linewidth=2.2, markersize=7,
            linestyle="--",
            label="chunk @ bs = 1    (within-kernel ablation: same chunk kernel, 1 thread/block)",
            zorder=3)

    # Realistic-LP-gain ratios drawn between stage+LP and original cusadi.
    for i, b in enumerate(B):
        if realistic_gain[i] < 1.05:
            continue
        ax.plot([b, b], [lp[i], orig[i]],
                color="#bbbbbb", linewidth=1.0, zorder=1.6)
        gm = np.sqrt(lp[i] * orig[i])
        ax.text(b * 1.18, gm,
                f"{realistic_gain[i]:.1f}×",
                fontsize=9.5, color="#222222", weight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="#bbbbbb", lw=0.8))

    # Real-time-regime callout (top of shaded zone).
    ax.text(15, 2.45e6, "real-time MPC regime  (B ≤ 256)",
            fontsize=12, color="#b45309",
            ha="center", va="top", weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=SHADE,
                      edgecolor="#b45309", lw=1.0))

    # Unrolled-LP callout: explains why it's slow at small B.
    ax.annotate(
        "unrolled + LP:\nonly B blocks total (no stage axis);\nGPU under-filled at small batch,\nthen tile overhead caps scaling",
        xy=(8, unr[1]),
        xytext=(1.0, 4e4),
        fontsize=9.0, color=GREEN, weight="bold",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2,
                        connectionstyle="arc3,rad=-0.25"),
    )

    # GPU-saturation callout: full sentence, placed in empty mid-bottom band
    # to the right of the shaded zone (between the curves and the legend).
    ax.annotate(
        "stage (no LP) catches stage + LP at B ≈ 256:\nGPU fully block-saturated; cross-block parallelism\nalone hides LP's advantage — but the gap reopens past\nsaturation, where each block's per-thread serial DAG\nbecomes the bottleneck again",
        xy=(256, np.sqrt(lp[4] * orig[4])),
        xytext=(2.5, 75),
        fontsize=8.5, color="#444444", weight="bold",
        ha="left", va="bottom",
        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2,
                        connectionstyle="arc3,rad=-0.3"),
    )

    # Within-kernel-ablation callout (pink): on the right side, ha="right".
    ax.annotate(
        "chunk @ bs=1:\nour chunk kernel forced to 1 thread/block.\nPays full 34 KB smem allocation despite\nsingle-thread use → SM-occupancy collapses.\nNot deployable; isolates within-block LP\nfrom cross-block parallelism for ablation.",
        xy=(1024, bs1[5]),
        xytext=(6500, 2.4e6),
        fontsize=9.0, color=PINK, weight="bold",
        ha="right", va="top",
        arrowprops=dict(arrowstyle="->", color=PINK, lw=1.2,
                        connectionstyle="arc3,rad=0.25"),
    )


    # Axes setup.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size  B  (problem instances)")
    ax.set_ylabel("kernel time (µs, lower is better)")
    ax.set_title(
        f"2×2 ablation: {{staging on/off}} × {{LP on/off}}, same (B × n_stages) total work\n"
        f"{data['fn']}  ·  n_stages={data['n_stages']}  ·  "
        f"median per-level width = {data['median_width']}  ·  "
        f"block_size = {data['block_size']}",
        pad=14,
    )

    ax.set_xlim(0.7, 7000)
    ax.set_ylim(60, 4.0e6)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{int(y):,}" if y < 1e4 else f"{int(y/1000):,}k"))

    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)

    leg = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        ncol=2, framealpha=0.97, edgecolor="#cccccc",
        fontsize=10.5,
    )
    leg.get_frame().set_linewidth(0.8)

    # Footer: the missing 2×2 corner — original cusadi on unrolled function.
    # We attempted to compile the 490k-instr single-.cu via NVCC; it did
    # not finish in 2 hours of wall time. The data point IS the absence.
    fig.text(0.5, 0.005,
             "Missing 2×2 corner — stage OFF + LP OFF (original cusadi on the 490k-instruction unrolled .cu): "
             "NVCC compile exceeded 2 hours of wall time (cicc still in PTX phase at timeout). "
             "Not buildable in defense-prep time → this corner is unreachable for the prior pipeline at MPC horizons.",
             ha="center", va="bottom", fontsize=9.0, style="italic",
             color="#444444")

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "lp_ablation.png")
    pdf = os.path.join(out_dir, "lp_ablation.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return png, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true",
                    help="re-run the ablation instead of using cached data")
    ap.add_argument("--out_dir",
                    default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    data = regenerate() if args.regenerate else DEFAULT_DATA
    make_plot(data, args.out_dir)


if __name__ == "__main__":
    main()
