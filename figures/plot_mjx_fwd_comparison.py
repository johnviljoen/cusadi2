"""Defense slide D — GPU-batched forward dynamics throughput.

Honest "worst case" plot: forward-dynamics-only (no Jacobian). MJX
and mujoco-warp are purpose-built for this workload, so the gap is
smaller than the Jacobian gap — but LP still wins from B≥8 upward.

Five curves per panel (A1 left, H1 right):
    ours (LP fp64)            blue
    seq cusadi (fp64)         grey
    MJX fp32                  light pink
    MJX fp64                  dark pink
    mujoco-warp (fp32)        green

Speedup ratios annotated between LP and MJX fp32 (the strongest
non-LP backend, since fp32 is what MJX is optimised for).

Usage:
    python figures/plot_mjx_fwd_comparison.py
        [--results_dir benchmark/mjx-comparison/results]
        [--out_dir figures]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_sweep(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    r = d["results"]
    bs = sorted(int(k) for k in r.keys())
    med_us = [r[str(b)]["median_s"] * 1e6 for b in bs]
    return np.array(bs), np.array(med_us, dtype=float)


def panel(ax, results_dir, robot, *, show_legend, show_ylabel):
    BLUE = "#2563eb"
    GREY = "#6b7280"
    PINK_LIGHT = "#f9a8d4"
    PINK = "#db2777"
    GREEN = "#16a34a"

    series = {}
    for key, fname in [
        ("LP",        f"{robot}_fwd_lp.json"),
        ("Seq",       f"{robot}_fwd_seq.json"),
        ("MJX fp32",  f"{robot}_fwd_mjx_fp32.json"),
        ("MJX fp64",  f"{robot}_fwd_mjx_fp64.json"),
        ("mjw fp32",  f"{robot}_fwd_mjw.json"),
    ]:
        s = load_sweep(os.path.join(results_dir, fname))
        if s is not None:
            series[key] = s

    if "LP" not in series:
        ax.text(0.5, 0.5, f"no LP fwd data for {robot}",
                transform=ax.transAxes, ha="center", va="center")
        return

    if "MJX fp64" in series:
        b, t = series["MJX fp64"]
        ax.plot(b, t, color=PINK, marker="v", linewidth=2.2, markersize=8,
                label="MJX fp64", zorder=3)
    if "MJX fp32" in series:
        b, t = series["MJX fp32"]
        ax.plot(b, t, color=PINK_LIGHT, marker="^", linewidth=2.2, markersize=8,
                label="MJX fp32", zorder=4)
    if "mjw fp32" in series:
        b, t = series["mjw fp32"]
        ax.plot(b, t, color=GREEN, marker="D", linewidth=2.2, markersize=8,
                label="mujoco-warp fp32", zorder=5)
    if "Seq" in series:
        b, t = series["Seq"]
        ax.plot(b, t, color=GREY, marker="s", linewidth=2.0, markersize=7,
                linestyle="--",
                label="seq cusadi fp64", zorder=6)
    b_lp, t_lp = series["LP"]
    ax.plot(b_lp, t_lp, color=BLUE, marker="o", linewidth=3.0, markersize=11,
            label="ours (LP fp64)", zorder=8)

    # Speedup ratios: LP vs FASTEST competitor at each batch
    # (best of MJX fp32, MJX fp64, mujoco-warp fp32 — winner can change
    # with B). Labels alternate above/below to prevent overlap.
    competitors = {k: series[k] for k
                   in ("MJX fp32", "MJX fp64", "mjw fp32") if k in series}
    for j, bb in enumerate(b_lp):
        best = float("inf")
        for _, (b_c, t_c) in competitors.items():
            m = (b_c == bb)
            if m.any():
                v = float(t_c[m][0])
                if np.isfinite(v):
                    best = min(best, v)
        if not np.isfinite(best):
            continue
        r = best / t_lp[j]
        if r < 1.5:
            continue
        ax.plot([bb, bb], [t_lp[j], best],
                color="#bbbbbb", linewidth=1.0, zorder=15)
        gm = np.sqrt(t_lp[j] * best)
        y_factor = 0.55 if (j % 2 == 0) else 1.85
        ax.text(bb * 1.05, gm * y_factor, f"{r:.0f}×",
                fontsize=10.5, color="#831843", weight="bold",
                ha="left", va="center", zorder=20,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor=PINK_LIGHT, lw=0.8))

    # Sequential-cusadi-wins-at-B=1 callout (honest disclosure). Place the
    # text on top of the LP curve in a region of empty space (between LP
    # and seq, low-batch) so it does NOT overlap the x-axis tick labels.
    if "Seq" in series:
        b_seq, t_seq = series["Seq"]
        if b_seq[0] == 1 and b_lp[0] == 1 and t_seq[0] < t_lp[0]:
            ax.annotate(
                f"seq cusadi wins at B=1\n({t_seq[0]:.0f} µs vs {t_lp[0]:.0f} µs)",
                xy=(1, t_seq[0]),
                xytext=(2.5, t_seq[0] * 1.55),
                fontsize=9, color=GREY, weight="bold",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0,
                                connectionstyle="arc3,rad=-0.20"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size  B")
    if show_ylabel:
        ax.set_ylabel("kernel time (µs, lower is better)")
    if robot == "a1":
        ax.set_title("A1 quadruped  ·  $n_v$ = 18", pad=10)
    else:
        ax.set_title("H1 humanoid  ·  $n_v$ = 25", pad=10)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{int(y):,}" if y < 1e4 else f"{int(y/1000):,}k"))
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)

    all_t = np.concatenate([t_lp] +
                           [s[1] for k, s in series.items() if k != "LP"])
    ax.set_xlim(0.7, b_lp[-1] * 2.0)
    ax.set_ylim(np.nanmin(all_t) * 0.4, np.nanmax(all_t) * 2.5)

    if show_legend:
        leg = ax.legend(loc="upper left", framealpha=0.97,
                        edgecolor="#cccccc", fontsize=10.5)
        leg.get_frame().set_linewidth(0.8)


def make_figure(results_dir, out_dir):
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":        13,
        "axes.titlesize":   15,
        "axes.labelsize":   14,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  10.5,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333333",
        "grid.color":       "#cccccc",
        "grid.linestyle":   ":",
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    panel(axes[0], results_dir, "a1", show_legend=True,  show_ylabel=True)
    panel(axes[1], results_dir, "h1", show_legend=False, show_ylabel=False)

    fig.suptitle(
        "GPU-batched forward dynamics throughput  ·  L40S  ·  "
        "ours (LP) vs MJX (fp32 / fp64) vs mujoco-warp (fp32) vs sequential cusadi",
        fontsize=15, y=1.0)

    fig.text(0.5, -0.04,
             "Forward dynamics only — no Jacobian. MJX and mujoco-warp "
             "are purpose-built for this workload. Sequential cusadi wins "
             "at B=1 (no smem prologue overhead). Ratios annotate ours vs "
             "MJX fp32 (MJX's native, strongest precision for fwd dynamics). "
             "Both: contacts disabled, same URDF, 10 warmup + 100 timed; JIT excluded.",
             ha="center", va="top", fontsize=10, style="italic", color="#444444")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "mjx_fwd_comparison.png")
    pdf = os.path.join(out_dir, "mjx_fwd_comparison.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default=os.path.join(repo, "benchmark", "mjx-comparison",
                                          "results"))
    ap.add_argument("--out_dir", default=here)
    args = ap.parse_args()
    make_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
