"""Defense slide C — GPU-batched ABA Jacobian throughput: ours vs MJX.

Two-panel figure (A1, H1), same log-log axes, ours = LP fp64 in blue,
MJX fp32 / fp64 in pinks, sequential CusADi in grey. Speedup ratios
are annotated between LP and MJX fp64. Same defense visual style as
lp_ablation_main.png.

Usage:
    python figures/plot_mjx_comparison.py
        [--results_dir experiments/benchmarks/mjx-comparison/results]
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
    """Draw a single robot's panel."""
    BLUE = "#2563eb"
    GREY = "#6b7280"
    PINK_LIGHT = "#f9a8d4"
    PINK = "#db2777"

    series = {}
    for key, fname in [
        ("LP",       f"{robot}_lp_fp64.json"),
        ("Seq",      f"{robot}_seq_fp64.json"),
        ("MJX fp32", f"{robot}_mjx_fp32.json"),
        ("MJX fp64", f"{robot}_mjx_fp64.json"),
    ]:
        s = load_sweep(os.path.join(results_dir, fname))
        if s is not None:
            series[key] = s

    if "LP" not in series:
        ax.text(0.5, 0.5, f"no LP data for {robot}",
                transform=ax.transAxes, ha="center", va="center")
        return

    # Plot order: weakest at the back.
    if "MJX fp64" in series:
        b, t = series["MJX fp64"]
        ax.plot(b, t, color=PINK, marker="v", linewidth=2.4, markersize=9,
                label="MJX fp64", zorder=3)
    if "MJX fp32" in series:
        b, t = series["MJX fp32"]
        ax.plot(b, t, color=PINK_LIGHT, marker="^", linewidth=2.2, markersize=8,
                label="MJX fp32", zorder=4)
    if "Seq" in series:
        b, t = series["Seq"]
        ax.plot(b, t, color=GREY, marker="s", linewidth=2.2, markersize=8,
                label="seq cusadi fp64", zorder=5)
    b_lp, t_lp = series["LP"]
    ax.plot(b_lp, t_lp, color=BLUE, marker="o", linewidth=3.0, markersize=11,
            label="ours (LP fp64)", zorder=8)

    # Speedup ratios: LP vs the FASTEST competitor at each batch — the
    # winning competitor can change with B (MJX fp32 vs fp64). Labels
    # alternate above/below the geometric mean to prevent overlap.
    competitors = {k: series[k] for k in ("MJX fp32", "MJX fp64") if k in series}
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
                fontsize=11, color="#831843", weight="bold",
                ha="left", va="center", zorder=20,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor=PINK, lw=0.8))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size  B")
    if show_ylabel:
        ax.set_ylabel("kernel time (µs, lower is better)")
    if robot == "a1":
        ax.set_title(f"A1 quadruped  ·  $n_v$ = 18", pad=10)
    else:
        ax.set_title(f"H1 humanoid  ·  $n_v$ = 25", pad=10)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{int(y):,}" if y < 1e4 else f"{int(y/1000):,}k"))
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)

    # Set sensible axis limits across all curves.
    all_t = np.concatenate([t_lp] +
                           [s[1] for k, s in series.items() if k != "LP"])
    ax.set_xlim(0.7, b_lp[-1] * 2.0)
    ax.set_ylim(np.nanmin(all_t) * 0.5, np.nanmax(all_t) * 2.5)

    if show_legend:
        leg = ax.legend(loc="upper left", framealpha=0.97,
                        edgecolor="#cccccc", fontsize=11)
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
        "legend.fontsize":  11,
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
        "GPU-batched ABA Jacobian throughput  ·  L40S  ·  "
        "ours (LP fp64) vs MJX (fp32 / fp64) vs sequential cusadi",
        fontsize=15, y=1.0)

    fig.text(0.5, -0.04,
             "MJX = jax.jacfwd(mjx.forward), vmap'd, jit'd. "
             "LP = pinocchio.casadi analytical Carpentier derivatives, "
             "compiled through level-parallel CUDA kernel. "
             "Both: contacts disabled, same URDF, 10 warmup + 100 timed. "
             "MJX JIT compile time (30–160 s per shape) excluded from medians.",
             ha="center", va="top", fontsize=10, style="italic", color="#444444")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "mjx_comparison.png")
    pdf = os.path.join(out_dir, "mjx_comparison.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default=os.path.join(repo, "experiments", "benchmarks", "mjx-comparison",
                                          "results"))
    ap.add_argument("--out_dir", default=here)
    args = ap.parse_args()
    make_figure(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
