"""Defense slide E — constrained ABA Jacobian (with foot contacts).

A1 = 4 contact feet, H1 = 2 contact feet. Compares the cost of including
contact dynamics in each backend's Jacobian:

  ours, contact ON     LP fp64 on the constrained Pinocchio Jacobian
                       (foot contact wrenches folded analytically into
                       the DAG via Carpentier's algorithm).
  ours, contact OFF    LP fp64 on the unconstrained Jacobian (faded
                       reference — what you lose by adding contact).
  MJX, contact ON      jax.jacfwd(mjx.forward) with cylinder→capsule
                       swap and contype/conaffinity active. The full
                       MuJoCo contact LCP runs inside autodiff.
  MJX, contact OFF     baseline from the main Jacobian comparison
                       (faded reference, same units).

mujoco-warp is not on this plot because it has no autodiff path.

Speedup ratios are LP-constrained vs MJX-with-contact-fp64 — the
apples-to-apples *with-contact* head-to-head.

Usage:
    python figures/plot_mjx_constrained_comparison.py
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


def panel(ax, results_dir, robot, *, show_legend, show_ylabel, n_feet):
    BLUE = "#2563eb"
    BLUE_FADE = "#93c5fd"
    PINK = "#db2777"
    PINK_FADE = "#fbcfe8"

    series = {}
    paths = {
        "LP contact":          f"{robot}_lp_fp64_contact.json",
        "LP no-contact":       f"{robot}_lp_fp64.json",
        "MJX contact fp32":    f"{robot}_mjx_fp32_contact.json",
        "MJX no-contact fp32": f"{robot}_mjx_fp32.json",
    }
    for key, fname in paths.items():
        s = load_sweep(os.path.join(results_dir, fname))
        if s is not None:
            series[key] = s

    if "LP contact" not in series:
        ax.text(0.5, 0.5, f"missing LP contact data for {robot}",
                transform=ax.transAxes, ha="center", va="center")
        return

    # Faded MJX no-contact reference — also a valid competitor for the
    # speedup ratios (an audience member could ask "why not just skip
    # contacts and use MJX?"; we want to answer that head-on).
    if "MJX no-contact fp32" in series:
        b, t = series["MJX no-contact fp32"]
        ax.plot(b, t, color=PINK_FADE, marker="v", linestyle="--",
                linewidth=1.6, markersize=6,
                label="MJX fp32, contact OFF (baseline)", zorder=2)
    if "LP no-contact" in series:
        b, t = series["LP no-contact"]
        ax.plot(b, t, color=BLUE_FADE, marker="o", linestyle="--",
                linewidth=1.6, markersize=6,
                label="ours fp64, contact OFF (baseline)", zorder=3)

    # Bold contact-on MJX (fp32 only — fp64 is strictly slower so it just
    # adds clutter on this plot).
    if "MJX contact fp32" in series:
        b, t = series["MJX contact fp32"]
        ax.plot(b, t, color=PINK, marker="^",
                linewidth=2.4, markersize=9,
                label="MJX fp32, contact ON", zorder=4)
    b_lpc, t_lpc = series["LP contact"]
    ax.plot(b_lpc, t_lpc, color=BLUE, marker="o",
            linewidth=3.0, markersize=11,
            label=f"ours fp64, contact ON ({n_feet} feet)", zorder=8)

    # Speedup ratios: LP-contact vs the FASTEST competitor at each batch,
    # contact-on AND contact-off variants (the contact-off MJX curves are
    # genuine alternatives — audience may ask "why not skip contact?").
    # Winning competitor can change as B grows.
    competitors = {k: series[k] for k in (
        "MJX contact fp32",
        "MJX no-contact fp32",
    ) if k in series}
    for j, bb in enumerate(b_lpc):
        best = float("inf")
        for _, (b_c, t_c) in competitors.items():
            m = (b_c == bb)
            if m.any():
                v = float(t_c[m][0])
                if np.isfinite(v):
                    best = min(best, v)
        if not np.isfinite(best):
            continue
        r = best / t_lpc[j]
        if r < 1.5:
            continue
        ax.plot([bb, bb], [t_lpc[j], best],
                color="#bbbbbb", linewidth=1.0, zorder=15)
        gm = np.sqrt(t_lpc[j] * best)
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
        ax.set_title(f"A1 quadruped  ·  {n_feet}  ·  $n_v$ = 18", pad=10)
    else:
        ax.set_title(f"H1 humanoid  ·  {n_feet}  ·  $n_v$ = 25", pad=10)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{int(y):,}" if y < 1e4 else f"{int(y/1000):,}k"))
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)

    all_t = np.concatenate([s[1] for s in series.values()])
    ax.set_xlim(0.7, b_lpc[-1] * 2.0)
    ax.set_ylim(np.nanmin(all_t) * 0.5, np.nanmax(all_t) * 2.5)

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
    panel(axes[0], results_dir, "a1", show_legend=True,
          show_ylabel=True, n_feet="4×3D")
    panel(axes[1], results_dir, "h1", show_legend=False,
          show_ylabel=False, n_feet="2×6D")

    fig.suptitle(
        "GPU-batched constrained ABA Jacobian throughput  ·  L40S  ·  "
        "ours (LP fp64) vs MJX (fp64), contact ON / OFF",
        fontsize=15, y=1.0)

    fig.text(0.5, -0.04,
             "Solid curves = contact ON. Dashed curves = baseline contact OFF (same numbers as the main "
             "Jacobian comparison). LP folds foot-contact wrenches into the analytical Pinocchio Jacobian "
             "via Carpentier's algorithm; MJX runs the full LCP solver inside jacfwd. mujoco-warp not "
             "shown — no autodiff path. Same URDF, 10 warmup + 100 timed; JIT excluded.",
             ha="center", va="top", fontsize=10, style="italic", color="#444444")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "mjx_constrained_comparison.png")
    pdf = os.path.join(out_dir, "mjx_constrained_comparison.pdf")
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
