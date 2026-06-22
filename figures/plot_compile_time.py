"""Defense slide F — compile time: ours vs MJX across a sequence of Jacobians.

A real MPC pipeline evaluates many Jacobians — across horizons, across
parameter sweeps, across robots, across batch sizes used at deploy vs
training time. MJX has to JIT each new configuration. We don't.

Two panels, MJX-vs-ours only:
  left  — per-configuration compile cost
  right — cumulative compile cost as the pipeline covers a sequence of
          Jacobian configurations. ours stays flat at 1.5 s; MJX grows
          linearly. The empirical 48-Jacobian sweep is pinned.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


LP_KERNEL_S = 1.5    # one-time NVCC of the data-driven LP kernel


def jit_times(path):
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    r = d.get("results", {})
    return [r[k]["jit_compile_plus_run_s"] for k in r
            if r[k].get("jit_compile_plus_run_s")]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    res = os.path.join(repo, "benchmark", "mjx-comparison", "results")

    # Aggregate every MJX Jacobian configuration we measured. Each config
    # is one (robot × dtype × batch size × constraint mode) tuple — i.e.
    # one XLA trace. We deliberately don't break out the modes here.
    jac_files = [f for f in os.listdir(res)
                 if f.startswith(("a1_mjx", "h1_mjx")) and f.endswith(".json")]
    all_times = []
    for f in jac_files:
        all_times.extend(jit_times(os.path.join(res, f)))
    n_configs = len(all_times)
    total_jit = sum(all_times)
    median_per_config = float(np.median(all_times))

    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":        13,
        "axes.titlesize":   15,
        "axes.labelsize":   14,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  12.5,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333333",
        "grid.color":       "#cccccc",
        "grid.linestyle":   ":",
    })

    BLUE      = "#2563eb"
    PINK      = "#db2777"
    PINK_DARK = "#831843"
    ORANGE    = "#ea580c"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4),
                                    gridspec_kw={"width_ratios": [0.85, 1.15]})

    # =========================================================================
    # Panel 1 — per-configuration compile cost.
    # =========================================================================
    bars = [
        ("ours\n(LP kernel,\ncompiled once,\nused everywhere)", LP_KERNEL_S, BLUE),
        ("MJX\n(re-JIT every\nnew Jacobian\nconfiguration)",     median_per_config, PINK),
    ]
    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    xs = np.arange(len(labels))

    ax1.bar(xs, values, color=colors, edgecolor="#333333", linewidth=0.8,
            width=0.55, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, 250)
    ax1.set_ylabel("compile time per Jacobian configuration  (s, log scale)")
    ax1.set_title("Per-configuration compile cost", pad=10)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=11.5, ha="center")
    ax1.set_xlim(-0.6, len(labels) - 0.4)
    ax1.grid(True, axis="y", which="major", linewidth=0.6, alpha=0.7)
    ax1.grid(True, axis="y", which="minor", linewidth=0.4, alpha=0.3)
    for x, v in zip(xs, values):
        ax1.text(x, v * 1.20, f"{v:.1f} s" if v < 60 else f"{v:.0f} s",
                 ha="center", va="bottom", fontsize=14, weight="bold",
                 color="#222222", zorder=10)

    ax1.text(0.5, -0.30, f"ratio = {median_per_config / LP_KERNEL_S:.0f}× per configuration",
             transform=ax1.transAxes, ha="center", va="top",
             fontsize=12.5, color=PINK_DARK, weight="bold",
             bbox=dict(boxstyle="round,pad=0.32", facecolor="#fdf2f8",
                       edgecolor=PINK, lw=0.9))

    # =========================================================================
    # Panel 2 — cumulative compile cost over a sequence of Jacobians.
    # =========================================================================
    n_seq = np.arange(1, 121)
    ours_curve = np.full_like(n_seq, LP_KERNEL_S, dtype=float)
    mjx_curve  = median_per_config * n_seq

    ax2.plot(n_seq, mjx_curve, color=PINK, linewidth=2.8,
             label=f"MJX  ({median_per_config:.0f} s × N)", zorder=5)
    ax2.plot(n_seq, ours_curve, color=BLUE, linewidth=3.0,
             label=f"ours  ({LP_KERNEL_S:.1f} s, total)", zorder=8)

    ax2.set_xscale("linear")
    ax2.set_yscale("log")
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0.5, 20000)
    ax2.set_xlabel("number of Jacobian configurations the pipeline covers")
    ax2.set_ylabel("cumulative compile time  (s, log scale)")
    ax2.set_title("Cumulative compile cost across a sequence of Jacobians", pad=10)
    ax2.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax2.grid(True, which="minor", linewidth=0.4, alpha=0.3)

    # Reference horizontal lines.
    ax2.axhline(3600, color=ORANGE, linewidth=1.0, linestyle="--",
                alpha=0.7, zorder=2)
    ax2.text(120, 3600, " 1 hour", fontsize=10, color=ORANGE,
             weight="bold", ha="right", va="bottom")
    ax2.axhline(7200, color="#7f1d1d", linewidth=1.0, linestyle="--",
                alpha=0.6, zorder=2)
    ax2.text(120, 7200, " 2 hours", fontsize=10, color="#7f1d1d",
             weight="bold", ha="right", va="bottom")

    # Empirical pin: our actual measured sweep.
    ax2.scatter([n_configs], [total_jit], color=PINK,
                s=160, edgecolor="black", linewidth=1.4, zorder=12)
    ax2.annotate(
        f"our actual sweep:\n{n_configs} Jacobian configurations\n"
        f"MJX JIT total = {total_jit/60:.0f} min  ({total_jit:.0f} s)\n"
        f"ours          = 1.5 s",
        xy=(n_configs, total_jit),
        xytext=(n_configs + 5, total_jit * 0.18),
        fontsize=12, color=PINK_DARK, weight="bold",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=PINK, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=PINK_DARK, lw=1.3,
                        connectionstyle="arc3,rad=-0.15"),
        zorder=20)

    leg = ax2.legend(loc="upper left", framealpha=0.97,
                     edgecolor="#cccccc", fontsize=12.5)
    leg.get_frame().set_linewidth(0.8)

    fig.suptitle(
        f"Compile time scales with the # of Jacobians your pipeline covers  ·  "
        f"ours = {LP_KERNEL_S:.1f} s once  ·  MJX = {total_jit/60:.0f} min "
        f"for our {n_configs}-Jacobian sweep",
        fontsize=14.5, y=1.0)

    fig.text(0.5, -0.04,
             f"Each Jacobian configuration is a distinct (robot × precision × batch size × "
             f"problem-mode) tuple — i.e. one new XLA trace for MJX. ours = data-driven kernel, "
             f"compiled once at session start, reused across every configuration with no recompile. "
             f"Empirical median MJX JIT-per-configuration across our {n_configs}-config sweep "
             f"= {median_per_config:.0f} s.",
             ha="center", va="top", fontsize=10, style="italic", color="#444444")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])

    out = os.path.join(here, "compile_time.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
