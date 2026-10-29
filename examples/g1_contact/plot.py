"""Figures from results/*.csv, in the paper's style (Palatino, 8 pt, one-column width).

    python plot.py phase1          results/phase1_latency.{pdf,png}: step latency vs B, standing and drop panels
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(HERE, "..", "menagerie_bench"))
CUSADI_COLOR, WARP_COLOR = "#2ca02c", "#9467bd"           # the campaign's figure conventions (paper_plots.py)
FONT_PT = 8.0


def _style():
    import matplotlib
    matplotlib.use("Agg")
    from paper_fonts import use_palatino
    use_palatino()
    matplotlib.rcParams.update({"font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT, "legend.fontsize": FONT_PT,
                                "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "axes.linewidth": 0.5, "grid.linewidth": 0.3,
                                "lines.linewidth": 1.2, "lines.markersize": 3})


def _read(path):
    import csv
    with open(path) as f:
        return [{k: (float(v) if k not in ("arm", "variant", "dtype", "scenario") else v) for k, v in r.items()} for r in csv.DictReader(f)]


SERIES = [("cusadi2", "nogeom", CUSADI_COLOR, "-", "cusadi2 (smooth + integrate)"),
          ("cusadi2", "geom", CUSADI_COLOR, "--", "cusadi2 + geom Jacobians"),
          ("warp", "step", WARP_COLOR, "-", "MuJoCo Warp step")]


def phase1():
    _style()
    import matplotlib.pyplot as plt
    scens = [s for s in ("standing", "drop") if os.path.exists(os.path.join(RESULTS, f"phase1_{s}.csv"))]
    fig, axes = plt.subplots(1, len(scens), figsize=(3.5 * len(scens), 2.4), sharey=True, squeeze=False)
    for ax, scen in zip(axes[0], scens):
        rows = _read(os.path.join(RESULTS, f"phase1_{scen}.csv"))
        for arm, var, color, ls, label in SERIES:
            r = sorted([x for x in rows if x["arm"] == arm and x["variant"] == var], key=lambda x: x["B"])
            if not r:
                continue
            B = np.array([x["B"] for x in r]); us = np.array([x["median_ms"] / x["n_steps"] * 1e3 for x in r])
            ax.plot(B, us, ls, marker="o", color=color, label=label)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("batch size B"); ax.set_title(f"{scen}, contacts off, fp32, {int(rows[0]['n_steps'])} chained steps")
        ax.grid(True, which="major", color="0.85"); ax.grid(True, which="minor", color="0.93")
        ax.set_xticks([x["B"] for x in rows if x["arm"] == "warp"])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    axes[0][0].set_ylabel("latency per step [µs]")
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTS, f"phase1_latency.{ext}"), dpi=200)
    print("wrote", os.path.join(RESULTS, "phase1_latency.{pdf,png}"))


if __name__ == "__main__":
    {"phase1": phase1}[sys.argv[1]]()
