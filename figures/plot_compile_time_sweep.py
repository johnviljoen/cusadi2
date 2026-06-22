"""Defense slide F — cold-compile time vs batch size, A1, fp32.

Two panels side by side. Left panel: forward dynamics compile time vs
batch size, comparing ours / MJX fp32 / mujoco-warp fp32. Right panel:
Jacobian compile time vs batch size, ours vs MJX fp32 (mujoco-warp has
no autodiff path so it's absent from the right panel).

Data source: benchmark/mjx-comparison/results/compile_time_sweep.json
produced by run_compile_time_sweep.sh — one cold-cache fresh-subprocess
measurement per (backend, task, batch).
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_data(path):
    with open(path) as f:
        return json.load(f)


def select(rows, **filters):
    out = []
    for r in rows:
        if "error" in r:
            continue
        ok = True
        for k, v in filters.items():
            if r.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    out.sort(key=lambda r: r["batch"])
    return out


def panel(ax, rows, task, *, show_legend, show_ylabel):
    BLUE  = "#2563eb"
    PINK  = "#db2777"
    GREEN = "#16a34a"

    series = {}
    for backend, label, color, marker in [
        ("lp",  "ours fp64",         BLUE,  "o"),
        ("mjx", "MJX fp32",          PINK,  "v"),
        ("mjw", "mujoco-warp fp32",  GREEN, "D"),
    ]:
        if task == "jac" and backend == "mjw":
            continue  # mujoco-warp has no Jacobian path
        sel = select(rows, backend=backend, task=task)
        if not sel:
            continue
        bs = np.array([r["batch"] for r in sel])
        ts = np.array([r["compile_time_s"] for r in sel])
        ax.plot(bs, ts, marker=marker, color=color,
                linewidth=2.6 if backend == "lp" else 2.2,
                markersize=10 if backend == "lp" else 8,
                label=label,
                zorder=8 if backend == "lp" else 4)
        series[backend] = (bs, ts)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size  B")
    if show_ylabel:
        ax.set_ylabel("cold compile time  (s, log scale)")
    ax.set_xlim(0.7, 14000)
    ax.set_ylim(0.5, 200)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}" if x >= 1 else ""))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f"{y:.1f}" if y < 10 else f"{int(y)}"))
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)

    title_map = {"fwd": "Forward dynamics",
                 "jac": "Jacobian"}
    ax.set_title(title_map[task], pad=10)

    if show_legend:
        leg = ax.legend(loc="upper left", framealpha=0.97,
                        edgecolor="#cccccc", fontsize=11)
        leg.get_frame().set_linewidth(0.8)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    src = os.path.join(repo, "benchmark", "mjx-comparison",
                        "results", "compile_time_sweep.json")
    rows = load_data(src)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4))
    panel(ax1, rows, "fwd", show_legend=True,  show_ylabel=True)
    panel(ax2, rows, "jac", show_legend=True,  show_ylabel=False)

    fig.suptitle(
        "Cold compile time vs batch size  ·  A1  ·  fp32  ·  L40S",
        fontsize=15, y=1.0)

    fig.text(0.5, -0.04,
             "Each point = one cold-cache fresh-subprocess measurement. "
             "ours = NVCC compile of the data-driven LP kernel (1.5 s, batch-independent). "
             "MJX = XLA JIT through mjx.forward (or jacfwd thereof), retraced per shape. "
             "mujoco-warp = Warp module compile, batch-independent within a process. "
             "Contacts disabled in MJX/warp so the comparison is on smooth dynamics only.",
             ha="center", va="top", fontsize=10, style="italic", color="#444444")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = os.path.join(here, "compile_time_sweep.png")
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
