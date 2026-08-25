#!/usr/bin/env python
"""Paper figures assembled from the completed Menagerie campaign results.

Standalone plots (composed manually in Inkscape afterwards):

  plot1 — speedup-vs-batch for the frame Jacobian: per robot and batch B,
          speedup = MJX jac_batched wall / cusadi2 `full` wall (both
          Python-hosted, B*N work each). Median line per precision with a
          percentile band across the robot ensemble; linear y, log x.
          Only robots with complete curves at every batch are included, so
          the ensemble composition is identical along the x-axis.

    python examples/menagerie_bench/paper_plots.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]

# (dtype, fp tag, line color, band color)
SERIES = [("float", "fp32", "C1", "C1"), ("double", "fp64", "C0", "C0")]
BAND = (25, 75)          # percentile band around the median


def _speedups(func="fkjac", mjx_key="jac_batched", config="full"):
    """{fp: {robot: {B: speedup}}} using Python-hosted walls on both sides."""
    out = {}
    for dtype, fp, _, _ in SERIES:
        per = {}
        for robot in sorted(os.listdir(RESULTS)):
            rdir = os.path.join(RESULTS, robot)
            mf, cf = (os.path.join(rdir, f"mjx_{fp}.json"),
                      os.path.join(rdir, f"latency_py_{dtype}.csv"))
            if not (os.path.isfile(mf) and os.path.isfile(cf)):
                continue
            try:
                jb = (json.load(open(mf)).get("results") or {}).get(mjx_key) or {}
            except Exception:
                continue
            cus = {}
            for r in csv.DictReader(open(cf)):
                if r["func"] == func and r["config"] == config:
                    try:
                        cus[int(r["batch"])] = float(r["wall_us"])
                    except ValueError:
                        pass
            sp = {B: jb[str(B)] * 1e6 / cus[B] for B in BATCHES
                  if jb.get(str(B)) is not None and B in cus}
            if sp:
                per[robot] = sp
        out[fp] = per
    return out


def _panel(ax, data, title, legend=False, raw=True, rng=None):
    """Median + IQR band per precision on `ax`, with the raw per-robot points
    (complete-ensemble robots only) overlaid in a light tint. Returns the max
    raw value drawn (for y-limit decisions by the caller)."""
    rng = rng or np.random.default_rng(0)
    raw_max = 0.0
    for _, fp, color, band in SERIES:
        per = data[fp]
        complete = {r: sp for r, sp in per.items() if all(B in sp for B in BATCHES)}
        arr = np.array([[complete[r][B] for B in BATCHES] for r in sorted(complete)])
        if raw:
            for j, B in enumerate(BATCHES):
                ax.scatter(np.full(arr.shape[0], B), arr[:, j], s=5, color=color,
                           alpha=0.16, lw=0, zorder=1.5)
            raw_max = max(raw_max, float(arr.max()))
        med = np.median(arr, axis=0)
        lo, hi = (np.percentile(arr, BAND[0], axis=0),
                  np.percentile(arr, BAND[1], axis=0))
        ax.fill_between(BATCHES, lo, hi, color=band, alpha=0.20, lw=0, zorder=2)
        ax.plot(BATCHES, med, "-o", color=color, ms=3.5, lw=1.8, zorder=3,
                label=f"{fp} (n={len(complete)})")
    ax.axhline(1.0, color="gray", lw=1, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(BATCHES, [str(b) for b in BATCHES])
    ax.set_xlim(BATCHES[0] / 1.35, BATCHES[-1] * 1.35)
    ax.set_xlabel("batch size B")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", ls=":", alpha=0.4)
    if legend:
        ax.legend(fontsize=8, title="precision", title_fontsize=8)
    return raw_max


def plot_pair(out_stem="paper_speedup_vs_batch_pair", ymax=None):
    """The shared-y 1x2 figure: frame Jacobians (left) and FK (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jac = _speedups(func="fkjac", mjx_key="jac_batched")
    fk = _speedups(func="fk", mjx_key="fk_batched")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    m1 = _panel(axes[0], jac, "frame Jacobians", legend=True)
    m2 = _panel(axes[1], fk, "forward kinematics")
    axes[0].set_ylabel("speedup  (MJX batched / cusadi2 full)")
    axes[0].set_ylim(0, ymax if ymax is not None else max(m1, m2) * 1.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS, f"{out_stem}.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(RESULTS, out_stem)}.png/.pdf  "
          f"(raw max: jac {m1:.1f}, fk {m2:.1f})")


def _nv_by_robot():
    nv = {}
    for robot in os.listdir(RESULTS):
        st = os.path.join(RESULTS, robot, "status.json")
        if os.path.isfile(st):
            try:
                gate = json.load(open(st)).get("gate") or {}
            except Exception:
                continue
            if gate.get("status") == "pass":
                nv[robot] = int(gate["nv"])
    return nv


def plot_absolute(B=256, func="fkjac", mjx_key="jac_batched",
                  yscale="log", out_stem=None):
    """Absolute wall-clock (us, log y) at one batch size vs model size nv:
    cusadi2 `full` (solid) and MJX batched (dashed) per precision, one marker
    per robot. Same width as the pair figure; sits below it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_stem = out_stem or (f"paper_absolute_B{B}_{func}"
                            + ("_linear" if yscale == "linear" else ""))
    nv = _nv_by_robot()
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for dtype, fp, color, _ in SERIES:
        cus, mjx = [], []
        for robot, n in nv.items():
            rdir = os.path.join(RESULTS, robot)
            cf = os.path.join(rdir, f"latency_py_{dtype}.csv")
            if os.path.isfile(cf):
                for r in csv.DictReader(open(cf)):
                    if (r["func"] == func and r["config"] == "full"
                            and int(r["batch"]) == B):
                        try:
                            cus.append((n, float(r["wall_us"])))
                        except ValueError:
                            pass
            mf = os.path.join(rdir, f"mjx_{fp}.json")
            if os.path.isfile(mf):
                try:
                    v = ((json.load(open(mf)).get("results") or {})
                         .get(mjx_key) or {}).get(str(B))
                except Exception:
                    v = None
                if v is not None:
                    mjx.append((n, v * 1e6))
        for pts, ls, mk, label in ((cus, "-", "o", f"cusadi2 full {fp}"),
                                   (mjx, "--", "^", f"MJX batched {fp}")):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker=mk,
                    color=color, ms=3.5, lw=1.2, alpha=0.9, label=label,
                    markerfacecolor=color if ls == "-" else "none")
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    if yscale == "linear":
        ax.set_ylim(bottom=0)
    ax.set_xticks([2, 6, 8, 18, 30, 50, 70, 108],
                  ["2", "6", "8", "18", "30", "50", "70", "108"])
    ax.set_xlabel("model size nv (dof)")
    ax.set_ylabel(f"wall-clock per call (us)")
    ax.set_title(f"frame Jacobians, B={B} (x N={8} stages per call)", fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS, f"{out_stem}.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(RESULTS, out_stem)}.png/.pdf "
          f"(cusadi2 n={len(cus)}, mjx n={len(mjx)} for {fp})")


def plot_absolute_ranked(B=256, func="fkjac", mjx_key="jac_batched",
                         out_stem=None):
    """Absolute wall-clock (us, log y) at one batch size, one x slot per robot.
    Robots are ordered by nv, ties broken by ascending cusadi2-fp32 wall, so
    equal-nv groups read as a rising staircase instead of a vertical smear.
    nv group centers are labelled on the x axis with faint separators."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_stem = out_stem or f"paper_absolute_B{B}_{func}_ranked"
    nv = _nv_by_robot()

    def cus_wall(robot, dtype):
        cf = os.path.join(RESULTS, robot, f"latency_py_{dtype}.csv")
        if os.path.isfile(cf):
            for r in csv.DictReader(open(cf)):
                if (r["func"] == func and r["config"] == "full"
                        and int(r["batch"]) == B):
                    try:
                        return float(r["wall_us"])
                    except ValueError:
                        return None
        return None

    def mjx_wall(robot, fp):
        mf = os.path.join(RESULTS, robot, f"mjx_{fp}.json")
        if os.path.isfile(mf):
            try:
                v = ((json.load(open(mf)).get("results") or {})
                     .get(mjx_key) or {}).get(str(B))
                return None if v is None else v * 1e6
            except Exception:
                return None
        return None

    rows = []
    for robot, n in nv.items():
        vals = {"cus_float": cus_wall(robot, "float"),
                "cus_double": cus_wall(robot, "double"),
                "mjx_fp32": mjx_wall(robot, "fp32"),
                "mjx_fp64": mjx_wall(robot, "fp64")}
        if vals["cus_float"] is not None:
            rows.append((n, vals["cus_float"], robot, vals))
    rows.sort()                                  # (nv, cusadi2-fp32 wall)
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for dtype, fp, color, _ in SERIES:
        for key, ls, mk, label, mfc in (
                (f"cus_{dtype}", "-", "o", f"cusadi2 full {fp}", color),
                (f"mjx_{fp}", "--", "^", f"MJX batched {fp}", "none")):
            y = np.array([r[3][key] if r[3][key] is not None else np.nan
                          for r in rows], dtype=float)
            ax.plot(x, y, ls, marker=mk, color=color, ms=3.5, lw=1.2,
                    alpha=0.9, label=label, markerfacecolor=mfc)
    # nv group boundaries + centered labels
    ticks, labels = [], []
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1
        ticks.append((i + j - 1) / 2)
        labels.append(str(rows[i][0]))
        if j < len(rows):
            ax.axvline(j - 0.5, color="gray", lw=0.5, alpha=0.25, zorder=1)
        i = j
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.set_yscale("log")
    ax.set_xlabel("robots, ordered by model size nv (dof)")
    ax.set_ylabel("wall-clock per call (us)")
    fname = {"fkjac": "frame Jacobians", "fk": "forward kinematics"}.get(func, func)
    ax.set_title(f"{fname}, B={B} (x N=8 stages per call)", fontsize=10)
    ax.grid(True, which="major", axis="y", ls=":", alpha=0.35)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS, f"{out_stem}.{ext}"), dpi=160,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.join(RESULTS, out_stem)}.png/.pdf ({len(rows)} robots)")


if __name__ == "__main__":
    plot_pair()
    plot_absolute(B=256)
    plot_absolute_ranked(B=256)
