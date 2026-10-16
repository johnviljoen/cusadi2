#!/usr/bin/env python
"""Paper figures assembled from the completed Menagerie campaign results
(results/<robot>/runs/<k>/: every quantity is the mean over the runs).

Standalone figures (composed manually afterwards):

  paper_speedup_vs_batch_<baseline>   speedup vs batch B per function: per robot
        and B, speedup = baseline wall / cusadi2 `full` wall (both Python-hosted,
        B*N work each). Median line per precision with a percentile band across
        the robot ensemble, raw per-robot points overlaid (y-range from the bands;
        outliers above are clipped, the raw max is printed); linear y, log x. The
        ensemble at each B is every robot with both values at that B (a baseline
        that ran out of memory at large B drops out there); n is printed under
        each tick when it varies. baseline = mjx (batched), warp (fk/jac, fp32),
        cpu (CasADi C, 16 OpenMP threads, with the GPU->CPU->GPU transfer),
        cpu_ideal (the same without the transfer: compute only), cusadi (the
        original cusadi's execution model: the campaign's `batch` config, one
        thread per instance on the horizon-unrolled per-function kernel).
  paper_speedup_vs_batch_all          the five baselines on one set of axes per
        function (median + band + raw points as above): colour = baseline, solid /
        filled = fp64, dashed / open = fp32; log y (`_linear` variant: linear y).
  paper_absolute_B<B>_<func>          wall-clock at one fixed B vs model size nv:
        cusadi2 full, MJX batched, mujoco-warp (fp32), one marker per robot.
  paper_absolute_ranked_B<B>_<func>   the same plus the original CusADi, robots ranked by cusadi2 fp32 wall.
  paper_dag_shape_speedup_<baseline>_<fp>   DAG shape (mean level width vs depth of
        the plain level plan, from results/dag/<robot>.json written by dag_shape.py)
        of every menagerie function, one panel per batch size, the point colour the
        speedup at that B (baseline / cusadi2 full, log colour scale shared across
        the panels). Marker = quantity: circle fk, square Jacobians, triangle Hessians.
        One figure per (baseline, precision) in DAG_SPEEDUP_CASES; warp has no Hessians.

    python examples/menagerie_bench/paper_plots.py
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("CUSADI2_RESULTS", os.path.join(HERE, "results"))   # as run.py
BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
FUNCS = [("fk", "forward kinematics", "fk"), ("fkjac", "frame Jacobians", "jac"), ("fkhess", "frame Hessians", "hess")]

# (dtype, fp tag, color) — the campaign's figure conventions
SERIES = [("float", "fp32", "C1"), ("double", "fp64", "C0")]
WARP_COLOR, CPU_COLOR, CPU_IDEAL_COLOR, MJX_COLOR = "#9467bd", "#8c564b", "#d4a017", "#1f77b4"
CUSADI_COLOR = "#2ca02c"
FRAX_COLOR = "#e377c2"
# the frax campaign lives in its own results root (run.py --frax-only with CUSADI2_RESULTS)
FRAX_RESULTS = os.environ.get("CUSADI2_FRAX_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_frax"))
FRAX_BASELINE = ("frax", "frax", FRAX_COLOR)
CUSADI_CONFIG = "batch"  # the campaign config that runs the original cusadi's execution model
BASELINES = [("mjx", "MuJoCo XLA", MJX_COLOR), ("warp", "MuJoCo Warp", WARP_COLOR),
             ("cpu_ideal", "CPU idealized", CPU_IDEAL_COLOR), ("cpu", "CPU + transfer", CPU_COLOR),
             ("cusadi", "CusADi", CUSADI_COLOR)]
BAND = (25, 75)          # percentile band around the median
# Menagerie renders for the absolute-ranked figure: robot -> (file under the robot's
# Menagerie directory, subject box (left, top, right, bottom) in pixels that the
# crop is grown around to the insets' common aspect ratio, label)
MENAGERIE = os.path.join(os.path.dirname(HERE), "mujoco_menagerie")
PHOTOS = {"universal_robots_ur5e": ("ur5e.png", (1280, 440, 2600, 1720), "UR5e"),
          "franka_fr3_v2": ("fr3v2.png", (90, 30, 720, 620), "Franka FR3"),
          "i2rt_yam": ("yam.png", (1480, 280, 2800, 1760), "I2RT YAM"),
          "unitree_go2": ("go2.png", (1150, 500, 2750, 1560), "Unitree Go2"),
          "shadow_hand": ("shadow_hand.png", (1080, 320, 3040, 1600), "Shadow Hand"),
          "unitree_g1": ("g1.png", (1480, 20, 2200, 2000), "Unitree G1")}
CPU_THREADS = 16


# ---------------------------------------------------------------------------
# readers: mean over runs/<k>/ (the legacy single top-level file is accepted)
# ---------------------------------------------------------------------------
def _run_files(rdir, name):
    return sorted(glob.glob(os.path.join(rdir, "runs", "*", name))) or \
        [p for p in [os.path.join(rdir, name)] if os.path.isfile(p)]


def _avg(acc):
    return {k: {b: sum(v) / len(v) for b, v in d.items()} for k, d in acc.items()}


def _cus_wall(rdir, dtype, func, config="full"):
    """{B: mean wall_us} of one cusadi2 config from the Python-hosted sweep."""
    acc = {}
    for path in _run_files(rdir, f"latency_py_{dtype}.csv"):
        for r in csv.DictReader(open(path)):
            if r["func"] == func and r["config"] == config:
                try:
                    acc.setdefault(int(r["batch"]), []).append(float(r["wall_us"]))
                except ValueError:
                    pass
    return {b: sum(v) / len(v) for b, v in acc.items()}


def _json_results(rdir, name):
    """{series: {B: mean seconds}} over the runs of one baseline json."""
    acc = {}
    for path in _run_files(rdir, name):
        try:
            res = json.load(open(path)).get("results") or {}
        except Exception:
            continue
        for key, by_b in res.items():
            for b, v in by_b.items():
                if v is not None:
                    acc.setdefault(key, {}).setdefault(int(b), []).append(v)
    return _avg(acc)


def _cpu_wall(rdir, dtype, key="seq_transfer_us", threads=CPU_THREADS):
    """{func: {B: mean us}} of the CPU reference (idealized: seq_us; transfer: seq_transfer_us)."""
    acc = {}
    for path in _run_files(rdir, f"cpu_{dtype}.json"):
        try:
            d = json.load(open(path)).get(key) or {}
        except Exception:
            continue
        for func, by_t in d.items():
            for b, v in (by_t.get(str(threads)) or {}).items():
                if v is not None:
                    acc.setdefault(func, {}).setdefault(int(b), []).append(v)
    return _avg(acc)


def _baseline_wall_us(rdir, baseline, dtype, fp, func, kind):
    """{B: baseline wall us} for one robot / precision / function, or {}."""
    if baseline == "mjx":
        return {b: v * 1e6 for b, v in (_json_results(rdir, f"mjx_{fp}.json").get(f"{kind}_batched") or {}).items()}
    if baseline == "warp":
        if dtype != "float" or kind == "hess":
            return {}
        return {b: v * 1e6 for b, v in (_json_results(rdir, "warp.json").get(kind) or {}).items()}
    if baseline == "cpu":
        return _cpu_wall(rdir, dtype).get(func) or {}
    if baseline == "cpu_ideal":
        return _cpu_wall(rdir, dtype, key="seq_us").get(func) or {}
    if baseline == "cusadi":
        return _cus_wall(rdir, dtype, func, CUSADI_CONFIG)
    if baseline == "frax":
        frdir = os.path.join(FRAX_RESULTS, os.path.basename(rdir))
        return {b: v * 1e6 for b, v in (_json_results(frdir, f"frax_{fp}.json").get(f"{kind}_batched") or {}).items()}
    raise ValueError(baseline)


def _robots():
    return sorted(r for r in os.listdir(RESULTS) if os.path.isdir(os.path.join(RESULTS, r, "runs")))


def _speedups(func, kind, baseline="mjx", config="full"):
    """{fp: {robot: {B: speedup}}} using Python-hosted walls on both sides
    (a robot drops out of a B where either wall is missing or NaN, e.g. a CPU
    run that timed out there)."""
    out = {}
    for dtype, fp, _ in SERIES:
        per = {}
        for robot in _robots():
            rdir = os.path.join(RESULTS, robot)
            cus = _cus_wall(rdir, dtype, func, config)
            base = _baseline_wall_us(rdir, baseline, dtype, fp, func, kind)
            sp = {B: base[B] / cus[B] for B in BATCHES
                  if np.isfinite(base.get(B, np.nan)) and np.isfinite(cus.get(B, np.nan)) and cus[B] > 0}
            if sp:
                per[robot] = sp
        out[fp] = per
    return out


# ---------------------------------------------------------------------------
# speedup vs batch: median + IQR band across robots, raw points overlaid
# ---------------------------------------------------------------------------
def _ensemble(per):
    """(xs, n, median, lo, hi) over the robots that have each B in {robot: {B: v}}."""
    xs, ns, med, lo, hi = [], [], [], [], []
    for B in BATCHES:
        vals = np.array([sp[B] for sp in per.values() if B in sp])
        if vals.size == 0:
            continue
        xs.append(B); ns.append(int(vals.size))
        med.append(float(np.median(vals)))
        lo.append(float(np.percentile(vals, BAND[0]))); hi.append(float(np.percentile(vals, BAND[1])))
    return xs, ns, med, lo, hi


def _panel(ax, data, title, color_by_fp, legend=False, raw=True, rng=None):
    """Per precision: median + IQR band over the robots that have each B; raw
    per-robot points in a light tint. Returns (max raw value, n per B per fp)."""
    rng = rng or np.random.default_rng(0)
    raw_max, band_max, counts = 0.0, 0.0, {}
    for _, fp, _ in SERIES:
        per = data.get(fp) or {}
        color = color_by_fp[fp]
        xs, ns, med, lo, hi = _ensemble(per)
        if raw:
            for B in xs:
                vals = np.array([sp[B] for sp in per.values() if B in sp])
                ax.scatter(np.full(vals.size, B), vals, s=5, color=color, alpha=0.16, lw=0, zorder=1.5)
                raw_max = max(raw_max, float(vals.max()))
        if not xs:
            continue
        counts[fp] = dict(zip(xs, ns))
        band_max = max(band_max, float(max(hi)))
        ax.fill_between(xs, lo, hi, color=color, alpha=0.20, lw=0, zorder=2)
        ax.plot(xs, med, "-o", color=color, ms=3.5, lw=1.8, zorder=3,
                label=f"{fp} (n={min(ns)}" + (f"–{max(ns)}" if min(ns) != max(ns) else "") + ")")
    ax.axhline(1.0, color="gray", lw=1, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(BATCHES, [str(b) for b in BATCHES])
    ax.set_xlim(BATCHES[0] / 1.35, BATCHES[-1] * 1.35)
    ax.set_xlabel("batch size B")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", ls=":", alpha=0.4)
    # per-batch ensemble size under the ticks when it varies along the axis
    for fp, by_b in counts.items():
        if len(set(by_b.values())) > 1:
            for B, n in by_b.items():
                ax.annotate(f"{n}", (B, 0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=6,
                            color=color_by_fp[fp], alpha=0.8)
            break
    if legend:
        ax.legend(fontsize=8, title="precision", title_fontsize=8)
    return raw_max, band_max, counts


def _panels_extent(fig, axes):
    """(bottom, top) of the panels with their labels and titles, in figure fractions:
    square panels do not fill the figure height, so the suptitle and the shared
    legend are anchored to the panels instead of the figure edge."""
    fig.canvas.draw()
    boxes = [ax.get_tightbbox().transformed(fig.transFigure.inverted()) for ax in axes]
    return min(b.y0 for b in boxes), max(b.y1 for b in boxes)


def plot_speedup_vs_batch(baseline="mjx", ymax=None, config="full"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = {"mjx": "MuJoCo XLA", "warp": "MuJoCo Warp", "cpu": "CPU", "cpu_ideal": "CPU idealized",
             "cusadi": "CusADi"}[baseline]
    note = {"mjx": "", "warp": " (fp32, no autodiff)",
            "cpu": f" (CasADi C codegen, {CPU_THREADS} OpenMP threads, GPU→CPU→GPU transfer)",
            "cpu_ideal": f" (CasADi C codegen, {CPU_THREADS} OpenMP threads, compute only, no transfer)",
            "cusadi": " (original execution model: one thread per instance, horizon-unrolled per-function kernel)"}[baseline]
    funcs = [f for f in FUNCS if not (baseline == "warp" and f[2] == "hess")]
    color_by_fp = {fp: (WARP_COLOR if baseline == "warp" else c) for _, fp, c in SERIES}
    fig, axes = plt.subplots(1, len(funcs), figsize=(4.4 * len(funcs), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_box_aspect(1)
    m, bm = 0.0, 0.0
    for i, (ax, (func, title, kind)) in enumerate(zip(axes, funcs)):
        data = _speedups(func, kind, baseline=baseline, config=config)
        mx, bx, counts = _panel(ax, data, title, color_by_fp, legend=(i == 0))
        m, bm = max(m, mx), max(bm, bx)
        print(f"  {baseline} {func}: robots per B " + "; ".join(f"{fp}: {min(c.values())}-{max(c.values())}" for fp, c in counts.items()))
    axes[0].set_ylabel(f"speedup  ({label} / cusadi2 {config})", fontsize=9)
    # y-range from the IQR bands (raw outliers above it are clipped; the max is printed)
    axes[0].set_ylim(0, ymax if ymax is not None else bm * 1.6)
    fig.tight_layout()
    if note:
        fig.suptitle(f"{label}{note}", fontsize=9, y=_panels_extent(fig, axes)[1] + 0.01, va="bottom")
    stem = os.path.join(RESULTS, f"paper_speedup_vs_batch_{baseline}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf  (raw max {m:.1f})")


def plot_speedup_vs_batch_all(yscale="log", config="full", baselines=BASELINES, stem="paper_speedup_vs_batch_all",
                              fps=("fp32", "fp64")):
    """All baselines on one set of axes per function: colour = baseline, solid
    line + filled marker = fp64, dashed + open = fp32. Median + IQR band per
    series with the raw per-robot points in a light tint (y-range from the
    bands; outliers above are clipped, the raw max is printed). With a single
    precision in `fps` every series is solid + filled and the labels drop it;
    the stem then gets a `_<fp>` suffix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [sr for sr in SERIES if sr[1] in fps]
    single = len(series) == 1
    style = {"fp64": ("-", "o", True), "fp32": ("-", "o", True) if single else ("--", "o", False)}
    if single:
        stem = f"{stem}_{series[0][1]}"
    fig, axes = plt.subplots(1, len(FUNCS), figsize=(4.4 * len(FUNCS), 5.2), sharey=True)
    for ax in axes:
        ax.set_box_aspect(1)
    band_max, band_min, raw_max = 0.0, np.inf, 0.0
    legend = {}                                    # baseline label -> [(handle, label)] of the first panel
    for i, (ax, (func, title, kind)) in enumerate(zip(axes, FUNCS)):
        for baseline, blabel, color in baselines:
            data = _speedups(func, kind, baseline=baseline, config=config)
            for _, fp, _ in series:
                per = data.get(fp) or {}
                xs, ns, med, lo, hi = _ensemble(per)
                if not xs:
                    continue
                ls, mk, filled = style[fp]
                for B in xs:
                    vals = np.array([sp[B] for sp in per.values() if B in sp])
                    ax.scatter(np.full(vals.size, B), vals, s=5, color=color, alpha=0.12, lw=0, zorder=1.5)
                    raw_max = max(raw_max, float(vals.max()))
                n = f"{min(ns)}" + (f"–{max(ns)}" if min(ns) != max(ns) else "")
                ax.fill_between(xs, lo, hi, color=color, alpha=0.13, lw=0, zorder=2)
                line, = ax.plot(xs, med, ls, marker=mk, color=color, ms=3.5, lw=1.6, zorder=3,
                                markerfacecolor=color if filled else "white", markeredgewidth=1.2)
                if i == 0:
                    legend.setdefault(blabel, []).append((line, blabel if single else f"{blabel} {fp} (n={n})"))
                band_max, band_min = max(band_max, max(hi)), min(band_min, min(lo))
                print(f"  all {func} {baseline} {fp}: robots per B {min(ns)}-{max(ns)}, "
                      f"median {min(med):.1f}-{max(med):.1f}")
        ax.axhline(1.0, color="gray", lw=1, ls=":", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(BATCHES, [str(b) for b in BATCHES])
        ax.set_xlim(BATCHES[0] / 1.35, BATCHES[-1] * 1.35)
        ax.set_xlabel("batch size B")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="major", ls=":", alpha=0.4)
    axes[0].set_ylabel("speedup", fontsize=9)
    axes[0].set_yscale(yscale)
    if yscale == "log":
        lo_lim = min(0.7, band_min / 1.3)
        axes[0].set_ylim(lo_lim, band_max * 1.8)
        ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if lo_lim <= t <= band_max * 1.8]
        axes[0].set_yticks(ticks, [f"{t:g}" for t in ticks])
        axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        axes[0].set_ylim(0, band_max * 1.15)
    fig.tight_layout()
    bottom, _ = _panels_extent(fig, axes)
    # one legend under the square panels (entries and n from the first panel), a
    # column per baseline; the single-precision ones last so the columns stay aligned
    cols = sorted(legend.values(), key=lambda c: -len(c))
    if single:   # the absolute-ranked figure's order: GPU baselines, then the CPU ones
        order = [l for _, _, _, l, _, _ in _absolute_series(fps)]
        cols = sorted(legend.values(), key=lambda c: order.index(c[0][1]) if c[0][1] in order else len(order))
    rows_ = 1 if single else 2
    entries = [e for c in cols for e in c + [(matplotlib.lines.Line2D([], [], ls=""), "")] * (rows_ - len(c))]
    fig.legend([h for h, _ in entries], [l for _, l in entries], fontsize=8, ncols=len(cols), loc="upper center",
               bbox_to_anchor=(0.5, bottom - 0.01), columnspacing=1.6, handlelength=2.4, frameon=False)
    stem = os.path.join(RESULTS, stem + ("_linear" if yscale == "linear" else ""))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf  (raw max {raw_max:.1f})")


# ---------------------------------------------------------------------------
# absolute wall-clock at one fixed B vs model size
# ---------------------------------------------------------------------------
def _nv_by_robot():
    nv = {}
    for robot in _robots():
        try:
            gate = json.load(open(os.path.join(RESULTS, robot, "status.json"))).get("gate") or {}
        except Exception:
            continue
        if gate.get("status") == "pass":
            nv[robot] = int(gate["nv"])
    return nv


def _absolute_rows(B, func, kind):
    """{robot: {nv, cus_<dtype>, cusadi_<fp>, cpu_<fp>, cpu_ideal_<fp>, frax_<fp>, mjx_<fp>, warp}}
    walls (us) at batch B, None where absent."""
    rows = {}
    for robot, n in _nv_by_robot().items():
        rdir = os.path.join(RESULTS, robot)
        d = {"nv": n}
        for dtype, fp, _ in SERIES:
            d[f"cus_{dtype}"] = _cus_wall(rdir, dtype, func).get(B)
            d[f"mjx_{fp}"] = _baseline_wall_us(rdir, "mjx", dtype, fp, func, kind).get(B)
            d[f"cusadi_{fp}"] = _baseline_wall_us(rdir, "cusadi", dtype, fp, func, kind).get(B)
            d[f"cpu_{fp}"] = _baseline_wall_us(rdir, "cpu", dtype, fp, func, kind).get(B)
            d[f"cpu_ideal_{fp}"] = _baseline_wall_us(rdir, "cpu_ideal", dtype, fp, func, kind).get(B)
            d[f"frax_{fp}"] = _baseline_wall_us(rdir, "frax", dtype, fp, func, kind).get(B)
        d["warp"] = _baseline_wall_us(rdir, "warp", "float", "fp32", func, kind).get(B)
        rows[robot] = d
    return rows


def _absolute_series(fps=("fp32", "fp64")):
    """(key, linestyle, marker, label, color, filled) in draw order, for the
    precisions in `fps`; the precision is left off the labels when there is one."""
    series = [(d, fp, c) for d, fp, c in SERIES if fp in fps]
    tag = (lambda fp: f" {fp}") if len(series) > 1 else (lambda fp: "")
    out = []
    for dtype, fp, color in series:
        out.append((f"cus_{dtype}", "-", "o", f"CusADi2{tag(fp)}", color, True))
        out.append((f"mjx_{fp}", "--", "^", f"MuJoCo XLA{tag(fp)}", color, False))
    if "fp32" in fps:
        out.append(("warp", "-.", "D", f"MuJoCo Warp{tag('fp32')}", WARP_COLOR, False))
    for _, fp, _ in series:
        out.append((f"cusadi_{fp}", ":", "s", f"CusADi{tag(fp)}", CUSADI_COLOR, fp == "fp32"))
    for _, fp, _ in series:
        out.append((f"frax_{fp}", "--", "*", f"frax{tag(fp)}", FRAX_COLOR, fp == "fp32"))
    for _, fp, _ in series:
        out.append((f"cpu_ideal_{fp}", (0, (3, 1, 1, 1)), "v", f"CPU idealized{tag(fp)}", CPU_IDEAL_COLOR, fp == "fp32"))
    for _, fp, _ in series:
        out.append((f"cpu_{fp}", (0, (3, 1, 1, 1)), "P", f"CPU + transfer{tag(fp)}", CPU_COLOR, fp == "fp32"))
    return out


def plot_absolute(B=256, func="fkjac", yscale="log"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind = dict((f, k) for f, _, k in FUNCS)[func]
    title = dict((f, t) for f, t, _ in FUNCS)[func]
    rows = _absolute_rows(B, func, kind)
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    counts = {}
    for key, ls, mk, label, color, filled in _absolute_series():
        pts = sorted((d["nv"], d[key]) for d in rows.values() if d.get(key) is not None)
        if not pts:
            continue
        counts[label] = len(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle=ls, marker=mk, color=color, ms=3.5, lw=1.2,
                alpha=0.9, label=label, markerfacecolor=color if filled else "none")
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    if yscale == "linear":
        ax.set_ylim(bottom=0)
    ticks = sorted({d["nv"] for d in rows.values()})
    ax.set_xticks([t for t in ticks if t in (2, 6, 8, 18, 30, 50, 70, 108)] or ticks[::4],
                  [str(t) for t in ticks if t in (2, 6, 8, 18, 30, 50, 70, 108)] or [str(t) for t in ticks[::4]])
    ax.set_xlabel("model size nv (dof)")
    ax.set_ylabel("wall-clock per call (us)")
    ax.set_title(f"{title}, B={B} (x N=8 stages per call)", fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=8, ncols=3)
    fig.tight_layout()
    stem = os.path.join(RESULTS, f"paper_absolute_B{B}_{func}" + ("_linear" if yscale == "linear" else ""))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf  robots per series: {counts}")


def _crop_to_aspect(im, box, aspect):
    """Grow `box` (left, top, right, bottom) about its centre to width/height
    `aspect`, clamped to the image, and return the crop."""
    l, t, r, b = box
    w, h = r - l, b - t
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (l + r) / 2, (t + b) / 2
    w, h = min(w, im.width), min(h, im.height)
    l = min(max(cx - w / 2, 0), im.width - w)
    t = min(max(cy - h / 2, 0), im.height - h)
    return im.crop((int(l), int(t), int(l + w), int(t + h)))


def _photo_insets(fig, ax, rows, photos, size_in=(0.85, 0.85), gap=0.04):
    """A fine vertical line through the panel at each photographed robot, and its
    render (cropped to the aspect ratio of `size_in` around the subject box,
    drawn at exactly `size_in` inches, labelled) sitting on the top end of that
    line, `gap` axes-fractions above the top spine."""
    from PIL import Image
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker
    from matplotlib.transforms import blended_transform_factory
    dpi = fig.dpi
    x_of = {r[2]: i for i, r in enumerate(rows)}
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for robot, (fname, box, label) in photos.items():
        if robot not in x_of:
            continue
        x = x_of[robot]
        ax.plot([x, x], [0, 1 + gap], color="0.35", lw=0.6, transform=tr, clip_on=False, zorder=1.2)
        im = Image.open(os.path.join(MENAGERIE, robot, fname)).convert("RGB")
        im = _crop_to_aspect(im, box, size_in[0] / size_in[1])
        im = im.resize((int(size_in[0] * dpi * 2), int(size_in[1] * dpi * 2)), Image.LANCZOS)
        packed = VPacker(children=[OffsetImage(np.asarray(im), zoom=0.5),
                                   TextArea(label, textprops=dict(fontsize=8, ha="center"))],
                         align="center", pad=0, sep=2)
        ab = AnnotationBbox(packed, (x, 1 + gap), xycoords=tr, box_alignment=(0.5, 0),
                            frameon=True, pad=0.2, bboxprops=dict(edgecolor="0.6", lw=0.6),
                            annotation_clip=False, zorder=5)
        ax.add_artist(ab)


def plot_absolute_ranked(B=256, func="fkjac", photos=PHOTOS, fps=("fp32", "fp64")):
    """Robots on a categorical x-axis ranked by (nv, cusadi2 fp32 wall) with nv
    group boundaries — the same series as plot_absolute. `photos` (robot ->
    (Menagerie render, crop box, label)) are drawn as labelled insets above the
    panel, each with an arrow to that robot's cusadi2 fp32 marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind = dict((f, k) for f, _, k in FUNCS)[func]
    title = dict((f, t) for f, t, _ in FUNCS)[func]
    rows = [(d["nv"], d["cus_float"], robot, d) for robot, d in _absolute_rows(B, func, kind).items()
            if d.get("cus_float") is not None]
    rows.sort()
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.6, 2.6))
    for key, ls, mk, label, color, filled in _absolute_series(fps):
        y = np.array([r[3][key] if r[3].get(key) is not None else np.nan for r in rows], dtype=float)
        if np.all(np.isnan(y)):
            continue
        ax.plot(x, y, linestyle=ls, marker=mk, color=color, ms=3.5, lw=1.2, alpha=0.9, label=label,
                markerfacecolor=color if filled else "none")
    ticks, labels, i = [], [], 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1
        ticks.append((i + j - 1) / 2); labels.append(str(rows[i][0]))
        if j < len(rows):
            ax.axvline(j - 0.5, color="gray", lw=0.5, alpha=0.25, zorder=1)
        i = j
    ax.set_xticks(ticks, labels, fontsize=7)
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.set_yscale("log")
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("latency (us)")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.35)
    fig.tight_layout()
    if photos:
        _photo_insets(fig, ax, rows, photos)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=6 if len(handles) > 7 else 7, ncols=len(handles), loc="upper center",
               columnspacing=0.7 if len(handles) > 7 else 1.0, handlelength=1.4 if len(handles) > 7 else 2.0,
               handletextpad=0.35 if len(handles) > 7 else 0.5,
               bbox_to_anchor=(0.5, _panels_extent(fig, [ax])[0] - 0.02), frameon=False)
    stem = os.path.join(RESULTS, f"paper_absolute_ranked_B{B}_{func}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# DAG shape coloured by speedup, one panel per batch size
# ---------------------------------------------------------------------------
DAG_DIR = os.environ.get("CUSADI2_DAG_DIR", os.path.join(RESULTS, "dag"))
KIND_MARKER = {"fk": "o", "fkjac": "s", "fkhess": "^"}
LABEL_ROBOTS = {"unitree_a1": "A1", "unitree_h1": "H1"}
DAG_SPEEDUP_CASES = [("mjx", "fp32"), ("mjx", "fp64"), ("warp", "fp32"), ("cusadi", "fp32"), ("cusadi", "fp64"),
                     ("cpu", "fp32"), ("cpu", "fp64"), ("cpu_ideal", "fp32"), ("cpu_ideal", "fp64")]


def _dag_shape():
    """{robot: {func: (mean_width, depth)}} from the dag_shape.py cache."""
    out = {}
    for path in glob.glob(os.path.join(DAG_DIR, "*.json")):
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        fns = {f: (r["mean_width"], r["depth"]) for f, r in (rec.get("functions") or {}).items()
               if "mean_width" in r and "depth" in r}
        if fns:
            out[rec["robot"]] = fns
    return out


def plot_dag_shape_speedup(baseline="mjx", fp="fp32", batches=(16, 64, 256), config="full", annotate=False):
    """annotate: name the LABEL_ROBOTS points (A1 / H1) next to their markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    blabel = dict((b, l) for b, l, _ in BASELINES + [FRAX_BASELINE])[baseline]
    shape = _dag_shape()
    sp_by_func = {func: _speedups(func, kind, baseline=baseline, config=config)[fp] for func, _, kind in FUNCS}
    # points: (B, func, robot, mean_width, depth, speedup)
    pts = [(B, func, robot, *shape[robot][func], sp[B])
           for B in batches for func, _, _ in FUNCS
           for robot, sp in sp_by_func[func].items() if B in sp and func in shape.get(robot, {})]
    if not pts:
        raise SystemExit(f"no points: is {DAG_DIR} populated (menagerie_bench/dag_shape.py)?")
    vals = np.array([p[5] for p in pts])
    norm = LogNorm(vmin=float(np.percentile(vals, 2)), vmax=float(np.percentile(vals, 98)))
    cmap = plt.get_cmap("coolwarm")

    fig, axes = plt.subplots(1, len(batches), figsize=(4.4 * len(batches) + 0.8, 4.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, B in zip(axes, batches):
        here = [p for p in pts if p[0] == B]
        n = {func: 0 for func, _, _ in FUNCS}
        for _, func, robot, w, d, v in sorted(here, key=lambda p: p[5]):     # fast on top
            n[func] += 1
            ax.scatter(w, d, c=[v], cmap=cmap, norm=norm, marker=KIND_MARKER[func], s=26,
                       edgecolor="#444", lw=0.35, zorder=3)
            if annotate and robot in LABEL_ROBOTS:
                ax.annotate(f"{LABEL_ROBOTS[robot]} {func}", (w, d), textcoords="offset points",
                            xytext=(5, 3), fontsize=6, color="#444", zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
        ax.set_xlabel("mean width")
        ax.set_title(f"B = {B}", fontsize=10)
        ax.grid(True, which="both", color="#e5e5e5", lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        print(f"  dag_shape {baseline} {fp} B={B}: " + ", ".join(f"{f}: {k}" for f, k in n.items()))
    axes[0].set_ylabel("depth")
    yt = [t for t in (2, 5, 10, 20, 50, 100) if axes[0].get_ylim()[0] <= t <= axes[0].get_ylim()[1]]
    axes[0].set_yticks(yt, [str(t) for t in yt])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    present = {p[1] for p in pts}
    axes[0].legend(handles=[Line2D([], [], ls="", marker=KIND_MARKER[f], color="#888", markeredgecolor="#444",
                                   markersize=6, label=t) for f, t, _ in FUNCS if f in present],
                   frameon=False, fontsize=8, loc="upper left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, pad=0.015, fraction=0.03, aspect=28)
    cb.set_label("speedup", fontsize=9)
    ct = [t for t in (0.5, 1, 2, 3, 5, 10, 15, 20, 30, 50) if norm.vmin <= t <= norm.vmax]
    cb.set_ticks(ct, labels=[f"{t:g}" for t in ct])
    cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    stem = os.path.join(RESULTS, f"paper_dag_shape_speedup_{baseline}_{fp}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf  (speedup range {vals.min():.2f}-{vals.max():.1f}, "
          f"colour {norm.vmin:.2f}-{norm.vmax:.1f})")


if __name__ == "__main__":
    for baseline in ("mjx", "warp", "cpu", "cpu_ideal", "cusadi"):
        plot_speedup_vs_batch(baseline)
    for yscale in ("log", "linear"):
        plot_speedup_vs_batch_all(yscale=yscale)
    for B in (256, 16384):
        for func, _, _ in FUNCS:
            plot_absolute(B=B, func=func)
    plot_absolute_ranked(B=256, func="fk", fps=("fp32",))
    for baseline, fp in DAG_SPEEDUP_CASES:
        plot_dag_shape_speedup(baseline=baseline, fp=fp)
