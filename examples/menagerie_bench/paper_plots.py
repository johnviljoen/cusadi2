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
        of every menagerie function, one panel per batch size (B = 64 and 256; the
        one-column figure, Palatino, saved exactly columnwidth wide with the printed
        sizes of PRINT_PT), the point colour the speedup at that B (baseline / cusadi2
        full, log colour scale shared across the panels). Marker = quantity: circle fk,
        square Jacobians, triangle Hessians. One figure per (baseline, precision) in
        DAG_SPEEDUP_CASES; warp has no Hessians.

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
                              fps=("fp32", "fp64"), height_in=None, palatino=True, print_pt=None, textwidth_in=7.16):
    """All baselines on one set of axes per function: colour = baseline, solid
    line + filled marker = fp64, dashed + open = fp32. Median + IQR band per
    series with the raw per-robot points in a light tint (y-range from the
    bands; outliers above are clipped, the raw max is printed). With a single
    precision in `fps` every series is solid + filled and the labels drop it;
    the stem then gets a `_<fp>` suffix. height_in: the saved figure's height
    (tight bbox, iterated on the panels' box aspect; the square-panel default
    saves 4.98 in high at 13.07 in wide); palatino: paper_fonts.use_palatino;
    print_pt: the size every text element (titles, axis labels, ticks, legend)
    PRINTS at when the figure spans `textwidth_in` (\\includegraphics[width=
    \\textwidth]): the drawn size is print_pt x saved width / textwidth, found
    from the saved PDF's width and iterated with the height. None keeps
    matplotlib's defaults (10 / 9 / 8 pt as drawn, i.e. 5.5 / 4.9 / 4.4 pt in print)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if palatino:
        from paper_fonts import use_palatino
        use_palatino()

    series = [sr for sr in SERIES if sr[1] in fps]
    single = len(series) == 1
    style = {"fp64": ("-", "o", True), "fp32": ("-", "o", True) if single else ("--", "o", False)}
    if single:
        stem = f"{stem}_{series[0][1]}"
    stem = os.path.join(RESULTS, stem + ("_linear" if yscale == "linear" else ""))
    box_aspect, font_pt, saved_w = 1.0, None, 4.4 * len(FUNCS)
    for attempt in range(8):
        if print_pt is not None:
            font_pt = print_pt * saved_w / textwidth_in
        fig, raw_max = _draw_speedup_vs_batch_all(plt, matplotlib, yscale, config, baselines, series, single, style, box_aspect, fps, font_pt)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        saved_w, h = pdf_width_in(f"{stem}.pdf"), pdf_width_in(f"{stem}.pdf", height=True)
        font_ok = print_pt is None or abs(font_pt * textwidth_in / saved_w - print_pt) < 0.02
        print(f"  box aspect {box_aspect:.3f}, font {font_pt or 'default'} pt -> saved {saved_w:.3f} x {h:.3f} in"
              + (f" (printed font {font_pt * textwidth_in / saved_w:.2f} pt, target {print_pt})" if print_pt is not None else "") + (f" (target height {height_in})" if height_in else ""))
        if (height_in is None or abs(h - height_in) < 0.01) and font_ok:
            break
        plt.close(fig)
        if height_in is None:
            continue
        # the non-panel height (title, x label, legend) is fixed: scale only the panel part
        panel_h = box_aspect * (13.07 / len(FUNCS)) * 0.78          # ~ the panel height at this aspect (figsize 4.4 in per panel, tight_layout)
        box_aspect = max(0.05, box_aspect * (panel_h - (h - height_in)) / panel_h)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf  (raw max {raw_max:.1f}, {pdf_width_in(stem + '.pdf'):.2f} x {pdf_width_in(stem + '.pdf', height=True):.2f} in)")


def _draw_speedup_vs_batch_all(plt, matplotlib, yscale, config, baselines, series, single, style, box_aspect, fps, font_pt=None):
    f = dict(title=10, label=9, legend=8) if font_pt is None else dict(title=font_pt, label=font_pt, legend=font_pt)   # drawn sizes
    if font_pt is not None:
        plt.rcParams.update({"font.size": font_pt, "axes.titlesize": font_pt, "axes.labelsize": font_pt, "xtick.labelsize": font_pt, "ytick.labelsize": font_pt, "legend.fontsize": font_pt})
    fig, axes = plt.subplots(1, len(FUNCS), figsize=(4.4 * len(FUNCS), 5.2 * max(box_aspect, 0.4)), sharey=True)
    for ax in axes:
        ax.set_box_aspect(box_aspect)
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
        ax.set_title(title, fontsize=f["title"])
        ax.grid(True, which="major", ls=":", alpha=0.4)
    axes[0].set_ylabel("speedup", fontsize=f["label"])
    axes[0].set_yscale(yscale)
    if yscale == "log":
        lo_lim = min(0.7, band_min / 1.3)
        axes[0].set_ylim(lo_lim, band_max * 1.8)
        cand = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if box_aspect >= 0.6 else (0.1, 0.3, 1, 3, 10, 30, 100, 300)   # flat panels: half the labels
        ticks = [t for t in cand if lo_lim <= t <= band_max * 1.8]
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
    fig.legend([h for h, _ in entries], [l for _, l in entries], fontsize=f["legend"], ncols=len(cols), loc="upper center",
               bbox_to_anchor=(0.5, bottom - 0.01), columnspacing=1.6, handlelength=2.4, frameon=False)
    return fig, raw_max


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


INSET_BLUE = "#2f6f9f"          # the photo frames and their guide lines (John's Inkscape mock-up)
GROUP_GREY = "0.45"             # the DoF under-brackets
SEP_GREY = "0.6"                # the vertical DoF-group separators


def _photo_insets(fig, ax, rows, photos, size_in=(0.85, 0.85), gap=0.04, font_pt=8):
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
        ax.plot([x, x], [0, 1 + gap], color=INSET_BLUE, lw=0.7, transform=tr, clip_on=False, zorder=1.2)
        im = Image.open(os.path.join(MENAGERIE, robot, fname)).convert("RGB")
        im = _crop_to_aspect(im, box, size_in[0] / size_in[1])
        im = im.resize((int(size_in[0] * dpi * 2), int(size_in[1] * dpi * 2)), Image.LANCZOS)
        packed = VPacker(children=[OffsetImage(np.asarray(im), zoom=0.5),
                                   TextArea(label, textprops=dict(fontsize=font_pt, ha="center"))],
                         align="center", pad=0, sep=2)
        ab = AnnotationBbox(packed, (x, 1 + gap), xycoords=tr, box_alignment=(0.5, 0),
                            frameon=True, pad=0.2, bboxprops=dict(edgecolor=INSET_BLUE, lw=0.7),
                            annotation_clip=False, zorder=5)
        ax.add_artist(ab)


def plot_absolute_ranked(B=256, func="fkjac", photos=PHOTOS, fps=("fp32", "fp64"), print_pt=None, textwidth_in=7.16, palatino=True, legend_inside=False):
    """Robots on a categorical x-axis ranked by (nv, cusadi2 fp32 wall) with nv
    group boundaries — the same series as plot_absolute. `photos` (robot ->
    (Menagerie render, crop box, label)) are drawn as labelled insets above the
    panel, each with an arrow to that robot's cusadi2 fp32 marker. print_pt: the
    size every text element PRINTS at when the banner spans `textwidth_in`
    (drawn size = print_pt x saved width / textwidth, from the saved PDF's width,
    iterated); None keeps the defaults (10 / 7 / 7 / 8 pt as drawn). legend_inside:
    the legend in the empty upper-left corner of the panel (two rows of four, white
    backing, the y axis given headroom) instead of a row beneath the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if palatino:
        from paper_fonts import use_palatino
        use_palatino()
    stem = os.path.join(RESULTS, f"paper_absolute_ranked_B{B}_{func}")
    font_pt, saved_w = None, 10.6
    for attempt in range(5):
        if print_pt is not None:
            font_pt = print_pt * saved_w / textwidth_in
        fig = _draw_absolute_ranked(plt, matplotlib, B, func, photos, fps, font_pt, legend_inside)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        saved_w = pdf_width_in(f"{stem}.pdf")
        if print_pt is None or abs(font_pt * textwidth_in / saved_w - print_pt) < 0.02:
            break
        print(f"  font {font_pt:.2f} pt -> saved {saved_w:.3f} in wide, printed font {font_pt * textwidth_in / saved_w:.2f} pt (target {print_pt})")
        plt.close(fig)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf  ({saved_w:.2f} x {pdf_width_in(stem + '.pdf', height=True):.2f} in"
          + (f", text {font_pt:.2f} pt drawn = {print_pt} pt printed at {textwidth_in} in" if print_pt is not None else "") + ")")


def _draw_absolute_ranked(plt, matplotlib, B, func, photos, fps, font_pt=None, legend_inside=False):
    # print_pt sizes the axis labels, legend and inset captions; the tick numbers keep their original drawn sizes (x 7 pt,
    # y matplotlib's 10 pt): at 8 pt printed the dense DoF groups (22 23 24 25) would need two rows, which John does not want.
    f = dict(label=10, tick=7, legend=None, inset=8) if font_pt is None else dict(label=font_pt, tick=7, legend=font_pt, inset=font_pt)
    if font_pt is not None:
        plt.rcParams.update({"font.size": font_pt, "axes.labelsize": font_pt, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": font_pt})
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
    groups, i = [], 0                                            # (first index, last index, nv) of each DoF group
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j][0] == rows[i][0]:
            j += 1
        groups.append((i, j - 1, rows[i][0]))
        if j < len(rows):
            ax.axvline(j - 0.5, color=SEP_GREY, lw=0.6, zorder=1)
        i = j
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.set_xticks([])
    ax.set_yscale("log")
    if legend_inside:                                            # headroom for the legend box above the data
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 2.2)
    ax.set_xlabel("degrees of freedom", labelpad=17)
    ax.set_ylabel("latency (us)")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.35)
    fig.tight_layout()
    # grey under-brackets beneath the axis, one per DoF group, the nv value under each (in place of tick marks + labels)
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax_h_pt = ax.get_window_extent().height / fig.dpi * 72
    pt = lambda p: p / ax_h_pt                                   # points below the x axis -> axes fraction
    for i0, i1, nv in groups:
        x0, x1 = i0 - 0.38, i1 + 0.38
        ax.plot([x0, x0, x1, x1], [pt(-3), pt(-6), pt(-6), pt(-3)], color=GROUP_GREY, lw=0.7, transform=tr, clip_on=False, zorder=2, solid_joinstyle="miter")
        ax.text((i0 + i1) / 2, pt(-8), str(nv), transform=tr, ha="center", va="top", fontsize=f["tick"], clip_on=False)
    if photos:
        _photo_insets(fig, ax, rows, photos, font_pt=f["inset"])
    handles, labels_ = ax.get_legend_handles_labels()
    if legend_inside:
        ax.legend(handles, labels_, fontsize=f["legend"] or 7, ncols=4, loc="upper left", columnspacing=1.0, handlelength=1.8, handletextpad=0.4,
                  labelspacing=0.3, borderaxespad=0.3, frameon=True, framealpha=0.75, facecolor="white", edgecolor="none", fancybox=False).set_zorder(6)   # translucent: the blue guide lines show through
    else:
        fig.legend(handles, labels_, fontsize=f["legend"] or (6 if len(handles) > 7 else 7), ncols=len(handles), loc="upper center",
                   columnspacing=0.7 if len(handles) > 7 else 1.0, handlelength=1.4 if len(handles) > 7 else 2.0,
                   handletextpad=0.35 if len(handles) > 7 else 0.5,
                   bbox_to_anchor=(0.5, _panels_extent(fig, [ax])[0] - 0.02), frameon=False)
    return fig


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


# Printed sizes of the one-column figures (plot_dag_shape_speedup below and panda_safe_mjpc/plot_speedup_paper.py), which
# are saved exactly `columnwidth` wide so \includegraphics[width=\columnwidth] scales them by 1: text at PRINT_FONT_PT
# (the footnote size, as the full-width figures' print_pt), lines and markers at the printed weight of the DAG-shape
# figure's original two-column form (drawn 11.75 in wide with matplotlib's defaults and printed at a 7.16 in text width,
# i.e. x 0.609).
REF_WIDTH_IN = 11.75                                        # the two-column original's saved width
PRINT_SCALE = 7.16 / REF_WIDTH_IN
PRINT_FONT_PT = 8.0                                         # every text element of the paper's figures prints at the footnote size (IEEEtran 8 pt)
PRINT_PT = dict(label=PRINT_FONT_PT, tick=PRINT_FONT_PT, title=PRINT_FONT_PT, legend=PRINT_FONT_PT, cbar=PRINT_FONT_PT,
                annotate=0.75 * PRINT_FONT_PT, grid_lw=0.5 * PRINT_SCALE, marker_edge_lw=0.35 * PRINT_SCALE, marker_area=26.0 * PRINT_SCALE ** 2,
                legend_marker=6.0 * PRINT_SCALE, axes_lw=0.8 * PRINT_SCALE, tick_len=3.5 * PRINT_SCALE, tick_w=0.8 * PRINT_SCALE,
                tick_pad=3.5 * PRINT_SCALE, labelpad=4.0 * PRINT_SCALE)


def pdf_width_in(path, height=False):
    """Page width (or height) of a PDF in inches (pdfinfo 'Page size: W x H pts')."""
    import re, subprocess
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True).stdout)
    return float(m.group(2 if height else 1)) / 72.0


def plot_dag_shape_speedup(baseline="mjx", fp="fp32", batches=(64, 256), config="full", annotate=False, columnwidth=3.5):
    """One-column version: one panel per batch in `batches` (the B = 16 panel of the two-column original dropped), Palatino,
    the PRINT_PT sizes, saved exactly `columnwidth` inches wide (tight bbox, iterated) for \includegraphics[width=\columnwidth].
    annotate: name the LABEL_ROBOTS points (A1 / H1) next to their markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from paper_fonts import use_palatino
    use_palatino()
    P = PRINT_PT

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
    present = {p[1] for p in pts}
    counts = {}
    rc = {"font.size": P["label"], "axes.labelsize": P["label"], "xtick.labelsize": P["tick"], "ytick.labelsize": P["tick"], "legend.fontsize": P["legend"],
          "axes.linewidth": P["axes_lw"], "xtick.major.width": P["tick_w"], "ytick.major.width": P["tick_w"], "xtick.major.size": P["tick_len"],
          "ytick.major.size": P["tick_len"], "xtick.minor.size": 0.57 * P["tick_len"], "ytick.minor.size": 0.57 * P["tick_len"],
          "xtick.minor.width": 0.75 * P["tick_w"], "ytick.minor.width": 0.75 * P["tick_w"], "xtick.major.pad": P["tick_pad"], "ytick.major.pad": P["tick_pad"],
          "axes.labelpad": P["labelpad"]}

    def draw(fig_w):
        fig, axes = plt.subplots(1, len(batches), figsize=(fig_w, fig_w * 0.5), sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, B in zip(axes, batches):
            here = [p for p in pts if p[0] == B]
            n = {func: 0 for func, _, _ in FUNCS}
            for _, func, robot, w, d, v in sorted(here, key=lambda p: p[5]):     # fast on top
                n[func] += 1
                ax.scatter(w, d, c=[v], cmap=cmap, norm=norm, marker=KIND_MARKER[func], s=P["marker_area"],
                           edgecolor="#444", lw=P["marker_edge_lw"], zorder=3)
                if annotate and robot in LABEL_ROBOTS:
                    ax.annotate(f"{LABEL_ROBOTS[robot]} {func}", (w, d), textcoords="offset points",
                                xytext=(5 * PRINT_SCALE, 3 * PRINT_SCALE), fontsize=P["annotate"], color="#444", zorder=4)
            counts[B] = n
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
            ax.set_xlabel("mean width")
            ax.set_title(f"B = {B}", fontsize=P["title"])
            ax.grid(True, which="both", color="#e5e5e5", lw=P["grid_lw"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].set_ylabel("depth")
        yt = [t for t in (2, 5, 10, 20, 50, 100) if axes[0].get_ylim()[0] <= t <= axes[0].get_ylim()[1]]
        axes[0].set_yticks(yt, [str(t) for t in yt])
        axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.04, aspect=24)
        cb.set_label("speedup", fontsize=P["cbar"])
        ct = [t for t in (0.5, 1, 2, 3, 5, 10, 15, 20, 30, 50) if norm.vmin <= t <= norm.vmax]
        cb.set_ticks(ct, labels=[f"{t:g}" for t in ct])
        cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        cb.ax.tick_params(labelsize=P["cbar"], length=0.7 * P["tick_len"], width=P["tick_w"], pad=0.6 * P["tick_pad"])
        cb.outline.set_linewidth(P["axes_lw"])
        # the marker legend beneath the panels (inside the first panel it covered the points at the footnote size)
        fig.canvas.draw()
        bottom = min(a.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0 for a in axes)
        fig.legend(handles=[Line2D([], [], ls="", marker=KIND_MARKER[f], color="#888", markeredgecolor="#444",
                                   markersize=P["legend_marker"], label=t) for f, t, _ in FUNCS if f in present],
                   frameon=False, fontsize=P["legend"], loc="upper center", ncol=3, columnspacing=0.7, handlelength=1.0, handletextpad=0.2, bbox_to_anchor=(0.5, bottom - 0.02))   # one row: tight spacing keeps the three entries within the column
        return fig

    stem = os.path.join(RESULTS, f"paper_dag_shape_speedup_{baseline}_{fp}")
    with matplotlib.rc_context(rc):
        fig_w = columnwidth
        for it in range(4):                                 # the tight bbox is narrower than figsize: iterate until the saved width == columnwidth
            fig = draw(fig_w); fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01); plt.close(fig)
            w = pdf_width_in(f"{stem}.pdf")
            if abs(w - columnwidth) < 0.005:
                break
            fig_w += columnwidth - w
        fig = draw(fig_w)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    for B, n in counts.items():
        print(f"  dag_shape {baseline} {fp} B={B}: " + ", ".join(f"{f}: {k}" for f, k in n.items()))
    print(f"wrote {stem}.png/.pdf  ({pdf_width_in(stem + '.pdf'):.3f} in wide = columnwidth {columnwidth}; speedup range {vals.min():.2f}-{vals.max():.1f}, "
          f"colour {norm.vmin:.2f}-{norm.vmax:.1f}; {blabel})")


def plot_speedup_dag_banner(fp="fp32", baseline_dag="cusadi", batches=(64, 256), config="full", baselines=None,
                            textwidth_in=7.16, gap=0.28, stem="paper_speedup_dag_banner", palatino=True):
    """Full-width banner for the paper: the three speedup-vs-batch panels of plot_speedup_vs_batch_all (one precision,
    every baseline) on the left, a gap, and the two DAG-shape panels of plot_dag_shape_speedup (baseline `baseline_dag`,
    `batches`) with their colour bar on the right. Five equal square panels; every text element at PRINT_FONT_PT, lines
    and markers at the printed weights of the one-column figures; saved exactly `textwidth_in` wide (tight bbox,
    iterated) for \\includegraphics[width=\\textwidth], Palatino. Each group gets its own legend beneath it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    if palatino:
        from paper_fonts import use_palatino
        use_palatino()
    P, F = PRINT_PT, PRINT_FONT_PT
    s_line = textwidth_in / (4.4 * 3)                              # the print scale of plot_speedup_vs_batch_all's lines (drawn 13.2 in wide)
    lw, ms, mew, raw_s = 1.6 * s_line, 3.5 * s_line, 1.2 * s_line, 5 * s_line ** 2
    baselines = list(BASELINES) + [FRAX_BASELINE] if baselines is None else baselines
    # ---- the speedup panels' data (as plot_speedup_vs_batch_all)
    sp, band_max, band_min, raw_max = {}, 0.0, np.inf, 0.0
    for func, title, kind in FUNCS:
        L = []
        for baseline, blabel, color in baselines:
            per = _speedups(func, kind, baseline=baseline, config=config).get(fp) or {}
            xs, ns, med, lo, hi = _ensemble(per)
            if not xs:
                continue
            raw = {B: np.array([v[B] for v in per.values() if B in v]) for B in xs}
            L.append((blabel, color, xs, med, lo, hi, raw))
            band_max, band_min = max(band_max, max(hi)), min(band_min, min(lo))
            raw_max = max(raw_max, max(float(v.max()) for v in raw.values()))
        sp[func] = L
    lo_lim, hi_lim = min(0.7, band_min / 1.3), band_max * 1.8
    # ---- the DAG panels' data (as plot_dag_shape_speedup)
    shape = _dag_shape()
    sp_by_func = {func: _speedups(func, kind, baseline=baseline_dag, config=config)[fp] for func, _, kind in FUNCS}
    pts = [(B, func, robot, *shape[robot][func], v[B]) for B in batches for func, _, _ in FUNCS
           for robot, v in sp_by_func[func].items() if B in v and func in shape.get(robot, {})]
    if not pts:
        raise SystemExit(f"no DAG points: is {DAG_DIR} populated (menagerie_bench/dag_shape.py)?")
    vals = np.array([q[5] for q in pts])
    norm = LogNorm(vmin=float(np.percentile(vals, 2)), vmax=float(np.percentile(vals, 98)))
    cmap = plt.get_cmap("coolwarm")
    present = {q[1] for q in pts}
    rc = {"font.size": F, "axes.titlesize": F, "axes.labelsize": F, "xtick.labelsize": F, "ytick.labelsize": F, "legend.fontsize": F,
          "axes.linewidth": P["axes_lw"], "xtick.major.width": P["tick_w"], "ytick.major.width": P["tick_w"], "xtick.major.size": P["tick_len"],
          "ytick.major.size": P["tick_len"], "xtick.minor.size": 0.57 * P["tick_len"], "ytick.minor.size": 0.57 * P["tick_len"],
          "xtick.minor.width": 0.75 * P["tick_w"], "ytick.minor.width": 0.75 * P["tick_w"], "xtick.major.pad": P["tick_pad"], "ytick.major.pad": P["tick_pad"],
          "axes.labelpad": P["labelpad"]}

    def draw(fig_w):
        fig = plt.figure(figsize=(fig_w, fig_w * 0.32))
        gs = GridSpec(1, 7, figure=fig, width_ratios=[1, 1, 1, gap, 1, 1, 0.07], wspace=0.14, left=0.05, right=0.985, bottom=0.2, top=0.9)
        axL = [fig.add_subplot(gs[0, 0])]; axL += [fig.add_subplot(gs[0, i], sharey=axL[0]) for i in (1, 2)]
        axR = [fig.add_subplot(gs[0, 4])]; axR += [fig.add_subplot(gs[0, 5], sharey=axR[0])]
        cax = fig.add_subplot(gs[0, 6])
        for ax in axL + axR:
            ax.set_box_aspect(1)
        legend = {}
        for ax, (func, title, kind) in zip(axL, FUNCS):
            for blabel, color, xs, med, lo, hi, raw in sp[func]:
                for B in xs:
                    ax.scatter(np.full(raw[B].size, B), raw[B], s=raw_s, color=color, alpha=0.12, lw=0, zorder=1.5)
                ax.fill_between(xs, lo, hi, color=color, alpha=0.13, lw=0, zorder=2)
                line, = ax.plot(xs, med, "-", marker="o", color=color, ms=ms, lw=lw, zorder=3, markerfacecolor=color, markeredgewidth=mew)
                if ax is axL[0]:
                    legend[blabel] = line
            ax.axhline(1.0, color="gray", lw=0.6 * s_line + 0.3, ls=":", zorder=1)
            ax.set_xscale("log")
            ax.set_xticks(BATCHES, [str(b) if i % 2 == 0 else "" for i, b in enumerate(BATCHES)])      # every other label: 1.2 in panels
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(BATCHES[0] / 1.35, BATCHES[-1] * 1.35)
            ax.set_xlabel("batch size B")
            ax.set_title(title, fontsize=F)
            ax.grid(True, which="major", ls=":", alpha=0.4, lw=P["grid_lw"] + 0.2)
            if ax is not axL[0]:
                ax.tick_params(labelleft=False)
        axL[0].set_ylabel("speedup")
        axL[0].set_yscale("log"); axL[0].set_ylim(lo_lim, hi_lim)
        ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30, 100, 300) if lo_lim <= t <= hi_lim]
        axL[0].set_yticks(ticks, [f"{t:g}" for t in ticks]); axL[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for ax, B in zip(axR, batches):
            here = [q for q in pts if q[0] == B]
            for _, func, robot, w, d, v in sorted(here, key=lambda q: q[5]):     # fast on top
                ax.scatter(w, d, c=[v], cmap=cmap, norm=norm, marker=KIND_MARKER[func], s=P["marker_area"], edgecolor="#444", lw=P["marker_edge_lw"], zorder=3)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
            ax.set_xlabel("mean width"); ax.set_title(f"B = {B}", fontsize=F)
            ax.grid(True, which="both", color="#e5e5e5", lw=P["grid_lw"])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if ax is not axR[0]:
                ax.tick_params(labelleft=False)
        axR[0].set_ylabel("depth")
        yt = [t for t in (2, 5, 10, 20, 50, 100) if axR[0].get_ylim()[0] <= t <= axR[0].get_ylim()[1]]
        axR[0].set_yticks(yt, [str(t) for t in yt]); axR[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("speedup", fontsize=F)
        ct = [t for t in (0.5, 1, 2, 3, 5, 10, 15, 20, 30, 50) if norm.vmin <= t <= norm.vmax]
        cb.set_ticks(ct, labels=[f"{t:g}" for t in ct]); cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        cb.ax.tick_params(labelsize=F, length=0.7 * P["tick_len"], width=P["tick_w"], pad=0.6 * P["tick_pad"]); cb.outline.set_linewidth(P["axes_lw"])
        # the colour bar as tall as the square panels
        fig.canvas.draw()
        b0 = axR[0].get_position(); cax.set_position([cax.get_position().x0, b0.y0, cax.get_position().width, b0.height])
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        def group_anchor(axes):
            bbs = [a.get_tightbbox(rend).transformed(fig.transFigure.inverted()) for a in axes]
            return (min(b.x0 for b in bbs) + max(b.x1 for b in bbs)) / 2, min(b.y0 for b in bbs)
        xc, yb = group_anchor(axL)
        order = [l for _, _, _, l, _, _ in _absolute_series((fp,))]                  # GPU baselines first, then the CPU ones (the paper's order)
        names = sorted(legend, key=lambda l: order.index(l) if l in order else len(order))
        fig.legend([legend[l] for l in names], names, frameon=False, loc="upper center", ncol=3, columnspacing=1.0, handlelength=1.8,
                   handletextpad=0.4, labelspacing=0.3, borderaxespad=0.0, bbox_to_anchor=(xc, yb - 0.015))
        xc, yb2 = group_anchor(axR + [cax])
        fig.legend(handles=[Line2D([], [], ls="", marker=KIND_MARKER[f], color="#888", markeredgecolor="#444", markersize=P["legend_marker"], label=t)
                            for f, t, _ in FUNCS if f in present],
                   frameon=False, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.0, handletextpad=0.2, labelspacing=0.3, borderaxespad=0.0, bbox_to_anchor=(xc, min(yb, yb2) - 0.015))
        # the separator: a thin grey rule in the gap, midway between the left group's last tick labels and the right group's
        # y label, from the bottom of the legends to the top of the titles
        fig.canvas.draw()
        bl = axL[2].get_tightbbox(rend).transformed(fig.transFigure.inverted()); br = axR[0].get_tightbbox(rend).transformed(fig.transFigure.inverted())
        tops = [a.get_tightbbox(rend).transformed(fig.transFigure.inverted()).y1 for a in axL + axR]
        bottoms = [lg.get_window_extent(rend).transformed(fig.transFigure.inverted()).y0 for lg in fig.legends]
        fig.add_artist(Line2D([(bl.x1 + br.x0) / 2] * 2, [min(bottoms), max(tops)], transform=fig.transFigure, color=SEP_GREY, lw=0.6, clip_on=False))
        # (A) / (B) at the top-left corner of each group, level with the panel titles
        for tag, axes in (("(A)", axL), ("(B)", axR)):
            bb = [a.get_tightbbox(rend).transformed(fig.transFigure.inverted()) for a in axes]
            fig.text(min(b.x0 for b in bb), max(tops), tag, ha="left", va="top", fontsize=F, fontweight="bold")
        return fig

    out = os.path.join(RESULTS, f"{stem}_{fp}")
    with matplotlib.rc_context(rc):
        fig_w = textwidth_in
        for it in range(5):
            fig = draw(fig_w); fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.01); plt.close(fig)
            w = pdf_width_in(f"{out}.pdf")
            print(f"  pass {it}: figsize {fig_w:.3f} in -> saved {w:.3f} in (target {textwidth_in})")
            if abs(w - textwidth_in) < 0.005:
                break
            fig_w += textwidth_in - w
        fig = draw(fig_w)
        fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.01)
        fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
    print(f"wrote {out}.png/.pdf  ({pdf_width_in(out + '.pdf'):.3f} x {pdf_width_in(out + '.pdf', height=True):.3f} in; speedup raw max {raw_max:.1f}, "
          f"DAG colour {norm.vmin:.2f}-{norm.vmax:.1f})")


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
