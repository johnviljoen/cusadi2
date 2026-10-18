# Adapted 2026-10-17 from ~/.cache/cusadi2_fp32_check/plot_f32_speedup5.py (the script that made panda_safe_mppi/results/speedup_f32_paper.pdf)
# to this example: results in this directory, the sweep is K at T = 35 with 3 iterations per tick (bench.py defaults), the CPU-idealized rows
# come from bench_cpu.py (evaluator "cpu_ideal").
"""Single-column paper figure from results/bench_f32_runs.csv (5 appended runs of bench.py): speedup of CusADi2 over each
backend vs batch size B = K at T = 35 (3 planner iterations per control step), fp32. Line = ratio of the 5-run mean control-step medians; band = +- 1 std of the
per-run ratios (run i of the baseline over run i of CusADi2). Dotted line at 1 = CusADi2, unlabelled.

Sizes are the printed sizes of the paper's DAG-shape figure in its original two-column form (menagerie_bench/paper_plots.py
PRINT_PT: matplotlib's defaults drawn 11.75 in wide and printed at textwidth, i.e. x 0.609); this figure is saved at exactly
`columnwidth` (tight bbox, iterated) with those sizes, so \\includegraphics[width=\\columnwidth] scales it by 1 and it reads the
same as the (now also one-column) DAG-shape figure. Fonts: Palatino (TeX Gyre Pagella, paper_fonts.use_palatino).

    python plot_speedup_paper.py [--T 35] [--textwidth 7.16] [--columnwidth 3.5]     (inches; defaults = IEEEtran two-column)
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--textwidth", type=float, default=7.16); ap.add_argument("--columnwidth", type=float, default=3.5)
ap.add_argument("--out", default=None); ap.add_argument("--T", type=int, default=35, help="the horizon of the K sweep (bench.py --KT)")
a = ap.parse_args()
R = Path(__file__).resolve().parent / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "menagerie_bench"))
from paper_fonts import use_palatino  # noqa: E402
from paper_plots import PRINT_SCALE, REF_WIDTH_IN, PRINT_FONT_PT  # noqa: E402
# the reference figure's sizes as drawn (paper_plots.plot_dag_shape_speedup in its two-column form; matplotlib defaults where it set nothing)
REF_PT = dict(label=10.0, tick=10.0, title=10.0, legend=8.0, cbar=9.0, grid_lw=0.5, marker_edge_lw=0.35, marker_d=np.sqrt(26.0), legend_marker=6.0)
REF_LINE = dict(lw=1.8, ms=3.5)        # the paper's line figures (paper_speedup_vs_batch), 13.2 in wide, for the line/marker weight

import re, subprocess
def pdf_width_in(path):
    """page width of a PDF in inches (pdfinfo 'Page size: W x H pts')."""
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, check=True).stdout)
    return float(m.group(1)) / 72.0
ref_w_in = REF_WIDTH_IN                                          # the two-column DAG-shape figure was 11.75 in wide
s_ref = a.textwidth / ref_w_in                                   # its print scale (= PRINT_SCALE at the default text width)
s_line = a.textwidth / (4.4 * 3)                                 # print scale of the paper's line figures
P = {k: v * s_ref for k, v in REF_PT.items()}                    # printed sizes (pt) to reproduce: lines and markers ...
P.update(label=PRINT_FONT_PT, tick=PRINT_FONT_PT, title=PRINT_FONT_PT, legend=PRINT_FONT_PT, cbar=PRINT_FONT_PT)   # ... and all text at the footnote size
lw_print, ms_print = REF_LINE["lw"] * s_line, REF_LINE["ms"] * s_line
print(f"reference width {ref_w_in:.2f} in -> print scale {s_ref:.3f}: labels {P['label']:.2f} pt, ticks {P['tick']:.2f} pt, legend {P['legend']:.2f} pt, "
      f"marker {P['marker_d']:.2f} pt; line figures scale {s_line:.3f}: lw {lw_print:.2f} pt, ms {ms_print:.2f} pt")

rows = list(csv.DictReader(open(R / "bench_f32_runs.csv")))
if (R / "bench_f32_cpu_ideal_runs.csv").exists():                 # the CPU-only driver (bench_cpu.py), evaluator "cpu_ideal"
    rows += list(csv.DictReader(open(R / "bench_f32_cpu_ideal_runs.csv")))
runs, p99s, models = defaultdict(list), defaultdict(list), defaultdict(list)
for r in rows:
    k = (r["evaluator"], int(r["K"]), int(r["T"]))
    runs[k].append(float(r["tick_median_ms"])); p99s[k].append(float(r["tick_p99_ms"])); models[k].append(float(r["model_ms"]))
T0 = a.T
Ks = sorted({K for (ev, K, T) in runs if ev == "cusadi2" and T == T0})
# colours and markers per baseline as in menagerie_bench/paper_plots.py (BASELINES, FRAX_BASELINE, _absolute_series)
EVS = [("mjx", "MuJoCo XLA", "#1f77b4", "^"), ("cusadi", "CusADi", "#2ca02c", "s"), ("frax", "frax", "#e377c2", "*"),
       ("cpu_ideal", "CPU idealized", "#d4a017", "v"), ("cpu", "CPU + transfer", "#8c564b", "P")]
NAMES = dict(cusadi2="CusADi2 (ours)", **{e: n for e, n, _, _ in EVS})
series, L = {}, []
for ev, name, c, m in EVS:
    xs, mean, lo, hi, txt = [], [], [], [], []
    for K in Ks:
        A, B_ = np.array(runs.get((ev, K, T0), [])), np.array(runs[("cusadi2", K, T0)])
        if len(A) == 0: continue
        n = min(len(A), len(B_)); ratios = A[:n] / B_[:n]
        s = A.mean() / B_.mean(); sd = ratios.std(ddof=1) if n > 1 else 0.0
        xs.append(K); mean.append(s); lo.append(s - sd); hi.append(s + sd); txt.append(f"B={K}: {s:.2f} +- {sd:.2f}")
    series[ev] = (xs, mean, lo, hi); L.append(f"- {name}: " + ", ".join(txt))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
use_palatino()
plt.rcParams.update({"pdf.fonttype": 42, "font.size": P["label"], "axes.labelsize": P["label"], "xtick.labelsize": P["tick"], "ytick.labelsize": P["tick"],
                     "legend.fontsize": P["legend"], "axes.linewidth": 0.8 * s_ref, "xtick.major.width": 0.8 * s_ref, "ytick.major.width": 0.8 * s_ref,
                     "xtick.major.size": 3.5 * s_ref, "ytick.major.size": 3.5 * s_ref, "xtick.major.pad": 3.5 * s_ref, "ytick.major.pad": 3.5 * s_ref,
                     "axes.labelpad": 4.0 * s_ref})
# absolute control-step latency per backend (cusadi2 included): mean +- std over runs of the per-run tick medians
CUS = ("cusadi2", "CusADi2", "C1", "o")                           # paper_plots SERIES: fp32 cusadi2 is "C1", marker "o"
absolute = {}
for ev, name, c, m in [CUS] + EVS:
    xs, mean, sd = [], [], []
    for K in Ks:
        A = np.array(runs.get((ev, K, T0), []))
        if len(A) == 0: continue
        xs.append(K); mean.append(A.mean()); sd.append(A.std(ddof=1) if len(A) > 1 else 0.0)
    absolute[ev] = (xs, np.array(mean), np.array(sd))
    L.append(f"- {name} tick (ms): " + ", ".join(f"B={K}: {mu:.2f} +- {s:.2f}" for K, mu, s in zip(xs, mean, sd)))

def log_axis(ax, ticks, lim):
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(Ks); ax.set_xticklabels([str(K) if i % 2 == 0 else "" for i, K in enumerate(Ks)]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())   # every other label: at the footnote size "1024 2048" collide
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.set_ylim(*lim)
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))   # the integer minor ticks of a log axis, as in the reference
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="y", which="minor", length=2.0 * s_ref, width=0.6 * s_ref)
    ax.set_xlabel("batch size B")
    ax.grid(True, which="both", color="#e5e5e5", lw=P["grid_lw"])

# axis limits from the data (the MPPI figure's fixed limits were 0.18-30 and 2.5-600 ms), a 10 % margin on the log scale
_sp = np.concatenate([np.array(series[ev][2] + series[ev][3]) for ev in series if series[ev][0]])
_lat = np.concatenate([absolute[ev][1] - absolute[ev][2] for ev in absolute if len(absolute[ev][0])] + [absolute[ev][1] + absolute[ev][2] for ev in absolute if len(absolute[ev][0])])
sp_lim = (min(_sp.min(), 0.9) / 1.2, max(_sp.max(), 1.1) * 1.4); lat_lim = (_lat.min() / 1.2, _lat.max() * 1.4)

def draw(fig_w):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_w * 0.46))
    for ev, name, c, m in EVS:                                      # left: speedup of cusadi2 over each backend
        xs, mean, lo, hi = series[ev]
        ax.plot(xs, mean, marker=m, ms=ms_print, lw=lw_print, color=c, label=name)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.25, lw=0)
    ax.axhline(1.0, color="k", ls=":", lw=1.0 * s_ref)
    log_axis(ax, [t for t in [0.2, 0.5, 1, 2, 5, 10, 20, 50] if sp_lim[0] <= t <= sp_lim[1]], sp_lim); ax.set_ylabel("speedup")
    for ev, name, c, m in [CUS] + EVS:                              # right: absolute control-step latency, cusadi2 included
        xs, mean, sd = absolute[ev]
        ax2.plot(xs, mean, marker=m, ms=ms_print, lw=lw_print, color=c, label=name)
        ax2.fill_between(xs, mean - sd, mean + sd, color=c, alpha=0.25, lw=0)
    log_axis(ax2, [t for t in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] if lat_lim[0] <= t <= lat_lim[1]], lat_lim); ax2.set_ylabel("latency (ms)")
    fig.tight_layout(w_pad=0.8)
    h, l = ax2.get_legend_handles_labels()                          # beneath the panels, as the paper's other figures (paper_plots: loc upper center anchored below the axes)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for tag, a_ in (("(A)", ax), ("(B)", ax2)):                     # panel labels inside the top-left corner (no title row; above the frame they cost height)
        a_.text(0.03, 0.96, tag, transform=a_.transAxes, ha="left", va="top", fontsize=P["title"], fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
    top = max(a.get_position().y1 for a in (ax, ax2))
    # the legend ABOVE the panels, tight against the frames (no title row), two rows of three: six entries at the footnote
    # size do not fit one 3.5 in row, and the figure's bbox must not be set by the legend
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, columnspacing=1.0, handlelength=1.4, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.0,
               bbox_to_anchor=(0.5, top + 0.012))
    return fig
out = Path(a.out) if a.out else R / "speedup_f32_paper"
fig_w = a.columnwidth
for it in range(4):                                     # tight bbox is narrower than figsize: iterate until the saved width == columnwidth
    fig = draw(fig_w); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01); plt.close(fig)
    w = pdf_width_in(out.with_suffix(".pdf"))
    print(f"  pass {it}: figsize {fig_w:.3f} in -> saved {w:.3f} in (target {a.columnwidth})")
    if abs(w - a.columnwidth) < 0.005: break
    if w - fig_w > 0.5: raise SystemExit(f"the tight bbox ({w:.2f} in) is far wider than the figure ({fig_w:.2f} in): something outside the axes (legend?) sets the width")
    fig_w += a.columnwidth - w
fig = draw(fig_w); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01); fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.01); plt.close(fig)
(R / "speedup_f32_paper_sizes.txt").write_text(f"reference: the two-column DAG-shape figure, {ref_w_in:.3f} in wide, textwidth {a.textwidth} in -> scale {s_ref:.4f}; font {use_palatino()}\n"
    f"printed sizes (pt): " + ", ".join(f"{k} {v:.2f}" for k, v in P.items()) + f", line lw {lw_print:.2f}, marker ms {ms_print:.2f}\n"
    f"this figure: saved width {w:.3f} in = columnwidth {a.columnwidth} in (scale 1)\n" + "\n".join(L) + "\n")
print("\n".join(L)); print(out)
