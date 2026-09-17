"""DAG shape of the benchmark functions: depth vs width.

For each of the six functions in bench.FUNCS (A1/H1 forward kinematics, frame
Jacobians, frame Hessians) and the five original-cusadi benchmark functions
fn_1e1 .. fn_1e5 (bench_orig.functions, from orig_functions/) the plain
longest-path level plan (no windowing) is extracted and the number of compute
nodes per level is recorded. INPUT/OUTPUT nodes are excluded, so depth = number
of compute levels. The same three functions of every MuJoCo Menagerie robot are
read from the cache menagerie_bench/dag_shape.py writes
(menagerie_bench/results/dag/<robot>.json).

Writes results/dag_shape.{png,pdf} (left: histogram of level widths over the
menagerie, one outline per function kind; right: one point per function, mean
width vs depth, the A1/H1 and original functions labelled, the menagerie as
unlabelled dots) and results/dag_shape.csv (every function, `family` column:
paper / orig / menagerie) with the windowing facts the menagerie cache records.

    python dag_shape.py            # cusadi2 conda env
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench import FUNCS, build_function                       # noqa: E402
from bench_orig import FNS as ORIG_FNS, functions as orig_functions   # noqa: E402
from cusadi2.dag import IO_OPS, extract_level_plan             # noqa: E402

RESULTS = os.path.join(HERE, "results")
MENAGERIE_DAG = os.path.join(HERE, "..", "menagerie_bench", "results", "dag")
PLANS = os.path.join(HERE, "build", "double", "plans")        # as benchmarked (windowed), if present

COLOR = {"fk": "#2a78d6", "fkjac": "#eb6834", "fkhess": "#1baf7a"}   # hue = quantity
MARKER = {"a1": "o", "h1": "s", "orig": "^", "menagerie": "."}
KIND_LABEL = {"fk": "forward kinematics", "fkjac": "frame Jacobians", "fkhess": "frame Hessians"}
ORIG_COLOR = ["#a3a3a3", "#7f7f7f", "#5c5c5c", "#3a3a3a", "#1a1a1a"]  # ordinal gray, 1e1 -> 1e5


def compute_widths(plan):
    """Compute-node count per level, dropping levels that hold only IO nodes."""
    ops = np.asarray(plan.node_ops)
    is_compute = ~np.isin(ops, list(IO_OPS))
    widths = [int(is_compute[o:o + w].sum()) for o, w in zip(plan.level_offsets, plan.level_widths)]
    return [w for w in widths if w > 0]


def load_menagerie():
    """Rows from menagerie_bench's cache -> (rows, windowed notes, failed notes)."""
    import glob
    import json
    rows, windowed, failed = [], [], []
    for path in sorted(glob.glob(os.path.join(MENAGERIE_DAG, "*.json"))):
        rec = json.load(open(path))
        if rec.get("status") == "failed":
            failed.append(f"{rec['robot']} during {rec['failed'].get('during')} rc={rec['failed'].get('returncode')}")
        for stem, r in rec.get("functions", {}).items():
            fit = r.get("fit", {})
            row = dict(family="menagerie", stem=f"{rec['robot']}_{stem}", robot=rec["robot"], kind=stem,
                       nq=rec["nq"], nv=rec["nv"], n_bodies=rec["n_bodies"], n_instr=r["n_instr"],
                       n_compute=r["n_compute"], depth=r["depth"], mean_width=r["mean_width"],
                       median_width=r["median_width"], max_width=r["max_width"],
                       peak_live_slots=r["peak_live_slots"], widths=r["widths"])
            for d in ("double", "float"):
                f = fit.get(d, {})
                row[f"windowed_{d}"] = "error" if "error" in f else f.get("windowed")
                if "error" in f:
                    windowed.append(f"{rec['robot']}/{stem} {d}: {f['error']}")
                elif f.get("windowed"):
                    windowed.append(f"{rec['robot']}/{stem} {d}: {r['n_levels']} -> {f['n_levels']} levels")
            rows.append(row)
    return rows, windowed, failed


def main():
    rows = []
    cases = [(stem, label, lambda _s=stem: build_function(_s), *stem.split("_")) for stem, label in FUNCS]
    cases += [(f.name(), f"original cusadi {f.name()[3:]}", lambda _f=f: _f, "orig", f.name()[3:])
              for f in orig_functions()]
    for stem, label, build, robot, kind in cases:
        t0 = time.perf_counter()
        fn = build()
        plan = extract_level_plan(fn)                    # plain longest-path levels, no window
        widths = compute_widths(plan)
        r = dict(stem=stem, label=label, robot=robot, kind=kind, n_instr=fn.n_instructions(),
                 n_compute=int(sum(widths)), depth=len(widths), mean_width=sum(widths) / len(widths),
                 median_width=float(np.median(widths)), max_width=max(widths), widths=widths)
        p = os.path.join(PLANS, f"{stem}.plan")
        if os.path.exists(p):
            from cusadi2 import read_plan
            d = read_plan(p)
            r["bench_levels"] = len(d["level_widths"])
        rows.append(r)
        print(f"{stem:10s} nodes={r['n_compute']:8d} depth={r['depth']:5d} mean={r['mean_width']:8.1f} "
              f"median={r['median_width']:7.0f} max={r['max_width']:6d}"
              + (f"  (benchmark plan: {r['bench_levels']} levels incl. IO)"
                 if "bench_levels" in r else "") + f"  [{time.perf_counter() - t0:.1f}s]")

    for r in rows:
        r["family"] = "orig" if r["robot"] == "orig" else "paper"
    menagerie, windowed, failed = load_menagerie()
    print(f"menagerie: {len(menagerie)} functions; windowed by cusadi2 at the device limit: "
          f"{len(windowed)}; failed robots: {len(failed)}")
    for w in windowed:
        print("  windowed:", w)
    for f in failed:
        print("  FAILED:", f)

    os.makedirs(RESULTS, exist_ok=True)
    keys = ["family", "stem", "robot", "kind", "nq", "nv", "n_bodies", "n_instr", "n_compute", "depth",
            "mean_width", "median_width", "max_width", "peak_live_slots", "windowed_double", "windowed_float",
            "bench_levels"]
    with open(os.path.join(RESULTS, "dag_shape.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows + menagerie)
    with open(os.path.join(RESULTS, "dag_shape_widths.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "level", "width"])
        for r in rows:
            w.writerows((r["stem"], i, x) for i, x in enumerate(r["widths"]))

    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw=dict(width_ratios=[1, 1]))

    def color(r):
        return ORIG_COLOR[ORIG_FNS.index(r["stem"])] if r["robot"] == "orig" else COLOR[r["kind"]]

    def short(r):
        return r["kind"] if r["robot"] == "orig" else f"{r['robot'].upper()} {r['kind']}"

    # left: histogram of level widths over every menagerie function, per kind
    all_w = [x for r in menagerie for x in r["widths"]] or [1]
    bins = np.logspace(0, np.ceil(np.log10(max(all_w))), 40)
    for kind in ("fk", "fkjac", "fkhess"):
        ws = [x for r in menagerie if r["kind"] == kind for x in r["widths"]]
        n = sum(r["kind"] == kind for r in menagerie)
        if ws:
            ax.hist(ws, bins=bins, histtype="step", color=COLOR[kind], lw=1.5,
                    label=f"{KIND_LABEL[kind]} ({n} robots, {len(ws)} levels)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("width (compute nodes in the level)")
    ax.set_ylabel("number of levels")
    ax.set_title("Level widths over the MuJoCo Menagerie")
    ax.grid(True, which="both", color="#e5e5e5", lw=0.5)
    ax.set_ylim(top=ax.get_ylim()[1] * 4)                 # headroom for the legend
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # right: mean width vs depth, one point per function
    for r in menagerie:
        ax2.scatter(r["mean_width"], r["depth"], color=COLOR[r["kind"]], s=14, marker="o", lw=0,
                    alpha=0.55, zorder=2)
    for r in rows:
        ax2.scatter(r["mean_width"], r["depth"], color=color(r), s=60,
                    marker=MARKER[r["robot"]], edgecolor="white", lw=1, zorder=3)
        ax2.annotate(short(r), (r["mean_width"], r["depth"]), textcoords="offset points",
                     xytext=(6, -9 if r["robot"] == "a1" else 4), fontsize=8, color="#444")
    for kind in ("fk", "fkjac", "fkhess"):
        ax2.scatter([], [], color=COLOR[kind], s=14, alpha=0.55, label=f"menagerie {KIND_LABEL[kind]}")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    ax2.set_ylabel("depth (number of compute levels)")
    ax2.set_xlabel("mean width (compute nodes / level)")
    ax2.set_title("Mean width vs depth (circle A1, square H1, triangle original, dot menagerie)")
    ax2.grid(True, which="both", color="#e5e5e5", lw=0.5)
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS, f"dag_shape.{ext}"), dpi=140, bbox_inches="tight")
    print("wrote", os.path.join(RESULTS, "dag_shape.png"))


if __name__ == "__main__":
    main()
