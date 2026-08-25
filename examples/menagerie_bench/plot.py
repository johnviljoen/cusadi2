#!/usr/bin/env python
"""Per-robot plots for the Menagerie latency benchmark — the paper_results
figures (latency, throughput, compile time) adapted to one robot per figure:
a 1x2 grid (forward kinematics | frame Jacobians) instead of the 2x2
two-robot grid, with the robot's name in the titles. Everything else —
curves, overlays, precision split, timing semantics — is identical to
examples/paper_results/plot.py.

Reads results/<robot>/latency_<dtype>.csv (+ latency_py_<dtype>.csv,
mjx_<fpXX>.json, warp.json, compile_cusadi2_<dtype>.json) and writes
<stem>.png + <stem>.pdf next to them.

    python examples/menagerie_bench/plot.py results/unitree_a1 [robot_name]
"""
import csv
import json
import os
import sys
from collections import defaultdict

# Stable order + labels for the four configurations.
CONFIGS = [
    ("full",   "staged + LP + batch (full)"),
    ("staged", "staged + batch (no LP)"),
    ("lp",     "LP + batch, unrolled (no stages)"),
    ("batch",  "batch only, unrolled (~ orig cusadi)"),
]
# (csv func stem, panel title suffix); the robot name is prepended per figure.
FUNCS = [
    ("fk",    "forward kinematics"),
    ("fkjac", "frame Jacobians"),
]
XB = [1, 4, 16, 64, 256, 1024, 4096, 16384]   # batch x-range for flat (batch-independent) lines


def _read(csv_path, col="wall_us"):
    """csv -> {func: {config: ([B...], [us...])}}, sorted by B, NaNs dropped.

    col selects which timing column: 'wall_us' (end-to-end host wall-clock, symmetric
    with the MJX/warp baselines) or 'latency_us' (cudaEvent kernel-only, for throughput).
    Falls back to latency_us if the requested column is absent (older CSVs).
    """
    rows = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                us = float(r.get(col) or r["latency_us"])
            except ValueError:
                continue  # 'nan' skip row (infeasible / did not fit in GPU memory)
            rows[r["func"]][r["config"]].append((int(r["batch"]), us))
    out = {}
    for func, by_cfg in rows.items():
        out[func] = {}
        for cfg, pts in by_cfg.items():
            pts.sort()
            out[func][cfg] = ([b for b, _ in pts], [u for _, u in pts])
    return out


def _read_series(path):
    """results json -> {series: ([B...], [us...])} or {} if absent/unavailable."""
    if not os.path.exists(path):
        return {}
    res = json.load(open(path)).get("results")   # absent when mjx/warp_unavailable
    if not res:
        return {}
    out = {}
    for key, by_b in res.items():
        pts = sorted((int(b), v * 1e6) for b, v in by_b.items() if v is not None)
        if pts:
            out[key] = ([b for b, _ in pts], [u for _, u in pts])
    return out


def _read_mjx(out_dir, fp):
    return _read_series(os.path.join(out_dir, f"mjx_{fp}.json"))


def plot_one(csv_path, out_dir, n_stages, robot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _read(csv_path)
    # Python-called twin sweep (run.py --pylib), if it has been produced.
    py_csv = os.path.join(os.path.dirname(csv_path),
                          os.path.basename(csv_path).replace("latency_", "latency_py_"))
    py_data = _read(py_csv) if os.path.exists(py_csv) else {}
    dtype = "fp64" if "double" in os.path.basename(csv_path) else "fp32"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, (func, title) in zip(axes.flat, FUNCS):
        series = data.get(func, {})
        for cfg, label in CONFIGS:
            if cfg in series:
                bs, us = series[cfg]
                (ln,) = ax.plot(bs, us, marker="o", ms=4, label=label)
                p = py_data.get(func, {}).get(cfg)
                if p:   # same config through Python + ctypes: same color, dashed, x
                    ax.plot(p[0], p[1], color=ln.get_color(), ls="--", marker="x",
                            ms=5, lw=1, alpha=0.8)
        # MJX baseline overlay (precision-matched). Two curves, both doing B*N
        # total evals like cusadi2: batched (parallel horizon) and rollout
        # (N sequential mjx.step()s). FK panels use the fk_* series, Jacobian
        # panels the jac_* series.
        kind = "jac" if "jac" in func else "fk"
        mjx = _read_mjx(out_dir, dtype)
        for suffix, style, mk, lab in [
            ("batched", "--", "s", f"MJX batched, B·N ({dtype})"),
            ("rollout", ":",  "^", f"MJX rollout, N steps ({dtype})"),
        ]:
            s = mjx.get(f"{kind}_{suffix}")
            if s:
                ax.plot(s[0], s[1], marker=mk, ms=4, color="black", ls=style, label=lab)
        # mujoco-warp (fp32, no autodiff): FK panels only, and only on the fp32
        # figure — it can't be compared to fp64 cusadi2.
        if kind == "fk" and dtype == "fp32":
            w = _read_series(os.path.join(out_dir, "warp.json")).get("fk")
            if w:
                ax.plot(w[0], w[1], marker="D", ms=4, color="#9467bd", ls="-.",
                        label="mujoco-warp (fp32)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{robot} {title}")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.set_xlabel("batch size B")
        ax.set_ylabel("wall-clock latency (µs)")
    if py_data:
        axes.flat[0].plot([], [], color="gray", ls="--", marker="x", ms=5, lw=1,
                          label="dashed ×: same, called from Python (ctypes)")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{robot} — single-call wall-clock latency vs batch  "
                 f"({dtype}, horizon N={n_stages}, total work = B·N)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(csv_path))[0])
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")
    return f"{stem}.png"


def plot_all(csv_paths, out_dir, n_stages, robot):
    return [plot_one(p, out_dir, n_stages, robot) for p in csv_paths]


# ---------------------------------------------------------------------------
# Throughput plots: per-eval cost (ns) vs batch, ALL launch/dispatch overhead
# removed on both sides. cusadi2 uses its cudaEvent kernel-only time; MJX uses the
# repeat-subtraction marginal GPU time (*_batched_gpu). Divisor is B·N (both do B·N
# evals). This is the like-for-like GPU throughput comparison — where the ~10x
# wall-clock gap turns out to be launch overhead, not compute.
# ---------------------------------------------------------------------------
def _read_gpu(out_dir, fp, n_stages):
    """MJX marginal GPU: {kind: ([B...], [ns_per_eval...])} from *_batched_gpu (seconds)."""
    path = os.path.join(out_dir, f"mjx_{fp}.json")
    if not os.path.exists(path):
        return {}
    res = json.load(open(path)).get("results")
    if not res:
        return {}
    out = {}
    for kind in ("fk", "jac"):
        by_b = res.get(f"{kind}_batched_gpu")
        if not by_b:
            continue
        pts = sorted((int(b), v * 1e9 / (int(b) * n_stages))
                     for b, v in by_b.items() if v is not None)
        if pts:
            out[kind] = ([b for b, _ in pts], [ns for _, ns in pts])
    return out


def plot_throughput_one(csv_path, out_dir, n_stages, robot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _read(csv_path, col="latency_us")     # kernel-only, overhead-free
    dtype = "fp64" if "double" in os.path.basename(csv_path) else "fp32"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, (func, title) in zip(axes.flat, FUNCS):
        series = data.get(func, {})
        for cfg, label in CONFIGS:
            if cfg in series:
                bs, us = series[cfg]
                ns = [u * 1e3 / (b * n_stages) for b, u in zip(bs, us)]   # ns per eval
                ax.plot(bs, ns, marker="o", ms=4, label=label)
        kind = "jac" if "jac" in func else "fk"
        g = _read_gpu(out_dir, dtype, n_stages).get(kind)
        if g:
            ax.plot(g[0], g[1], marker="s", ms=4, color="black", ls="--",
                    label=f"MJX batched, GPU-only ({dtype})")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{robot} {title}")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.set_xlabel("batch size B")
        ax.set_ylabel("GPU time per eval (ns)")
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{robot} — per-eval GPU throughput vs batch  (launch overhead "
                 f"removed; {dtype}, N={n_stages}, {n_stages}·B evals)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = os.path.join(out_dir, "throughput_" +
                        ("double" if dtype == "fp64" else "float"))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")
    return f"{stem}.png"


def plot_throughput_all(csv_paths, out_dir, n_stages, robot):
    return [plot_throughput_one(p, out_dir, n_stages, robot) for p in csv_paths]


# ---------------------------------------------------------------------------
# Compile / codegen time plots. cusadi2 and mujoco-warp compile once (batch- and,
# for cusadi2, function-independent kernel), so they're flat lines; MJX re-JITs per
# batch shape, so it rises. y = seconds.
# ---------------------------------------------------------------------------
def _read_mjx_compile(out_dir, fp):
    path = os.path.join(out_dir, f"mjx_{fp}.json")
    if not os.path.exists(path):
        return {}
    comp = json.load(open(path)).get("compile", {})
    out = {}
    for key, by_b in comp.items():
        pts = sorted((int(b), v) for b, v in by_b.items() if v is not None)  # seconds
        if pts:
            out[key] = ([b for b, _ in pts], [s for _, s in pts])
    return out


def plot_compile_one(dtype_name, out_dir, n_stages, robot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp = "fp64" if dtype_name == "double" else "fp32"
    cpath = os.path.join(out_dir, f"compile_cusadi2_{dtype_name}.json")
    cc = json.load(open(cpath)) if os.path.exists(cpath) else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, (func, title) in zip(axes.flat, FUNCS):
        kind = "jac" if "jac" in func else "fk"
        # cusadi2: per-function codegen + one-time (shared) nvcc kernel compile —
        # batch-independent, so a flat line.
        if cc:
            y = cc["codegen_s"].get(func, 0.0) + cc["nvcc_s"]
            ax.plot([XB[0], XB[-1]], [y, y], color="C0", lw=2,
                    label="cusadi2 (codegen + nvcc, once)")
        # MJX: first-call JIT compile, re-traced per batch shape -> rises.
        comp = _read_mjx_compile(out_dir, fp)
        for suffix, style, mk, lab in [("batched", "--", "s", "MJX batched"),
                                       ("rollout", ":", "^", "MJX rollout")]:
            s = comp.get(f"{kind}_{suffix}")
            if s:
                ax.plot(s[0], s[1], color="black", ls=style, marker=mk, ms=4,
                        label=f"{lab} JIT ({fp})")
        # mujoco-warp: compiles once (shape-generic kernels) -> flat; FK + fp32 only.
        if kind == "fk" and fp == "fp32":
            wpath = os.path.join(out_dir, "warp.json")
            if os.path.exists(wpath):
                w = json.load(open(wpath)).get("compile_once_s")
                if w:
                    ax.plot([XB[0], XB[-1]], [w, w], color="#9467bd", ls="-.", lw=2,
                            label="mujoco-warp (once, fp32)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{robot} {title}")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.set_xlabel("batch size B")
        ax.set_ylabel("compile time (s)")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{robot} — compile / codegen time vs batch  ({fp}, horizon "
                 f"N={n_stages})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = os.path.join(out_dir, f"compile_{dtype_name}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")
    return f"{stem}.png"


def plot_compile_all(out_dir, dtypes, n_stages, robot):
    return [plot_compile_one(d, out_dir, n_stages, robot) for d in dtypes]


def plot_robot(robot, res_dir, n_stages, dtypes=("double", "float")):
    """All figures for one robot's results directory (whatever CSVs exist)."""
    csvs = [os.path.join(res_dir, f"latency_{d}.csv") for d in dtypes]
    csvs = [p for p in csvs if os.path.exists(p)]
    plot_all(csvs, res_dir, n_stages, robot)
    plot_throughput_all(csvs, res_dir, n_stages, robot)
    plot_compile_all(res_dir, [d for d in dtypes if os.path.exists(
        os.path.join(res_dir, f"compile_cusadi2_{d}.json"))], n_stages, robot)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: plot.py <results/robot_dir> [robot_name]")
    res_dir = sys.argv[1]
    robot = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.normpath(res_dir))
    plot_robot(robot, res_dir, n_stages=8)
