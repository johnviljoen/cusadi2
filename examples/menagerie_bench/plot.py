#!/usr/bin/env python
"""Per-robot plots for the Menagerie latency benchmark — the paper_results
figures (latency, throughput, compile time) adapted to one robot per figure:
a 1x2 grid (forward kinematics | frame Jacobians) instead of the 2x2
two-robot grid, with the robot's name in the titles. Everything else —
curves, overlays, precision split, timing semantics — is identical to
examples/paper_results/plot.py.

Reads results/<robot>/runs/<k>/latency_<dtype>.csv (+ latency_py_<dtype>.csv,
cpu_<dtype>.json — the paper's CPU reference, idealized and with the
GPU->CPU->GPU transfer, drawn for CPU_PLOT_THREADS), mjx_<fpXX>.json and
warp.json (per run: mean +- std like the cusadi2 curves; the MJX compile
times are per shape, paid once), compile_cusadi2_<dtype>.json, and writes
<stem>.png + <stem>.pdf next to them.

    python examples/menagerie_bench/plot.py results/unitree_a1 [robot_name]
"""
import csv
import glob
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
    ("fk",     "forward kinematics"),
    ("fkjac",  "frame Jacobians"),
    ("fkhess", "frame Hessians"),
]
KIND = {"fk": "fk", "fkjac": "jac", "fkhess": "hess"}   # csv stem -> MJX series prefix (no hess in MJX)
XB = [1, 4, 16, 64, 256, 1024, 4096, 16384]   # batch x-range for flat (batch-independent) lines
# CPU reference thread counts drawn (paper_results measures 1/16/32/64 and draws 16;
# here the sequential and the parallel reference both go on the figure).
CPU_PLOT_THREADS = [1, 16]


def run_csvs(res_dir, name):
    """Every run's copy of results/<robot>/runs/<k>/<name>, in run order."""
    runs = sorted(glob.glob(os.path.join(res_dir, "runs", "*", name)),
                  key=lambda q: int(os.path.basename(os.path.dirname(q))))
    return runs


def _read(res_dir, name, col="wall_us"):
    """All runs of one CSV -> {func: {config: ([B...], [mean us...], [std us...])}},
    sorted by B, NaN rows dropped.

    col selects which timing column: 'wall_us' (end-to-end host wall-clock, symmetric
    with the MJX/warp baselines) or 'latency_us' (cudaEvent kernel-only, for throughput).
    """
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in run_csvs(res_dir, name):
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    us = float(r[col])
                except ValueError:
                    continue  # 'nan' skip row (infeasible / did not fit in GPU memory)
                rows[r["func"]][r["config"]][int(r["batch"])].append(us)
    out = {}
    for func, by_cfg in rows.items():
        out[func] = {}
        for cfg, by_b in by_cfg.items():
            bs = sorted(by_b)
            means = [sum(by_b[b]) / len(by_b[b]) for b in bs]
            stds = [(sum((x - m) ** 2 for x in by_b[b]) / len(by_b[b])) ** 0.5 for b, m in zip(bs, means)]
            out[func][cfg] = (bs, means, stds)
    return out


def _errline(ax, s, **kw):
    """Mean line with +-1 std caps; returns the line (for its color)."""
    return ax.errorbar(s[0], s[1], yerr=s[2], capsize=3, elinewidth=1, **kw).lines[0]


def _read_runs(out_dir, name, section="results"):
    """Every run's <name>.json (runs/<k>/<name>.json; the single legacy file at the
    top level is accepted) -> {series: ([B...], [mean us...], [std us...])}, nulls
    dropped, {} if absent or unavailable. section="compile" reads the JIT times (s)."""
    paths = run_csvs(out_dir, name) or ([os.path.join(out_dir, name)] if os.path.exists(os.path.join(out_dir, name)) else [])
    scale = 1e6 if section == "results" else 1.0
    acc = defaultdict(lambda: defaultdict(list))
    for path in paths:
        try:
            sec = json.load(open(path)).get(section)      # absent when mjx/warp_unavailable
        except Exception:
            sec = None
        for key, by_b in (sec or {}).items():
            for b, v in by_b.items():
                if v is not None:
                    acc[key][int(b)].append(v * scale)
    out = {}
    for key, by_b in acc.items():
        bs = sorted(by_b)
        means = [sum(by_b[b]) / len(by_b[b]) for b in bs]
        stds = [(sum((x - m) ** 2 for x in by_b[b]) / len(by_b[b])) ** 0.5 for b, m in zip(bs, means)]
        out[key] = (bs, means, stds)
    return out


def _read_mjx(out_dir, fp):
    return _read_runs(out_dir, f"mjx_{fp}.json")


def _read_warp_compile(out_dir, key="compile_once_s"):
    for path in run_csvs(out_dir, "warp.json") + [os.path.join(out_dir, "warp.json")]:
        try:
            w = json.load(open(path)).get(key)
        except Exception:
            w = None
        if w:
            return w
    return None


def _read_cpu(out_dir, dtype_name, key, func, threads, n_stages, per_eval=False):
    """All runs' cpu_<dtype>.json -> ([B], [mean], [std]) of seq_us / seq_transfer_us
    [func][threads] (us; ns per eval when per_eval), nulls dropped; None if absent."""
    by_b = defaultdict(list)
    for path in run_csvs(out_dir, f"cpu_{dtype_name}.json"):
        d = json.load(open(path)).get(key, {}).get(func, {}).get(str(threads), {})
        for b, v in d.items():
            if v is not None:
                by_b[int(b)].append(v * 1e3 / (int(b) * n_stages) if per_eval else v)
    if not by_b:
        return None
    bs = sorted(by_b)
    means = [sum(by_b[b]) / len(by_b[b]) for b in bs]
    stds = [(sum((x - m) ** 2 for x in by_b[b]) / len(by_b[b])) ** 0.5 for b, m in zip(bs, means)]
    return bs, means, stds


def _cpu_lines(ax, out_dir, dtype_name, func, n_stages, per_eval=False):
    """paper_results' _cpu_line: CasADi C codegen on the CPU, idealized (data in
    host memory) and with the GPU->CPU->GPU transfer, per thread count."""
    styles = ["-", "--", "-.", ":"]
    for n, thr in enumerate(CPU_PLOT_THREADS):
        for key, color, label in (("seq_us", "gold", "CPU idealized"),
                                  ("seq_transfer_us", "#8c564b", "CPU transfer (GPU→CPU→GPU)")):
            s = _read_cpu(out_dir, dtype_name, key, func, thr, n_stages, per_eval)
            if s:
                _errline(ax, s, marker="*", ms=7, color=color, markeredgecolor="black", ls=styles[n % 4], lw=1.5,
                         label=f"CasADi C codegen, {label}, B·N over {thr} thread{'s' if thr != 1 else ''}")


def plot_one(dtype_name, out_dir, n_stages, robot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _read(out_dir, f"latency_{dtype_name}.csv")
    py_data = _read(out_dir, f"latency_py_{dtype_name}.csv")   # Python-called twin sweep
    n_runs = len(run_csvs(out_dir, f"latency_{dtype_name}.csv"))
    dtype = "fp64" if dtype_name == "double" else "fp32"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for ax, (func, title) in zip(axes.flat, FUNCS):
        series = data.get(func, {})
        for cfg, label in CONFIGS:
            if cfg in series:
                ln = _errline(ax, series[cfg], marker="o", ms=4, label=label)
                p = py_data.get(func, {}).get(cfg)
                if p:   # same config through Python + ctypes: same color, dashed, x
                    _errline(ax, p, color=ln.get_color(), ls="--", marker="x", ms=5, lw=1, alpha=0.8)
        # MJX baseline overlay (precision-matched). Two curves, both doing B*N
        # total evals like cusadi2: batched (parallel horizon) and rollout
        # (N sequential mjx.step()s). FK panels use the fk_* series, Jacobian
        # panels the jac_* series.
        kind = KIND[func]
        mjx = _read_mjx(out_dir, dtype)
        for suffix, style, mk, lab in [
            ("batched", "--", "s", f"MJX batched, B·N ({dtype})"),
            ("rollout", ":",  "^", f"MJX rollout, N steps ({dtype})"),
        ]:
            s = mjx.get(f"{kind}_{suffix}")
            if s:
                _errline(ax, s, marker=mk, ms=4, color="black", ls=style, label=lab)
        # mujoco-warp (fp32, no autodiff): FK and analytical-Jacobian panels, and
        # only on the fp32 figure — it can't be compared to fp64 cusadi2.
        if kind in ("fk", "jac") and dtype == "fp32":
            w = _read_runs(out_dir, "warp.json").get(kind)
            if w:
                _errline(ax, w, marker="D", ms=4, color="#9467bd", ls="-.", label="mujoco-warp (fp32)")
        _cpu_lines(ax, out_dir, dtype_name, func, n_stages)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{robot} {title}")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.set_xlabel("batch size B")
        ax.set_ylabel("wall-clock latency (µs)")
    if py_data:
        axes.flat[0].plot([], [], color="gray", ls="--", marker="x", ms=5, lw=1,
                          label="dashed ×: same, called from Python (ctypes)")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{robot} — single-call wall-clock latency vs batch  "
                 f"({dtype}, horizon N={n_stages}, total work = B·N; mean ± std over {n_runs} run"
                 f"{'s' if n_runs != 1 else ''})", fontsize=13)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))

    stem = os.path.join(out_dir, f"latency_{dtype_name}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")
    return f"{stem}.png"


def plot_all(dtypes, out_dir, n_stages, robot):
    return [plot_one(d, out_dir, n_stages, robot) for d in dtypes]


# ---------------------------------------------------------------------------
# Throughput plots: per-eval cost (ns) vs batch, ALL launch/dispatch overhead
# removed on both sides. cusadi2 uses its cudaEvent kernel-only time; MJX uses the
# repeat-subtraction marginal GPU time (*_batched_gpu). Divisor is B·N (both do B·N
# evals). This is the like-for-like GPU throughput comparison — where the ~10x
# wall-clock gap turns out to be launch overhead, not compute.
# ---------------------------------------------------------------------------
def _read_gpu(out_dir, fp, n_stages):
    """MJX marginal GPU: {kind: ([B...], [ns_per_eval mean...], [std...])} from *_batched_gpu."""
    out = {}
    for key, (bs, us, sd) in _read_mjx(out_dir, fp).items():
        if key.endswith("_batched_gpu"):
            per = [1e3 / (b * n_stages) for b in bs]                       # us -> ns per eval
            out[key[:-len("_batched_gpu")]] = (bs, [u * f for u, f in zip(us, per)], [s * f for s, f in zip(sd, per)])
    return out


def plot_throughput_one(dtype_name, out_dir, n_stages, robot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _read(out_dir, f"latency_{dtype_name}.csv", col="latency_us")     # kernel-only, overhead-free
    dtype = "fp64" if dtype_name == "double" else "fp32"
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for ax, (func, title) in zip(axes.flat, FUNCS):
        series = data.get(func, {})
        for cfg, label in CONFIGS:
            if cfg in series:
                bs, us, sd = series[cfg]
                per = [1e3 / (b * n_stages) for b in bs]                    # us -> ns per eval
                _errline(ax, (bs, [u * f for u, f in zip(us, per)], [s * f for s, f in zip(sd, per)]),
                         marker="o", ms=4, label=label)
        kind = KIND[func]
        g = _read_gpu(out_dir, dtype, n_stages).get(kind)
        if g:
            _errline(ax, g, marker="s", ms=4, color="black", ls="--", label=f"MJX batched, GPU-only ({dtype})")
        _cpu_lines(ax, out_dir, dtype_name, func, n_stages, per_eval=True)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{robot} {title}")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.set_xlabel("batch size B")
        ax.set_ylabel("GPU time per eval (ns)")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{robot} — per-eval GPU throughput vs batch  (launch overhead "
                 f"removed; {dtype}, N={n_stages}, {n_stages}·B evals)", fontsize=13)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    stem = os.path.join(out_dir, "throughput_" +
                        ("double" if dtype == "fp64" else "float"))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")
    return f"{stem}.png"


def plot_throughput_all(dtypes, out_dir, n_stages, robot):
    return [plot_throughput_one(d, out_dir, n_stages, robot) for d in dtypes]


# ---------------------------------------------------------------------------
# Compile / codegen time plots. cusadi2 and mujoco-warp compile once (batch- and,
# for cusadi2, function-independent kernel), so they're flat lines; MJX re-JITs per
# batch shape, so it rises. y = seconds.
# ---------------------------------------------------------------------------
def _read_mjx_compile(out_dir, fp):
    """MJX first-call JIT time per shape (paid once; identical in every run file)."""
    return _read_runs(out_dir, f"mjx_{fp}.json", section="compile")


def plot_compile_one(dtype_name, out_dir, n_stages, robot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fp = "fp64" if dtype_name == "double" else "fp32"
    cpath = os.path.join(out_dir, f"compile_cusadi2_{dtype_name}.json")
    cc = json.load(open(cpath)) if os.path.exists(cpath) else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True)
    for ax, (func, title) in zip(axes.flat, FUNCS):
        kind = KIND[func]
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
        if kind in ("fk", "jac") and fp == "fp32":
            w = _read_warp_compile(out_dir, "compile_once_s" if kind == "fk" else "compile_jac_once_s")
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
    have = [d for d in dtypes if run_csvs(res_dir, f"latency_{d}.csv")]
    plot_all(have, res_dir, n_stages, robot)
    plot_throughput_all(have, res_dir, n_stages, robot)
    plot_compile_all(res_dir, [d for d in dtypes if os.path.exists(
        os.path.join(res_dir, f"compile_cusadi2_{d}.json"))], n_stages, robot)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: plot.py <results/robot_dir> [robot_name]")
    res_dir = sys.argv[1]
    robot = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.normpath(res_dir))
    plot_robot(robot, res_dir, n_stages=8)
