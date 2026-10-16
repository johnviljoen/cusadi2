"""Rollout-sweep latency against the sample count: cusadi2 level-parallel, cusadi2 serial mode (the
original-cusadi execution model) and CasADi C on the CPU, all evaluating the same go2_step1 on the
same inputs (mppi_bench.cu, cpu_bench.py).

    python bench.py <build_dir> [n_timed]      -> results/bench.csv, results/bench_latency.{png,pdf}

One sweep = 100 chained evaluations (0.4 s horizon at 4 ms) over B samples; a control step runs
`iters` sweeps plus the small MPPI kernels.
"""
import csv
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "build")
N_TIMED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
RES = os.path.join(HERE, "results")
B_GPU = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
B_CPU = {1: [64, 128, 256, 512, 1024], 16: [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]}
PY = sys.executable


def run(cmd):
    print("  " + " ".join(cmd), flush=True)
    return [line.split(",") for line in subprocess.check_output(cmd).decode().strip().splitlines()]


def main():
    os.makedirs(RES, exist_ok=True)
    rows = []
    for tag in ("f32", "f64"):
        for mode in ("level", "serial"):
            rows += run([os.path.join(BUILD, f"mppi_bench_{tag}"), BUILD, mode, str(N_TIMED)] + [str(b) for b in B_GPU])
    for dtype in ("float", "double"):
        for threads, bs in B_CPU.items():
            rows += run([PY, os.path.join(HERE, "cpu_bench.py"), BUILD, dtype, str(threads), str(max(3, N_TIMED // 2))] + [str(b) for b in bs])
    with open(os.path.join(RES, "bench.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["mode", "dtype", "B", "launches", "median_ms", "min_ms", "checksum"])
        w.writerows(rows)
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = [("level", "cusadi2 level-parallel", "#2a78d6"), ("serial", "cusadi2 serial (thread per sample)", "#eb6834"),
              ("cpu16", "CPU, 16 threads", "#1baf7a"), ("cpu1", "CPU, 1 thread", "#eda100")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, dtype in zip(axes, ("float", "double")):
        for mode, label, color in series:
            pts = sorted((int(r[2]), float(r[4])) for r in rows if r[0] == mode and r[1] == dtype)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color, ms=3.5, lw=1.8, label=label)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("samples B"); ax.set_title(f"{dtype}: one rollout sweep (100 steps)")
        ax.grid(True, which="both", color="0.9", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("latency [ms]"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RES, f"bench_latency.{ext}"), dpi=160)


if __name__ == "__main__":
    main()
