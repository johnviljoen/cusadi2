"""Tick and per-call timing of the CPU-only safe predictive sampling ("CPU idealized"), the protocol of bench.py: for every
(K, T) the tick with n_iter iterations (wall clock around the whole tick, median / p99 over `ticks` ticks on the model plant
from the default configuration toward a waypoint behind the obstacle) and the pieces timed alone on the sample buffers.
Appends rows with evaluator "cpu_ideal" to --out in bench.py's column layout.

    python bench_cpu.py [--T 10 25 35 50 100] [--K 256] [--Ks 64 128 256 512 1024 2048] [--KT 35] [--n-iter 3] [--ticks 60] [--runs 5]
"""
import argparse, csv, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cpu_driver import CpuPS  # noqa: E402
from model import N_KNOT  # noqa: E402
import numpy as np
import common
common.ensure_frax()
from frax import load_panda

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, nargs="+", default=[10, 25, 35, 50, 100]); ap.add_argument("--K", type=int, default=256)
ap.add_argument("--Ks", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048]); ap.add_argument("--KT", type=int, default=35)
ap.add_argument("--n-iter", type=int, default=3); ap.add_argument("--ticks", type=int, default=60); ap.add_argument("--runs", type=int, default=1)
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--dtype", default="float")
ap.add_argument("--out", default=str(HERE / "results" / "bench_f32_cpu_ideal_runs.csv"))
a = ap.parse_args()
q0 = np.asarray(load_panda().default_configuration)
WP = np.array([[0.05, 0.55, 0.55]])
FIELDS = ["evaluator", "dtype", "K", "T", "n_iter", "tick_median_ms", "tick_p99_ms", "tick_max_ms", "wall_median_ms", "iteration_ms", "rollout_ms", "model_ms", "integrate_ms", "project_ms", "info", "ticks"]
out = Path(a.out); out.parent.mkdir(exist_ok=True); new = not out.exists()
cases = [(a.K, T) for T in a.T] + [(K, a.KT) for K in a.Ks if (K, a.KT) not in [(a.K, T) for T in a.T]]
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader()
    for run in range(a.runs):
        for K, T in cases:
            d = CpuPS(K=K, T=T, n_iter=a.n_iter, dtype=a.dtype, threads=a.threads)
            x0 = np.zeros((1, 16)); x0[0, :7] = q0; d.set_plant(x0); d.set_plan(np.tile(q0, (1, N_KNOT)))
            g = []
            for k in range(a.ticks + 5):
                _, _, ms = d.tick(None, WP); g.append(ms)
            g = np.array(g[5:])
            row = dict(evaluator="cpu_ideal", dtype=a.dtype, K=K, T=T, n_iter=a.n_iter, tick_median_ms=f"{np.median(g):.3f}", tick_p99_ms=f"{np.percentile(g, 99):.3f}", tick_max_ms=f"{g.max():.3f}",
                       wall_median_ms=f"{np.median(g):.3f}", iteration_ms=f"{d.time_iteration(20)[0]:.3f}", rollout_ms=f"{d.time_rollout(20)[0]:.3f}", model_ms=f"{d.time_model(50)[0]:.4f}",
                       integrate_ms=f"{d.time_integrate(50)[0]:.4f}", project_ms=f"{d.time_project(50)[0]:.4f}", info=d.info, ticks=a.ticks)
            w.writerow(row); f.flush()
            print(f"run {run} cpu_ideal K={K:4d} T={T:3d}  tick {float(row['tick_median_ms']):7.2f} ms (p99 {float(row['tick_p99_ms']):7.2f})  iteration {float(row['iteration_ms']):7.2f}  "
                  f"rollout {float(row['rollout_ms']):7.2f}  model {float(row['model_ms']):6.3f}  integrate {float(row['integrate_ms']):6.3f}  project {float(row['project_ms']):6.3f}", flush=True)
            d.close()
