"""Tick and per-call timing of the safe predictive sampling through the identical driver, every evaluator (fp32).

    python bench.py [--T 10 20 30 50 100] [--K 256] [--Ks 64 128 256 512 1024 2048] [--ticks 60] [--evaluators cusadi2 frax mjx cusadi cpu]

For every (evaluator, K, T): the tick (CUDA events around the whole tick, median / p99 over `ticks` ticks on the model
plant from the frax default configuration toward a waypoint behind the obstacle) and the pieces timed alone on the
sample buffers: the T-step rollout, one model call, one integrate call, one projection. Appends results/bench.csv.
"""
import argparse, csv, os, sys, time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, nargs="+", default=[10, 20, 30, 50, 100])
ap.add_argument("--K", type=int, default=256, help="sample count of the T sweep")
ap.add_argument("--Ks", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048], help="sample counts of the K sweep (at --KT)")
ap.add_argument("--KT", type=int, default=100)
ap.add_argument("--dtype", default="float", choices=["float"]); ap.add_argument("--ticks", type=int, default=60)
ap.add_argument("--evaluators", nargs="+", default=["cusadi2", "frax", "mjx", "cusadi", "cpu"])
ap.add_argument("--out", default=str(HERE / "results" / "bench.csv"))
a = ap.parse_args()
from driver import SafePS
import common
common.ensure_frax()
from frax import load_panda
q0 = np.asarray(load_panda().default_configuration)
WP = np.array([[0.05, 0.55, 0.55]])
FIELDS = ["evaluator", "dtype", "K", "T", "tick_median_ms", "tick_p99_ms", "tick_max_ms", "wall_median_ms", "rollout_ms", "model_ms", "integrate_ms", "project_ms", "info", "ticks"]
out = Path(a.out); out.parent.mkdir(exist_ok=True); new = not out.exists()
cases = [(a.K, T) for T in a.T] + [(K, a.KT) for K in a.Ks if (K, a.KT) not in [(a.K, T) for T in a.T]]
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader()
    for ev in a.evaluators:
        for K, T in cases:
            try:
                d = SafePS(ev, K=K, T=T)
            except Exception as e:
                print(f"{ev} K={K} T={T}: {type(e).__name__}: {str(e)[:160]}", flush=True); continue
            x = np.zeros((1, 16)); x[0, :7] = q0; d.set_plant(x); d.set_plan(np.tile(q0, (1, 4)))
            g, wl = [], []
            for t in range(a.ticks + 5):
                u, stat, gm, wm = d.tick(None, WP)
                if t >= 5: g.append(gm); wl.append(wm)
            g = np.array(g)
            row = dict(evaluator=ev, dtype=a.dtype, K=K, T=T, tick_median_ms=f"{np.median(g):.3f}", tick_p99_ms=f"{np.percentile(g, 99):.3f}", tick_max_ms=f"{g.max():.3f}",
                       wall_median_ms=f"{np.median(wl):.3f}", rollout_ms=f"{d.time_rollout(20)[0]:.3f}", model_ms=f"{d.time_model(50)[0]:.4f}",
                       integrate_ms=f"{d.time_integrate(50)[0]:.4f}", project_ms=f"{d.time_project(50)[0]:.4f}", info=d.info, ticks=a.ticks)
            w.writerow(row); f.flush()
            print(f"{ev:8s} K={K:4d} T={T:3d}  tick {float(row['tick_median_ms']):7.2f} ms (p99 {float(row['tick_p99_ms']):7.2f})  rollout {float(row['rollout_ms']):7.2f}  "
                  f"model {float(row['model_ms']):6.3f}  integrate {float(row['integrate_ms']):6.3f}  project {float(row['project_ms']):6.3f}", flush=True)
            d.close()
