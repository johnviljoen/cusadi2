"""Iteration time of the cusadi2 evaluator over (T, K): which (T, K, n_iter) fit the 20 ms tick.
    python timing_grid.py [--T 25 50 100] [--K 64 256 512 1024]
Writes results/timing_grid.json."""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, nargs="+", default=[25, 50, 100]); ap.add_argument("--K", type=int, nargs="+", default=[64, 256, 512, 1024])
a = ap.parse_args()
from driver import SafePS
from model import N_KNOT
import common
common.ensure_frax()
from frax import load_panda
q0 = np.asarray(load_panda().default_configuration); WP = np.array([[0.05, 0.55, 0.55]])
rows = []
for T in a.T:
    for K in a.K:
        d = SafePS("cusadi2", K=K, T=T, n_iter=1)
        x = np.zeros((1, 16)); x[0, :7] = q0; d.set_plant(x); d.set_plan(np.tile(q0, (1, N_KNOT)))
        g = []
        for t in range(60):
            _, _, gm, _ = d.tick(None, WP); g.append(gm)
        g = np.array(g[10:]); it = d.time_iteration(20)[0]; m = d.time_model(50)[0]; i = d.time_integrate(50)[0]
        n_fit = int(20.0 // np.percentile(g, 99))
        rows.append(dict(T=T, K=K, tick_median_ms=float(np.median(g)), tick_p99_ms=float(np.percentile(g, 99)), iteration_ms=it, model_ms=m, integrate_ms=i, n_iter_fit=n_fit))
        print(f"T={T:3d} K={K:5d}: iteration {it:6.2f} ms (tick median {np.median(g):6.2f}, p99 {np.percentile(g, 99):6.2f}), model {m:.3f} integrate {i:.3f} -> {n_fit} iterations fit in 20 ms", flush=True)
        d.close()
(HERE / "results" / "timing_grid.json").write_text(json.dumps(rows, indent=1))
