"""Smoke test of the native driver: R robots from the frax default configuration, model plant, one waypoint.
    python smoke.py --evaluator cusadi2|frax|mjx|cusadi|cpu --K 64 --T 25 --n-iter 1 --ticks 50
"""
import argparse, os, sys, time
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--evaluator", default="cusadi2"); ap.add_argument("--R", type=int, default=1); ap.add_argument("--K", type=int, default=64)
ap.add_argument("--T", type=int, default=25); ap.add_argument("--ticks", type=int, default=50); ap.add_argument("--n-iter", type=int, default=1)
ap.add_argument("--no-graph", action="store_true"); ap.add_argument("--noise", type=float, default=0.1)
ap.add_argument("--par", default="", help="overrides, e.g. kp=50,w_v=1.0")
ap.add_argument("--wp", default="0.1,0.55,0.45")
a = ap.parse_args()
from driver import SafePS
from model import PARAMS, PAR_DEFAULT, N_KNOT
par = PAR_DEFAULT.copy()
for kv in filter(None, a.par.split(',')):
    k, v = kv.split('='); par[[n for n, _ in PARAMS].index(k)] = float(v)
print('par', dict(zip([n for n, _ in PARAMS], par)))
import common
common.ensure_frax()
from frax import load_panda
robot = load_panda()
q0 = np.asarray(robot.default_configuration)
import jax, jax.numpy as jnp
ee = jax.jit(jax.vmap(lambda q: robot.ee_transform(q)[:3, 3]))
h2 = None
d = SafePS(a.evaluator, a.K, a.T, R=a.R, n_iter=a.n_iter, graph=not a.no_graph, noise=a.noise, par=par)
print(d.info, flush=True)
x = np.zeros((a.R, 16)); x[:, :7] = q0
d.set_plant(x); d.set_plan(np.tile(q0, (a.R, N_KNOT)))
wp = np.tile(np.array([float(v) for v in a.wp.split(",")]), (a.R, 1))
print("ee start", np.asarray(ee(jnp.asarray(x[:1, :7])))[0])
t_gpu, t_wall = [], []
for t in range(a.ticks):
    u, stat, g, w = d.tick(None, wp)
    t_gpu.append(g); t_wall.append(w)
    if t % 10 == 0 or t == a.ticks - 1:
        xp = d.plant(); p = np.asarray(ee(jnp.asarray(xp[:, :7])))
        dist = np.linalg.norm(p - wp, axis=1)
        print(f"tick {t:3d}: gpu {g:6.2f} ms wall {w:6.2f} | cost min {stat[:,0].mean():8.3f} mean {stat[:,1].mean():8.3f} k* {int(stat[0,3]):3d} | "
              f"|u| {np.abs(u).max():6.1f} finite {np.isfinite(u).all()} | ee dist mean {dist.mean():.3f} min {dist.min():.3f} | |v| max {np.abs(xp[:, 7:14]).max():.2f}", flush=True)
t_gpu = np.array(t_gpu[5:]); print(f"gpu ms median {np.median(t_gpu):.2f} p99 {np.percentile(t_gpu, 99):.2f} | wall median {np.median(t_wall[5:]):.2f}")
print("iteration ms", d.time_iteration(20), "rollout ms", d.time_rollout(20), "model ms", d.time_model(50), "integrate ms", d.time_integrate(50), "project ms", d.time_project(50))
if a.evaluator == "frax": print("last pjrt exec ms", d.last_pjrt_exec_ms())
d.close()
