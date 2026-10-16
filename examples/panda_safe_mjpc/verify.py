"""Gate 1: the cusadi2 model / integrate functions (CasADi) equal frax + cbfpy's (JAX) in fp64, and the driver's
projection kernel equals its numpy reference.

    python verify.py [--T 20] [--n 2048]

Writes results/verify.json.
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=20); ap.add_argument("--n", type=int, default=2048)
args = ap.parse_args()
common.configure("gpu", x64=True)
import jax, jax.numpy as jnp, numpy as np
from frax_model import FraxStep
from model import build_model_function, build_integrate_function, project_greedy_np, NV, PROJ_ITERS
from dataset import make_dataset

t0 = time.perf_counter(); fm, robot, im = build_model_function(args.T); fi, _, ii = build_integrate_function(); t_build = time.perf_counter() - t0
print("casadi model:", im, " integrate:", ii, f"({t_build:.1f} s)", flush=True)
fs = FraxStep(args.T)
ds = make_dataset(fs.base, args.n, 0, args.T)
X, Y, RP, PAR = [jnp.asarray(ds[k]) for k in ("x", "Y", "rp", "par")]
un_f, G_f, h_f = [np.asarray(a) for a in jax.jit(jax.vmap(fs.model))(X, Y, RP, PAR)]
mm = fm.map(args.n, "thread", 16)
un_c, G_c, h_c = [np.asarray(a) for a in mm(ds["x"].T, ds["Y"].T, ds["rp"].T, ds["par"].T)]
un_c = un_c.T; h_c = h_c.T; G_c = G_c.reshape(64, 7, args.n, order="F").transpose(2, 0, 1) if G_c.shape == (64, 7 * args.n) else G_c
# CasADi's map returns the horizontally stacked 64 x (7 n) matrix: column block k is instance k
G_c = np.stack([np.asarray(G_c[:, 7 * k:7 * (k + 1)]) for k in range(args.n)]) if G_c.ndim == 2 else G_c
def rel(a, b):
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))
res = dict(T=args.T, n=args.n, model=dict(u_nom=rel(un_c, un_f), G=rel(G_c, G_f), h=rel(h_c, h_f)))
print("model rel err:", res["model"], flush=True)
u_max = np.asarray(robot.actuated_joint_max_forces)
U = np.stack([project_greedy_np(un_f[i], G_f[i], h_f[i], u_max, PROJ_ITERS) for i in range(args.n)])
Uj = jnp.asarray(U)
x1_f = np.asarray(jax.jit(jax.vmap(fs.integrate))(X, Uj, RP, PAR))
im_ = fi.map(args.n, "thread", 16)
x1_c = np.asarray(im_(ds["x"].T, U.T, ds["rp"].T, ds["par"].T)).T
res["integrate"] = dict(x1=rel(x1_c, x1_f))
print("integrate rel err:", res["integrate"], flush=True)
# the projection kernel against the numpy reference, through the driver on the same data
from driver import SafeMppi
d = SafeMppi("cusadi2", K=args.n, T=args.T, graph=False)
Yd = np.asarray(ds["Y"]); d.set_plan(Yd[:1])                   # the driver samples around a plan; feed the dataset rows instead via a tick + filter dump
# run the model + projection on the dataset states directly: emulate with sample 0 = plan, others = noise; here we only
# need identical inputs to the kernel, so compare the kernel output against the reference on the driver's own data
d.tick(None, np.zeros((1, 3)))
un_k, G_k, h_k, u_k = d.filter_data()
U_ref = np.stack([project_greedy_np(un_k[i], G_k[i], h_k[i], u_max, PROJ_ITERS) for i in range(d.B)])
res["kernel"] = dict(u_vs_numpy=rel(u_k, U_ref), rows_violated_after=float(((np.einsum("bij,bj->bi", G_k, u_k) - h_k) > 1e-6).any(1).mean()))
print("projection kernel vs numpy:", res["kernel"], flush=True)
d.close()
res["casadi"] = dict(model=im, integrate=ii, build_s=t_build)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "verify.json").write_text(json.dumps(res, indent=2, default=str))
ok = all(v < 1e-9 for v in res["model"].values()) and res["integrate"]["x1"] < 1e-9 and res["kernel"]["u_vs_numpy"] < 1e-9
print("GATE 1", "PASS" if ok else "FAIL")
