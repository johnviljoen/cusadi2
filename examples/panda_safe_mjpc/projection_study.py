"""The shared projection kernel's algorithm (greedy most-violated-row iterations + box clip, model.project_greedy_np) vs the elastic QP: row residual and, what matters, the
barrier minimum along 20-step filtered rollouts from random states with sampled plans. results/projection_study.json"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common
common.configure("gpu", x64=True)
import jax, jax.numpy as jnp, numpy as np, qpax
from frax_model import FraxStep
from model import NV, PARAMS, N_KNOT, DT
from dataset import make_dataset

T = 20; n = 2048
fs = FraxStep(T); ds = make_dataset(fs.base, n, 0, T)
X, Y, RP, PAR = [jnp.asarray(ds[k]) for k in ("x", "Y", "rp", "par")]
kind = np.asarray(ds["kind"]); ncbf = fs.cfg.num_cbf; m = NV; u_max = fs.u_max

def nominal(x, Y, par):
    q, v, tau = x[:m], x[m:2 * m], x[2 * m]
    P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
    M = fs.robot.mass_matrix(q); g = fs.robot.gravity_vector(q); c = fs.robot.centrifugal_coriolis_vector(q, v)
    w = fs.weights(tau); q_des = sum(w[j] * Y[j * m:(j + 1) * m] for j in range(N_KNOT))
    return jnp.clip(M @ (P["kp"] * (q_des - q) - P["kd"] * v) + g + c, -u_max, u_max), M, g, c
def rows(z):
    return fs.cbf.G_qp(z, jnp.zeros(m))[:ncbf], fs.cbf.h_qp(z, jnp.zeros(m))[:ncbf]
def proj_greedy(u, G, h, iters):
    gg = jnp.sum(G * G, axis=1) + 1e-12
    u = jnp.clip(u, -u_max, u_max)
    for _ in range(iters):
        r = (G @ u - h) / jnp.sqrt(gg)
        i = jnp.argmax(r)
        u = u - jnp.where(r[i] > 0, (G[i] @ u - h[i]) / gg[i], 0.0) * G[i]
        u = jnp.clip(u, -u_max, u_max)
    return u
def qp(u_nom, G, h):
    Gf = jnp.concatenate([G, jnp.eye(m), -jnp.eye(m)]); hf = jnp.concatenate([h, u_max, u_max])
    return qpax.solve_qp_elastic_primal(jnp.eye(m), -u_nom, Gf, hf, penalty=jnp.asarray(fs.base.penalty), solver_tol=fs.base.solver_tol)[:m]
FILTERS = {"none (box only)": lambda u, G, h: jnp.clip(u, -u_max, u_max), "greedy 16": lambda u, G, h: proj_greedy(u, G, h, 16),
           "greedy 32": lambda u, G, h: proj_greedy(u, G, h, 32), "greedy 48 (the kernel)": lambda u, G, h: proj_greedy(u, G, h, 48),
           "elastic QP (cbfpy / qpax)": qp}

def step(filt, x, Y, par):
    u0, M, g, c = nominal(x, Y, par); z = x[:2 * m]; G, h = rows(z)
    u = filt(u0, G, h)
    a = jnp.linalg.solve(M, u - g - c); v1 = x[m:2 * m] + DT * a; q1 = x[:m] + DT * v1
    return jnp.concatenate([q1, v1, jnp.array([x[2 * m] + DT, 0.0])]), u
res = {}
U0 = jax.jit(jax.vmap(lambda x, Y, par: nominal(x, Y, par)[0]))(X, Y, PAR)
G, H = jax.jit(jax.vmap(lambda x: rows(x[:2 * m])))(X)
h2_j = jax.jit(jax.vmap(lambda x: fs.cfg.h_2(x[:2 * m])))
h2min0 = np.asarray(jax.jit(jax.vmap(lambda x: fs.cfg.h_2(x[:2 * m])))(X)).min(1)
sel = (kind == 0) & (h2min0 > 0.02)                   # benign, initially safe states for the rollout metric
print("rollout metric on", int(sel.sum()), "initially safe random states", flush=True)
Xr, Yr, Pr = X[sel], Y[sel], PAR[sel]
Xr = Xr.at[:, 2 * m].set(0.0)
for name, filt in FILTERS.items():
    U = jax.jit(jax.vmap(lambda u, G, h: filt(u, G, h)))(U0, G, H)
    viol = (np.einsum("bij,bj->bi", np.asarray(G), np.asarray(U)) - np.asarray(H)) / np.maximum(1.0, np.abs(np.asarray(H)))
    r = {f"kind{k}_inst_frac": float((viol[kind == k] > 1e-6).any(1).mean()) for k in (0, 1, 2)}
    r.update({f"kind{k}_mean_pos": float(np.maximum(viol[kind == k], 0).mean()) for k in (0, 1, 2)})
    st = jax.jit(jax.vmap(lambda x, Y, par: step(filt, x, Y, par)))
    x = Xr; hmin = np.full(int(sel.sum()), np.inf)
    for s in range(T):
        x, _ = st(x, Yr, Pr); hmin = np.minimum(hmin, np.asarray(h2_j(x)).min(1))
    r["rollout_hmin_frac_below_-0.01"] = float((hmin < -0.01).mean()); r["rollout_hmin_frac_below_0"] = float((hmin < 0).mean())
    r["rollout_hmin_min"] = float(hmin.min()); r["rollout_hmin_p05"] = float(np.percentile(hmin, 5))
    res[name] = r; print(name, {k: round(v, 4) for k, v in r.items()}, flush=True)
(HERE / "results" / "projection_study.json").write_text(json.dumps(res, indent=2))
