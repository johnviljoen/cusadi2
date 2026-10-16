"""Gate: the CasADi model / integrate functions (cusadi2, CusADi-1, CasADi codegen) equal frax + cbfpy's and
MJX + cbfpy's (JAX) in fp64 on the cupboard scene, and the driver's projection kernel equals its numpy reference.

    python verify.py [--T 100] [--n 1000]

States: 60 % near the feasibility path (IK configurations of results/feasibility.json + 0.1 rad noise, so spheres sit
near slab faces, edges and the cup) and 40 % uniform in the joint box; v ~ N(0, 0.5), plan knots around q.
Agreement per state: max |a - b| / max(1, max |b|) over the output. Gate: rows (u_nom, G, h) <= 1e-10, step <= 1e-13.
States over the gate are listed with their worst row and its barrier value; they are reported, not dropped. Writes results/verify.json.
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=int, default=100); ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--no-kernel", action="store_true")
args = ap.parse_args()
common.configure("gpu", x64=True)
import jax, jax.numpy as jnp, numpy as np
import casadi as ca
from frax_model import FraxStep
from mjx_model import MjxStep
from model import build_model_function, build_integrate_function, build_aux_function, project_greedy_np, NV, NX, ND, N_KNOT, PAR_DEFAULT, PROJ_ITERS, N_CBF
import scene

ROW_TOL, STEP_TOL = 1e-10, 1e-13
t0 = time.perf_counter(); fm, robot, im = build_model_function(args.T); fi, _, ii = build_integrate_function(); t_build = time.perf_counter() - t0
print("casadi model:", im, " integrate:", ii, f"({t_build:.1f} s)", flush=True)
fs, ms = FraxStep(args.T), MjxStep(args.T)

# ---- states ----
rng = np.random.default_rng(0)
lo, hi = np.asarray(robot.actuated_joint_lower_limits), np.asarray(robot.actuated_joint_upper_limits)
feas = json.loads((HERE / "results" / "feasibility.json").read_text())
Qp = np.array([p["q"] for p in feas["path"]] + [feas["pregrasp"]["q"]])
n_near = int(0.6 * args.n)
q = np.concatenate([Qp[rng.integers(0, len(Qp), n_near)] + rng.normal(0, 0.1, (n_near, NV)), rng.uniform(lo, hi, (args.n - n_near, NV))])
q = np.clip(q, lo, hi)
x = np.zeros((args.n, NX)); x[:, :NV] = q; x[:, NV:2 * NV] = rng.normal(0, 0.5, (args.n, NV)); x[:, 2 * NV] = rng.uniform(0, 2.0, args.n)
Y = np.clip(np.repeat(q, N_KNOT, axis=0).reshape(args.n, ND) + rng.normal(0, 0.3, (args.n, ND)), np.tile(lo, N_KNOT), np.tile(hi, N_KNOT))
rp = np.tile(scene.P_PREGRASP, (args.n, 1)); par = np.tile(PAR_DEFAULT, (args.n, 1))
fa = build_aux_function()
h2 = np.asarray(fa.map(args.n, "thread", 16)(x.T)[0]).T
sl = scene.group_slices()
geo = h2[:, sl["floor"].start:]
near = (np.abs(geo) < 0.02).any(1)
print(f"states: {args.n} ({n_near} near the path); {near.mean()*100:.0f} % have a scene barrier within 2 cm of 0, "
      f"{(geo < 0).any(1).mean()*100:.0f} % are in collision", flush=True)

# ---- evaluate ----
X, Yj, RP, PAR = (jnp.asarray(a) for a in (x, Y, rp, par))
out = {"frax": [np.asarray(a) for a in jax.jit(jax.vmap(fs.model))(X, Yj, RP, PAR)],
       "mjx": [np.asarray(a) for a in jax.jit(jax.vmap(ms.model))(X, Yj, RP, PAR)]}
un_c, G_c, h_c = [np.asarray(a) for a in fm.map(args.n, "thread", 16)(x.T, Y.T, rp.T, par.T)]
G_c = np.stack([G_c[:, NV * k:NV * (k + 1)] for k in range(args.n)])   # map stacks the N_CBF x 7 blocks horizontally
out["casadi"] = [un_c.T, G_c, h_c.T]
u_max = np.asarray(robot.actuated_joint_max_forces)
U = np.stack([project_greedy_np(out["casadi"][0][i], out["casadi"][1][i], out["casadi"][2][i], u_max, PROJ_ITERS) for i in range(args.n)])
x1 = {"casadi": np.asarray(fi.map(args.n, "thread", 16)(x.T, U.T, rp.T, par.T)).T,
      "frax": np.asarray(jax.jit(jax.vmap(fs.integrate))(X, jnp.asarray(U), RP, PAR)),
      "mjx": np.asarray(jax.jit(jax.vmap(ms.integrate))(X, jnp.asarray(U), RP, PAR))}


def rel_state(a, b):
    a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
    return np.abs(a - b).max(1) / np.maximum(1.0, np.abs(b).max(1))


res = dict(T=args.T, n=args.n, near_fraction=float(near.mean()), collision_fraction=float((geo < 0).any(1).mean()), pairs={})
ok = True
for a_, b_ in [("casadi", "frax"), ("casadi", "mjx"), ("mjx", "frax")]:
    e_u = rel_state(out[a_][0], out[b_][0]); e_G = rel_state(out[a_][1], out[b_][1]); e_h = rel_state(out[a_][2], out[b_][2])
    e_rows = np.maximum(np.maximum(e_u, e_G), e_h); e_x = rel_state(x1[a_], x1[b_])
    bad = np.where((e_rows > ROW_TOL) | (e_x > STEP_TOL))[0]
    worst_rows = []
    for i in bad[np.argsort(-e_rows[bad])][:5]:
        dh = np.abs(out[a_][2][i] - out[b_][2][i]); dG = np.abs(out[a_][1][i] - out[b_][1][i]).max(1)
        r = int(np.argmax(np.maximum(dh, dG)))
        worst_rows.append(dict(state=int(i), rel=float(e_rows[i]), row=r, h2=float(h2[i, r]) if r < h2.shape[1] else None))
    res["pairs"][f"{a_}-{b_}"] = dict(rows_max=float(e_rows.max()), rows_median=float(np.median(e_rows)), u_nom=float(e_u.max()), G=float(e_G.max()),
                                      h=float(e_h.max()), step_max=float(e_x.max()), n_over=int(len(bad)), worst=worst_rows)
    print(f"{a_:6s} vs {b_:5s}: rows max {e_rows.max():.1e} (u_nom {e_u.max():.1e}, G {e_G.max():.1e}, h {e_h.max():.1e}), "
          f"step max {e_x.max():.1e}; states over the gate: {len(bad)}", flush=True)
    for w in worst_rows:
        print(f"     state {w['state']}: rel {w['rel']:.1e} at row {w['row']} (h2 = {w['h2']})")
    ok &= len(bad) == 0

# ---- every evaluator through the driver (sm_eval_model / sm_eval_integrate: the calls exactly as the rollout makes them)
# on the first 256 states, against the CasADi functions. This is the test of what the benchmark runs: the plans, unrolled
# kernels and C code are built from the CasADi graph in build.py, the frax / MJX programs are exported from JAX. ----
if not args.no_kernel:
    from driver import SafeMppi
    nb = 256; sel = slice(0, nb)
    res["driver"] = {}
    for ev in ["cusadi2", "frax", "mjx", "cusadi", "cpu"]:
        try:
            d = SafeMppi(ev, K=nb, T=args.T, graph=False)
        except Exception as e:
            res["driver"][ev] = dict(error=str(e)[:200]); print(f"driver {ev}: {type(e).__name__}: {str(e)[:120]}"); ok = False; continue
        un_d, G_d, h_d = d.eval_model(x[sel], Y[sel], rp[sel], par[sel]); x1_d = d.eval_integrate(x[sel], U[sel], rp[sel], par[sel]); d.close()
        e_rows = np.maximum(np.maximum(rel_state(un_d, out["casadi"][0][sel]), rel_state(G_d, out["casadi"][1][sel])), rel_state(h_d, out["casadi"][2][sel]))
        e_x = rel_state(x1_d, x1["casadi"][sel])
        res["driver"][ev] = dict(rows_max=float(e_rows.max()), step_max=float(e_x.max()), n_over=int(((e_rows > ROW_TOL) | (e_x > STEP_TOL)).sum()))
        print(f"driver {ev:8s} vs casadi: rows max {e_rows.max():.1e}, step max {e_x.max():.1e}; states over the gate: {res['driver'][ev]['n_over']}", flush=True)
        ok &= res["driver"][ev]["n_over"] == 0

# ---- the projection kernel (now NCBF = N_CBF rows, (N_CBF + 31) / 32 per lane) against numpy, through the driver ----
if not args.no_kernel:
    # plant states inside / at the mouth of the cupboard and plans pushed 0.4 rad off them, so the rollouts press into
    # the walls and the projection is active on many samples (the last rollout step's projection is compared)
    from driver import SafeMppi
    d = SafeMppi("cusadi2", K=256, T=args.T, graph=False)
    kern = []
    for j, qs in enumerate([feas["pregrasp"]["q"], feas["path"][len(feas["path"]) // 2]["q"], feas["path"][-4]["q"]]):
        qs = np.asarray(qs)
        xp = np.zeros((1, NX)); xp[0, :NV] = qs; d.set_plant(xp)
        d.set_plan(np.clip(np.tile(qs + 0.4 * rng.choice([-1.0, 1.0], NV), (1, N_KNOT)), np.tile(lo, N_KNOT), np.tile(hi, N_KNOT)))
        d.tick(None, scene.P_PREGRASP[None])
        un_k, G_k, h_k, u_k = d.filter_data()
        U_ref = np.stack([project_greedy_np(un_k[i], G_k[i], h_k[i], u_max, PROJ_ITERS) for i in range(d.B)])
        kern.append(dict(u_vs_numpy=float(rel_state(u_k, U_ref).max()),
                         active_fraction=float((np.einsum("bij,bj->bi", G_k, un_k) - h_k > 0).any(1).mean()),
                         rows_violated_after=float(((np.einsum("bij,bj->bi", G_k, u_k) - h_k) > 1e-6).any(1).mean()),
                         active_rows_beyond_64=int(((np.einsum("bij,bj->bi", G_k, un_k) - h_k)[:, 64:] > 0).sum())))
        print(f"projection kernel vs numpy, plant state {j}: {kern[-1]}", flush=True)
    res["kernel"] = dict(n_cbf=int(G_k.shape[1]), cases=kern)
    ok &= all(k["u_vs_numpy"] < 1e-12 for k in kern) and res["kernel"]["n_cbf"] == N_CBF and any(k["active_rows_beyond_64"] > 0 for k in kern)
    d.close()
res["casadi"] = dict(model=im, integrate=ii, build_s=t_build)
res["gate"] = bool(ok)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "verify.json").write_text(json.dumps(res, indent=2, default=str))
print("GATE", "PASS" if ok else "FAIL")
