"""Gates of the MJPC-faithful predictive sampling driver (cusadi2 evaluator, fp32), along a model-plant trajectory:

  spline     the CasADi spline weights (model.interp_weights, inside the plans) equal the numpy reference (and a direct
             cubic Hermite with MJPC's slopes) at random plan clocks, to 1e-12
  resample   the driver's k_resample kernel (the nominal one period later) equals model.resample_np of the winning
             sample, to fp32 tolerance
  gather     the applied torque (the gathered step-0 filtered torque of the selected rollout) equals the selected plan's
             torque recomputed as model + projection at the tick's plant state, exactly
  monotone   within a tick, the cost of sample 0 (the nominal) at iteration i + 1 equals the cost min of iteration i
             bitwise (the same plan rolled out from the same state), so the plan's cost never increases across iterations
  project    the projection kernel equals its numpy reference on the last rollout step's buffers

    python verify_ps.py [--K 64] [--T 25] [--n-iter 4] [--ticks 40]

Writes results/verify_ps.json.
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import numpy as np
import casadi as ca

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=256); ap.add_argument("--T", type=int, default=35); ap.add_argument("--n-iter", type=int, default=3)
ap.add_argument("--ticks", type=int, default=40)
a = ap.parse_args()
from driver import SafePS
from model import interp_weights, spline_np, resample_np, knot_dt, project_greedy_np, N_KNOT, NV, NX, PROJ_ITERS, DT
import common
common.ensure_frax()
from frax import load_panda
robot = load_panda(); q0 = np.asarray(robot.default_configuration); u_max = np.asarray(robot.actuated_joint_max_forces)
WP = np.array([[0.05, 0.55, 0.55]])
res = dict(K=a.K, T=a.T, n_iter=a.n_iter, ticks=a.ticks)

# ---- spline: CasADi weights vs numpy vs a direct Hermite ----
kdt = knot_dt(a.T); rng = np.random.default_rng(0)
Yt = rng.normal(size=(N_KNOT, 3)); times = np.arange(N_KNOT) * kdt
def hermite(tau):
    if tau <= times[0]: return Yt[0]
    if tau >= times[-1]: return Yt[-1]
    s = int(np.searchsorted(times, tau, side="right")) - 1
    def slope(j):
        if j == 0: return (Yt[1] - Yt[0]) / kdt
        if j == N_KNOT - 1: return (Yt[j] - Yt[j - 1]) / kdt
        return 0.5 * ((Yt[j + 1] - Yt[j]) / kdt + (Yt[j] - Yt[j - 1]) / kdt)
    t = (tau - times[s]) / kdt
    return (2 * t**3 - 3 * t**2 + 1) * Yt[s] + (t**3 - 2 * t**2 + t) * kdt * slope(s) + (-2 * t**3 + 3 * t**2) * Yt[s + 1] + (t**3 - t**2) * kdt * slope(s + 1)
taus = np.concatenate([rng.uniform(-0.1, (a.T + 5) * DT, 300), times, np.arange(a.T) * DT])
tau_s = ca.SX.sym("tau"); wf = ca.Function("w", [tau_s], [ca.vertcat(*interp_weights(tau_s, kdt))])
e_np = max(float(np.abs(spline_np(Yt, t, kdt) - hermite(t)).max()) for t in taus)
e_ca = max(float(np.abs(np.asarray(wf(t)).ravel() - np.asarray(interp_weights(float(t), kdt, np))).max()) for t in taus)
res["spline"] = dict(numpy_vs_hermite=e_np, casadi_vs_numpy=e_ca, kdt=kdt)
print(f"spline: numpy vs direct Hermite {e_np:.1e}, CasADi weights vs numpy {e_ca:.1e} (kdt {kdt:.4f} s)", flush=True)

# ---- the driver along a model-plant trajectory, external-plant ticks so the recompute gate applies ----
d = SafePS("cusadi2", K=a.K, T=a.T, n_iter=a.n_iter, graph=True, seed=1)
print(d.info, flush=True)
x = np.zeros((1, NX)); x[0, :7] = q0; d.set_plan(np.tile(q0, (1, N_KNOT)))
g_gather, g_resample, g_mono, g_c0, g_proj, improved = [], [], [], [], [], []
for k in range(a.ticks):
    Y0_before = d.plan().copy()
    u, stat, gm, wm = d.tick(x, WP)
    rc = d.recompute(); g_gather.append(float(np.abs(u[0] - rc).max()))
    Y, xT = d.samples(); ks = int(stat[0, 3])
    ref = resample_np(Y[ks].reshape(N_KNOT, NV), kdt).reshape(-1)
    g_resample.append(float(np.abs(d.plan()[0] - ref).max() / max(1.0, np.abs(ref).max())))
    it = d.iter_stats()[:, 0, :]                     # (n_iter, 5): cost min, mean, std, ks, cost of sample 0
    g_mono.append(bool(np.all(np.diff(it[:, 0]) <= 0)))
    g_c0.append(float(np.abs(it[1:, 4] - it[:-1, 0]).max()) if a.n_iter > 1 else 0.0)
    improved.append(int((it[:, 3] != 0).sum()))
    if k % 10 == 0:                                  # projection kernel vs numpy on the last step's buffers
        un, G, h, uk = d.filter_data()
        U = np.stack([project_greedy_np(un[i], G[i], h[i], u_max, PROJ_ITERS) for i in range(d.B)])
        g_proj.append(float(np.abs(uk - U).max() / max(1.0, np.abs(U).max())))
    x = np.zeros((1, NX)); x[0, :14] = xT[ks, :14]   # follow the selected rollout one step (the model plant)
d.close()
res["gather_max_abs"] = max(g_gather); res["resample_max_rel"] = max(g_resample); res["monotone_all"] = all(g_mono)
res["nominal_cost_carry_max_abs"] = max(g_c0); res["project_max_rel"] = max(g_proj); res["improving_iterations_per_tick_mean"] = float(np.mean(improved))
print(f"gather: |u - recompute| max {res['gather_max_abs']:.1e}   resample kernel vs numpy {res['resample_max_rel']:.1e}   "
      f"cost monotone within ticks {res['monotone_all']}   nominal-cost carry |c0(i+1) - cmin(i)| max {res['nominal_cost_carry_max_abs']:.1e}   "
      f"projection vs numpy {res['project_max_rel']:.1e}   improving iterations per tick {res['improving_iterations_per_tick_mean']:.2f} of {a.n_iter}", flush=True)
ok = e_np < 1e-12 and e_ca < 1e-12 and res["gather_max_abs"] == 0.0 and res["resample_max_rel"] < 1e-5 and res["monotone_all"] and res["nominal_cost_carry_max_abs"] == 0.0 and res["project_max_rel"] < 1e-5
res["pass"] = bool(ok)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "verify_ps.json").write_text(json.dumps(res, indent=2))
print("GATES", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
