"""What the plant-side re-filter does to the MPPI mean plan (fp32 closed loop, cusadi2 evaluator, no video).

    python refilter_study.py [--seconds 30] [--seed 1]

Every tick of the MuJoCo closed loop of closed_loop.py: the driver's applied torque is project(model(x_plant, Y_mean)).
This script recovers Y_mean (the plan before the shift), evaluates the CasADi model at the plant state in fp64, and
records how far the mean plan's own PD torque u_nom is from the CBF rows (most violated row, signed distance; number of
violated rows), how much the re-filter moved it (|u_applied - clip(u_nom)|), the same for the best sampled rollout's
plan, and what the mean plan would do without the filter (unfiltered T-step rollout: minimum barrier h2, first step
with a violated row). Writes results/refilter_study.json.
"""
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

p = argparse.ArgumentParser()
p.add_argument("--K", type=int, default=256); p.add_argument("--T", type=int, default=100)
p.add_argument("--seconds", type=float, default=30.0); p.add_argument("--seed", type=int, default=1)
p.add_argument("--sigma", type=float, default=0.1); p.add_argument("--temp", type=float, default=0.3)
p.add_argument("--rollout-iters", type=int, default=None, help="projection iterations inside the rollouts (0 = unfiltered rollouts, plant filter kept)")
p.add_argument("--lib", default=None); p.add_argument("--tag", default="")
args = p.parse_args()
common.configure("gpu", x64=True)
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np, mujoco
from panda_problem import PandaProblem
from model import DT, NX, ND, NV, N_KNOT, PAR_DEFAULT, PROJ_ITERS, build_model_function, build_integrate_function, build_aux_function, project_greedy_np, knot_dt
from driver import SafeMppi

CTRL_DT = DT; REACH, TIMEOUT = 0.04, 6.0
WAYPOINTS = np.array([[0.55, 0.05, 0.55], [0.05, 0.55, 0.55], [0.55, 0.05, 0.30], [0.05, 0.55, 0.30]])
prob = PandaProblem("qpax"); robot = prob.robot
ee_j = jax.jit(lambda z: robot.ee_transform(z[:7])[:3, 3])
model_fn, _, _ = build_model_function(args.T); integ_fn, _, _ = build_integrate_function(); aux_fn = build_aux_function()
ours = SafeMppi("cusadi2", K=args.K, T=args.T, dtype="float", sigma=(args.sigma,) * 4, temp=args.temp, seed=args.seed, rollout_proj_iters=args.rollout_iters, lib=args.lib)
print(ours.info, flush=True)
umax = ours.umax; a = DT / knot_dt(args.T)

def unshift(y):                              # inverse of k_shift: y_j_new = (1 - a) y_j + a y_{j+1}, last knot held
    y = y.reshape(N_KNOT, NV).copy()
    for j in range(N_KNOT - 2, -1, -1): y[j] = (y[j] - a * y[j + 1]) / (1 - a)
    return y.reshape(-1)

def filter_stats(x, Y, rp):
    un, G, h = (np.asarray(o) for o in model_fn(x, Y, rp, PAR_DEFAULT)); un = un.ravel(); h = h.ravel()
    uc = np.clip(un, -umax, umax); r = (G @ uc - h) / np.linalg.norm(G, axis=1)
    up = project_greedy_np(un, G, h, umax, PROJ_ITERS)
    return uc, up, float(r.max()), int((r > 0).sum())

def unfiltered_rollout(x, Y, rp):
    hmin, first = np.inf, None; xk = x.copy()
    for s in range(args.T):
        un, G, h = (np.asarray(o) for o in model_fn(xk, Y, rp, PAR_DEFAULT)); un = un.ravel(); h = h.ravel()
        uc = np.clip(un, -umax, umax)
        if first is None and ((G @ uc - h) > 0).any(): first = s
        h2 = np.asarray(aux_fn(xk)[0]).ravel(); hmin = min(hmin, float(h2.min()))
        xk = np.asarray(integ_fn(xk, uc, rp, PAR_DEFAULT)).ravel()
    return hmin, first

fx = common.FRAX_DIR / "examples/xml"
rs = mujoco.MjSpec.from_file(str(fx / "panda.xml")); ss = mujoco.MjSpec.from_file(str(fx / "obstacle_scene.xml"))
ss.attach(rs, prefix="", frame=ss.worldbody.add_frame(pos=[0, 0, 0]))
mj = ss.compile(); data = mujoco.MjData(mj)
data.qpos[:7] = robot.default_configuration; mujoco.mj_forward(mj, data)
substeps = int(round(CTRL_DT / mj.opt.timestep))
q0 = np.asarray(robot.default_configuration)
x0 = np.zeros((1, NX)); x0[0, :7] = q0; ours.set_plant(x0); ours.set_plan(np.tile(q0, (1, 4)))

n_ticks = int(round(args.seconds / CTRL_DT)); wi, t_wp = 0, 0.0; events = []
L = dict(t=[], viol_mean=[], nviol_mean=[], move_mean=[], unorm=[], agree=[], viol_best=[], nviol_best=[], move_best=[], dist_best=[], unf_hmin=[], unf_first=[], ess=[], h2min=[], dist=[])
h2_j = jax.jit(prob.cfg.h_2)
for k in range(n_ticks):
    t = k * CTRL_DT
    z = np.concatenate([data.qpos[:7], data.qvel[:7]]); pee = np.asarray(ee_j(jnp.asarray(z))); wp = WAYPOINTS[wi % 4]
    dist = float(np.linalg.norm(pee - wp))
    if dist < REACH or t - t_wp > TIMEOUT:
        events.append(dict(wp=int(wi % 4), t_start=t_wp, t_end=t, reached=bool(dist < REACH))); wi += 1; t_wp = t; wp = WAYPOINTS[wi % 4]
    x = np.zeros((1, NX)); x[0, :14] = z
    u, stat, gm, wm = ours.tick(x, wp[None]); u = u[0]
    Ymean = unshift(ours.plan()[0]); Ys, xT = ours.samples(); w = ours.weights(); kb = int(np.argmax(w))
    uc, up, viol, nv = filter_stats(x[0], Ymean, wp)
    ucb, upb, violb, nvb = filter_stats(x[0], Ys[kb], wp)
    L["t"].append(t); L["viol_mean"].append(viol); L["nviol_mean"].append(nv); L["move_mean"].append(float(np.linalg.norm(u - uc))); L["unorm"].append(float(np.linalg.norm(uc)))
    L["agree"].append(float(np.abs(u - up).max())); L["viol_best"].append(violb); L["nviol_best"].append(nvb); L["move_best"].append(float(np.linalg.norm(upb - ucb)))
    L["dist_best"].append(float(np.linalg.norm(u - upb))); L["ess"].append(float(stat[0, 3]))
    if k % 5 == 0:
        hm, fs = unfiltered_rollout(x[0], Ymean, wp); L["unf_hmin"].append(hm); L["unf_first"].append(-1 if fs is None else fs)
    L["h2min"].append(float(np.asarray(h2_j(jnp.asarray(z))).min())); L["dist"].append(dist)
    data.ctrl[:7] = u
    for _ in range(substeps): mujoco.mj_step(mj, data)
    if k % 250 == 0:
        print(f"t={t:5.1f} reached {sum(e['reached'] for e in events)} | mean plan: re-filter active {np.mean(np.array(L['viol_mean']) > 0):.2f} of ticks, "
              f"median move {np.median(L['move_mean']):.3f} N m, max {np.max(L['move_mean']):.2f} | agree {np.max(L['agree']):.1e}", flush=True)
A = {k: np.asarray(v) for k, v in L.items()}
act = A["viol_mean"] > 0
summary = dict(seed=args.seed, K=args.K, T=args.T, seconds=args.seconds, n_ticks=n_ticks, waypoints_reached=sum(e["reached"] for e in events), waypoints_timed_out=sum(not e["reached"] for e in events),
               driver_vs_reference_max_abs=float(A["agree"].max()),
               mean_plan=dict(refilter_active_frac=float(act.mean()), violated_rows_mean_when_active=float(A["nviol_mean"][act].mean()) if act.any() else 0.0,
                              violated_rows_max=int(A["nviol_mean"].max()), signed_distance_max=float(A["viol_mean"].max()),
                              move_median_Nm=float(np.median(A["move_mean"])), move_median_when_active_Nm=float(np.median(A["move_mean"][act])) if act.any() else 0.0,
                              move_p90_Nm=float(np.percentile(A["move_mean"], 90)), move_max_Nm=float(A["move_mean"].max()),
                              move_rel_median_when_active=float(np.median(A["move_mean"][act] / np.maximum(A["unorm"][act], 1e-9))) if act.any() else 0.0,
                              unfiltered_rollout_hmin_min=float(A["unf_hmin"].min()), unfiltered_rollout_hmin_median=float(np.median(A["unf_hmin"])),
                              unfiltered_rollout_violates_frac=float(np.mean(A["unf_first"] >= 0)), unfiltered_first_violation_step_median=float(np.median(A["unf_first"][A["unf_first"] >= 0])) if (A["unf_first"] >= 0).any() else None),
               best_sample=dict(refilter_active_frac=float(np.mean(A["viol_best"] > 0)), violated_rows_max=int(A["nviol_best"].max()), move_median_Nm=float(np.median(A["move_best"])), move_max_Nm=float(A["move_best"].max()),
                                applied_minus_best_torque_median_Nm=float(np.median(A["dist_best"])), applied_minus_best_torque_max_Nm=float(A["dist_best"].max())),
               ess_mean=float(A["ess"].mean()), h_min_plant=float(A["h2min"].min()), rollout_proj_iters=args.rollout_iters, events=events)
(HERE / "results").mkdir(exist_ok=True)
common.save_json(HERE / "results" / f"refilter_study{args.tag}.json", summary)
np.savez(HERE / "results" / f"refilter_study{args.tag}.npz", **A)
print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=1))
