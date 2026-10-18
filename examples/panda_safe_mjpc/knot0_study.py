"""Why does freezing knot 0 help? Two candidate mechanisms, measured on the tuned controller with and without knot-0 noise.

    python knot0_study.py run --variant frozen|noisy --seed 1 [--seconds 30]      -> results/knot0_study_<variant>_seed<s>.{json,npz}
    python knot0_study.py summarize                                               -> the comparison table

(A) torque continuity: noise on knot 0 enters the applied torque directly, so the torque jumps whenever a different
    sample wins. Measured: |u_t - u_{t-1}| between ticks, joint speed, barrier minima.
(B) selection bias: noise on knot 0 gives every candidate an immediate transient cost unrelated to its plan, and the
    cheapest transient is to pull knot 0 toward the current q (less PD torque, less velocity), so winners drift toward
    standing still. Measured on the last iteration of every tick: the mean cost of the candidates whose knot 0 moved
    toward q against those whose knot 0 moved away, and for the winner the change of |knot 0 - q| against the nominal.

Same plant, waypoints, pass rule and defaults as closed_loop.py (tuned: T 35, K 256, 3 iterations, sigma 0.10, kp 20,
w_pos 100); 'noisy' only sets the knot scale to all ones. No video.
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["run", "summarize"])
p.add_argument("--variant", default="frozen", choices=["frozen", "noisy"]); p.add_argument("--seed", type=int, default=1)
p.add_argument("--seconds", type=float, default=30.0); p.add_argument("--kp", type=float, default=None, help="override the tuned kp (20); files get the suffix _kp<kp>")
args = p.parse_args()
import numpy as np

CTRL_DT = 0.02; REACH, TIMEOUT = 0.04, 6.0
WAYPOINTS = np.array([[0.55, 0.05, 0.55], [0.05, 0.55, 0.55], [0.55, 0.05, 0.30], [0.05, 0.55, 0.30]])
NV = 7


def run():
    import common
    common.configure("gpu", x64=True); common.ensure_frax()
    import jax, jax.numpy as jnp, mujoco
    from panda_problem import PandaProblem
    from model import NX, N_KNOT, PARAMS, PAR_DEFAULT
    from driver import SafePS, TUNED
    prob = PandaProblem("qpax"); robot = prob.robot
    ee_j = jax.jit(lambda z: robot.ee_transform(z[:7])[:3, 3]); h2_j = jax.jit(prob.cfg.h_2)
    par = PAR_DEFAULT.copy()
    for k_, v_ in TUNED["par"].items(): par[[n for n, _ in PARAMS].index(k_)] = v_
    if args.kp is not None: par[[n for n, _ in PARAMS].index("kp")] = args.kp
    ks_ = list(TUNED["knot_scale"]) if args.variant == "frozen" else [1.0] * N_KNOT
    ours = SafePS("cusadi2", K=TUNED["K"], T=TUNED["T"], n_iter=TUNED["n_iter"], sigma=TUNED["sigma"], knot_scale=ks_, par=par, seed=args.seed, record=1)
    print(ours.info, "knot sigma", np.round(ours.sigma.reshape(N_KNOT, -1)[:, 0], 3).tolist(), flush=True)
    fx = common.FRAX_DIR / "examples/xml"
    rs = mujoco.MjSpec.from_file(str(fx / "panda.xml")); ss = mujoco.MjSpec.from_file(str(fx / "obstacle_scene.xml"))
    ss.attach(rs, prefix="", frame=ss.worldbody.add_frame(pos=[0, 0, 0])); mj = ss.compile(); data = mujoco.MjData(mj)
    data.qpos[:7] = robot.default_configuration; mujoco.mj_forward(mj, data)
    substeps = int(round(CTRL_DT / mj.opt.timestep)); q0 = np.asarray(robot.default_configuration)
    x0 = np.zeros((1, NX)); x0[0, :7] = q0; ours.set_plant(x0); ours.set_plan(np.tile(q0, (1, N_KNOT)))
    n_ticks = int(round(args.seconds / CTRL_DT)); wi, t_wp = 0, 0.0; events = []
    L = dict(t=[], dist=[], h_min=[], v_max=[], u=[], q=[], n_improved=[], ks=[], cost_start=[], cost_min=[],
             d_nom=[], d_win=[], cost_toward=[], cost_away=[], n_toward=[], frac_toward_among_better=[], frac_better=[])
    for k in range(n_ticks):
        t = k * CTRL_DT
        z = np.concatenate([data.qpos[:7], data.qvel[:7]]); pee = np.asarray(ee_j(jnp.asarray(z))); wp = WAYPOINTS[wi % 4]
        dist = float(np.linalg.norm(pee - wp))
        if dist < REACH or t - t_wp > TIMEOUT:
            events.append(dict(wp=int(wi % 4), t_start=t_wp, t_end=t, reached=bool(dist < REACH))); wi += 1; t_wp = t; wp = WAYPOINTS[wi % 4]; dist = float(np.linalg.norm(pee - wp))
        x = np.zeros((1, NX)); x[0, :14] = z
        u, stat, _, _ = ours.tick(x, wp[None]); u = u[0]
        its = ours.iter_stats()[:, 0, :]; ks = int(its[-1, 3])
        Y, xT = ours.samples(); c = xT[:, NX - 1]; q = z[:7]
        Y0 = Y[:, :NV]                                    # knot 0 of every candidate of the last iteration (row 0 = the nominal)
        dq = q - Y0[0]; dk = Y0 - Y0[0]                   # nominal's knot-0 gap to q; each candidate's knot-0 move
        toward = (dk @ dq) > 0; toward[0] = False; away = ~toward; away[0] = False
        fin = np.isfinite(c); better = fin & (c < c[0]); better[0] = False
        L["t"].append(t); L["dist"].append(dist); L["u"].append(u.copy()); L["q"].append(q.copy())
        L["h_min"].append(float(np.asarray(h2_j(jnp.asarray(z))).min())); L["v_max"].append(float(np.abs(z[7:]).max()))
        L["n_improved"].append(int((its[:, 3] != 0).sum())); L["ks"].append(ks); L["cost_start"].append(float(its[0, 4])); L["cost_min"].append(float(stat[0, 0]))
        L["d_nom"].append(float(np.linalg.norm(dq))); L["d_win"].append(float(np.linalg.norm(q - Y0[ks])))
        L["cost_toward"].append(float(c[toward & fin].mean()) if (toward & fin).any() else np.nan)
        L["cost_away"].append(float(c[away & fin].mean()) if (away & fin).any() else np.nan)
        L["n_toward"].append(int(toward.sum())); L["frac_better"].append(float(better.mean()))
        L["frac_toward_among_better"].append(float(toward[better].mean()) if better.any() else np.nan)
        data.ctrl[:7] = u
        for _ in range(substeps): mujoco.mj_step(mj, data)
        if k % 250 == 0: print(f"t={t:5.1f} reached {sum(e['reached'] for e in events)} timed out {sum(not e['reached'] for e in events)}", flush=True)
    A = {k: np.asarray(v) for k, v in L.items()}
    du = np.linalg.norm(np.diff(A["u"], axis=0), axis=1)
    S = dict(variant=args.variant + ("" if args.kp is None else f"_kp{args.kp:g}"), seed=args.seed, kp=float(par[[n for n, _ in PARAMS].index("kp")]), seconds=args.seconds, knot_scale=ks_,
             waypoints_reached=sum(e["reached"] for e in events), waypoints_timed_out=sum(not e["reached"] for e in events),
             time_to_reach_mean_s=float(np.mean([e["t_end"] - e["t_start"] for e in events if e["reached"]])) if any(e["reached"] for e in events) else None,
             h_min=float(A["h_min"].min()), v_max=float(A["v_max"].max()),
             du_median=float(np.median(du)), du_p90=float(np.percentile(du, 90)), du_p99=float(np.percentile(du, 99)), du_max=float(du.max()),
             plan_kept_frac=float(np.mean(A["n_improved"] == 0)), improved_iters_mean=float(A["n_improved"].mean()),
             frac_better_mean=float(A["frac_better"].mean()),
             cost_toward_minus_away_mean=float(np.nanmean(A["cost_toward"] - A["cost_away"])),
             cost_toward_lower_frac=float(np.nanmean(A["cost_toward"] < A["cost_away"])),
             frac_toward_among_better_mean=float(np.nanmean(A["frac_toward_among_better"])),
             d_win_minus_d_nom_mean_when_switched=float(np.mean((A["d_win"] - A["d_nom"])[A["ks"] != 0])) if (A["ks"] != 0).any() else None,
             d_nom_mean=float(A["d_nom"].mean()), events=events)
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(S, open(HERE / "results" / f"knot0_study_{S['variant']}_seed{args.seed}.json", "w"), indent=1)
    np.savez(HERE / "results" / f"knot0_study_{S['variant']}_seed{args.seed}.npz", **A)
    print(json.dumps({k: v for k, v in S.items() if k != "events"}, indent=1))


def summarize():
    import glob
    rows = {}
    for f in sorted(glob.glob(str(HERE / "results" / "knot0_study_*_seed*.json"))):
        d = json.load(open(f)); rows.setdefault(d["variant"], []).append(d)
    keys = [("waypoints_reached", "reached"), ("waypoints_timed_out", "timed out"), ("time_to_reach_mean_s", "s to reach"), ("h_min", "min h2"), ("v_max", "max |v|"),
            ("du_median", "|du| median N m"), ("du_p90", "|du| p90"), ("du_p99", "|du| p99"), ("du_max", "|du| max"),
            ("plan_kept_frac", "ticks with no improvement"), ("improved_iters_mean", "improving iterations / tick"), ("frac_better_mean", "candidates beating nominal"),
            ("cost_toward_minus_away_mean", "cost(toward q) - cost(away)"), ("cost_toward_lower_frac", "ticks where toward-q is cheaper"),
            ("frac_toward_among_better_mean", "toward-q share among better candidates"), ("d_win_minus_d_nom_mean_when_switched", "winner |k0 - q| - nominal |k0 - q| (rad)"),
            ("d_nom_mean", "nominal |k0 - q| (rad)")]
    vs = [v for v in sorted(rows, key=lambda v: (v.split("_kp")[-1] if "_kp" in v else "20", v))]
    print("| metric (mean over seeds) | " + " | ".join(f"{v} (n={len(rows[v])})" for v in vs) + " |"); print("|---|" + "---:|" * len(vs))
    for k, name in keys:
        vals = []
        for v in vs:
            xs = [r[k] for r in rows[v] if r.get(k) is not None]
            if k == "h_min": vals.append(f"{min(xs):.4f}" if xs else "-")
            elif k == "v_max": vals.append(f"{max(xs):.2f}" if xs else "-")
            else: vals.append(f"{np.mean(xs):.3f}" if xs else "-")
        print(f"| {name} | " + " | ".join(vals) + " |")
    for v in vs:
        print(v, "per seed reached/timed out:", [(r["seed"], r["waypoints_reached"], r["waypoints_timed_out"]) for r in rows[v]])


run() if args.mode == "run" else summarize()
