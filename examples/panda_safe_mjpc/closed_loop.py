"""Closed loop: one simulated Panda (MuJoCo, frax's demo scene) at 50 Hz on a reach-around waypoint cycle, fp32.

    python closed_loop.py --controller ps  [--K 256] [--T 35] [--n-iter 3] [--sigma 0.10] [--knot-scale 0,1,1,1,1,1,1] [--par kp=20,w_pos=100] [--seconds 30] [--frax-timing] [--no-video]
    (the tuned configuration is the default; the MJPC-faithful phase-1 run was --K 64 --T 25 --n-iter 4 --sigma mjpc --knot-scale "" --par "")
    python closed_loop.py --controller osc [--seconds 30]

ps:   MJPC-faithful safe predictive sampling through the native driver (cusadi2 evaluator, applied): every 20 ms tick
      runs n_iter sequential iterations from the measured state, each the best of K CBF-filtered rollouts of a cubic
      7-knot joint-target spline (noise = 0.1 x half the joint range on every knot); the final nominal's first filtered
      torque is the torque sent, nothing is re-filtered; the plan is then resampled one period later. With --frax-timing
      the frax evaluator is ticked on the same plant states every tick with ONE iteration (not applied) so its
      per-iteration time is logged next to ours.
osc:  the comparison controller, frax's published filtered operational-space controller (OSC + cbfpy one-step CBF-QP,
      qpax), on the same waypoints.

Waypoints alternate across the obstacle so the straight line between consecutive ones passes through the sphere; a
waypoint counts as reached within REACH m, and is abandoned after TIMEOUT s. Outputs results/closed_loop_{controller}.{json,mp4}.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

p = argparse.ArgumentParser()
p.add_argument("--controller", default="ps", choices=["ps", "osc"])
p.add_argument("--K", type=int, default=256); p.add_argument("--T", type=int, default=35); p.add_argument("--n-iter", type=int, default=None, help="default driver.N_ITER")
p.add_argument("--seconds", type=float, default=30.0); p.add_argument("--seed", type=int, default=1)
p.add_argument("--frax-timing", action="store_true"); p.add_argument("--no-video", action="store_true")
p.add_argument("--fps", type=int, default=25); p.add_argument("--size", default="1280x720")
p.add_argument("--noise", type=float, default=0.1, help="MJPC sampling_exploration: knot noise std = noise x half the joint range")
p.add_argument("--sigma", default=0.10, help="absolute knot noise std in rad on every joint (overrides --noise; 'mjpc' = MJPC scaling)")
p.add_argument("--knot-scale", default="0,1,1,1,1,1,1", help="per-knot multipliers of the noise std; 0,1,1,1,1,1,1 freezes knot 0 ('' = all knots)")
p.add_argument("--par", default="kp=20,w_pos=100", help="model parameter overrides, e.g. kp=30,w_v=2 ('' = model defaults)"); p.add_argument("--tag", default="", help="suffix of the results files")
p.add_argument("--proj-iters", type=int, default=48, help="greedy projection iterations of the rollout filter")
p.add_argument("--no-nominal", action="store_true", help="sample 0 is a perturbation too (the shifted plan is not in the batch)")
p.add_argument("--select", default="cost", choices=["cost", "feasible"], help="feasible: candidates with a CBF violation along the rollout rank below every feasible one (infinite cost); if none is feasible the least violating wins")
p.add_argument("--filter-output-only", action="store_true", help="baseline: rollouts unfiltered (0 projection iterations), the CBF filter applied once to the planner's output torque at the plant state")
p.set_defaults(); args = p.parse_args()
PS = args.controller == "ps"
if isinstance(args.sigma, str): args.sigma = None if args.sigma == "mjpc" else float(args.sigma)
common.configure("gpu", x64=True)
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np, mujoco
from panda_problem import PandaProblem
from model import DT, NX, DES_ROT, N_KNOT, PARAMS, PAR_DEFAULT, PROJ_ITERS, build_model_function, project_greedy_np

CTRL_DT = DT
REACH, TIMEOUT = 0.04, 6.0
WAYPOINTS = np.array([[0.55, 0.05, 0.55], [0.05, 0.55, 0.55], [0.55, 0.05, 0.30], [0.05, 0.55, 0.30]])
out = HERE / "results" / f"closed_loop_{args.controller}{args.tag}"; out.parent.mkdir(exist_ok=True)

prob = PandaProblem("qpax"); robot = prob.robot
ee_j = jax.jit(lambda z: robot.ee_transform(z[:7])[:3, 3])
ee_traj_j = jax.jit(jax.vmap(jax.vmap(lambda z: robot.ee_transform(z[:7])[:3, 3])))
h2_j = jax.jit(prob.cfg.h_2)
groups = []; i0 = 0
for name, n in prob.barrier_groups.items():
    groups.append((name, slice(i0, i0 + n))); i0 += n

# ---- controllers ----
if PS:
    from driver import SafePS, N_ITER
    n_iter = N_ITER if args.n_iter is None else args.n_iter
    record = args.K if not args.no_video else 32            # the video needs the selected rollout, which can be any sample
    par = PAR_DEFAULT.copy()
    for kv in filter(None, args.par.split(",")):
        k_, v_ = kv.split("="); par[[n for n, _ in PARAMS].index(k_)] = float(v_)
    knot_scale = [float(v) for v in args.knot_scale.split(",")] if args.knot_scale else None
    kw = dict(K=args.K, T=args.T, noise=args.noise, sigma=args.sigma, knot_scale=knot_scale, par=par, seed=args.seed, keep_nominal=not args.no_nominal, proj_iters=0 if args.filter_output_only else args.proj_iters, select=args.select)
    model_fn = build_model_function(args.T)[0] if args.filter_output_only else None     # the output filter: model + projection at the plant state
    ours = SafePS("cusadi2", n_iter=n_iter, record=record, **kw)
    print(ours.info, "sigma", np.round(ours.sigma.reshape(N_KNOT, -1)[:, 0], 3).tolist(), "par", dict(zip([n for n, _ in PARAMS], par)), flush=True)
    fr = None
    if args.frax_timing:
        fr = SafePS("frax", n_iter=1, **kw)
        print(fr.info, flush=True)
else:
    osc_j = jax.jit(prob.end_to_end)
    des_rot9 = DES_ROT.reshape(9)

# ---- physics ----
fx = common.FRAX_DIR / "examples/xml"
rs = mujoco.MjSpec.from_file(str(fx / "panda.xml")); ss = mujoco.MjSpec.from_file(str(fx / "obstacle_scene.xml"))
ss.attach(rs, prefix="", frame=ss.worldbody.add_frame(pos=[0, 0, 0]))
if not args.no_video:
    W, H = map(int, args.size.split("x"))
    ss.visual.global_.offwidth = W; ss.visual.global_.offheight = H; ss.visual.map.znear = 0.01; ss.visual.quality.shadowsize = 8192
model = ss.compile(); data = mujoco.MjData(model)
mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")]
data.qpos[:7] = robot.default_configuration; mujoco.mj_forward(model, data)
dt = model.opt.timestep; substeps = int(round(CTRL_DT / dt)); assert abs(substeps * dt - CTRL_DT) < 1e-9
q0 = np.asarray(robot.default_configuration)
if PS:
    x0 = np.zeros((1, NX)); x0[0, :7] = q0; ours.set_plant(x0); ours.set_plan(np.tile(q0, (1, N_KNOT)))
    if fr is not None: fr.set_plant(x0); fr.set_plan(np.tile(q0, (1, N_KNOT)))

# ---- video ----
if not args.no_video:
    renderer = mujoco.Renderer(model, height=H, width=W, max_geom=8000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0.3, 0.45]; cam.distance = 1.45; cam.azimuth = 300; cam.elevation = -18
    from PIL import Image, ImageDraw, ImageFont
    try:
        fs = H / 720.0; font = ImageFont.truetype("DejaVuSans.ttf", int(17 * fs)); font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", int(20 * fs))
    except OSError:
        font = font_b = ImageFont.load_default(); fs = 1.0
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out.with_suffix(".mp4"))], stdin=subprocess.PIPE)
    def add_trace(scene, pts, rgba, width=0.008):
        for a_, b_ in zip(pts[:-1], pts[1:]):
            if scene.ngeom >= scene.maxgeom: return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a_, b_); scene.ngeom += 1

# ---- loop ----
n_ticks = int(round(args.seconds / CTRL_DT)); ticks_per_frame = max(1, int(round(1.0 / (args.fps * CTRL_DT))))
log = dict(t=[], wp=[], dist=[], h_min_all=[], h_min={g: [] for g, _ in groups}, ours_ms=[], frax_ms=[], sim_ms=[], ctrl_ms=[], u_norm=[], ks=[], cost_min=[], cost_mean=[], v_max=[], n_improved=[], cost_start=[], n_feasible=[], viol_sel=[])
events = []                     # (waypoint index, t_start, t_end, reached)
wi, t_wp = 0, 0.0; trail = []; n_frames = 0; twall0 = time.perf_counter()
for k in range(n_ticks):
    t = k * CTRL_DT
    z = np.concatenate([data.qpos[:7], data.qvel[:7]])
    pee = np.asarray(ee_j(jnp.asarray(z))); wp = WAYPOINTS[wi % len(WAYPOINTS)]
    dist = float(np.linalg.norm(pee - wp))
    if dist < REACH or t - t_wp > TIMEOUT:
        events.append(dict(wp=int(wi % len(WAYPOINTS)), t_start=t_wp, t_end=t, reached=bool(dist < REACH)))
        wi += 1; t_wp = t; wp = WAYPOINTS[wi % len(WAYPOINTS)]; dist = float(np.linalg.norm(pee - wp))
    data.mocap_pos[mocap_id] = wp
    t_a = time.perf_counter()
    if PS:
        x = np.zeros((1, NX)); x[0, :14] = z
        u, stat, gm, wm = ours.tick(x, wp[None]); u = u[0]
        if args.filter_output_only:                       # project the selected plan's torque once, at the plant state
            Ys, _ = ours.samples(); un_, G_, h_ = (np.asarray(o) for o in model_fn(x[0], Ys[int(stat[0, 3])], wp, par))
            u = project_greedy_np(un_.ravel(), G_, h_.ravel(), ours.umax, PROJ_ITERS)
        log["ours_ms"].append(gm); log["ks"].append(int(stat[0, 3])); log["cost_min"].append(float(stat[0, 0])); log["cost_mean"].append(float(stat[0, 1]))
        its = ours.iter_stats()[:, 0, :]                 # per iteration: cost min, mean, std, selected index, cost of the nominal
        log["n_improved"].append(int((its[:, 3] != 0).sum())); log["cost_start"].append(float(its[0, 4])); log["n_feasible"].append(int(its[-1, 5])); log["viol_sel"].append(float(its[-1, 6]))
        if fr is not None:
            _, _, gf, _ = fr.tick(x, wp[None]); log["frax_ms"].append(gf)
    else:
        zd = np.concatenate([wp, des_rot9, np.zeros(6)])
        u = np.asarray(osc_j(jnp.asarray(z), jnp.asarray(zd))); log["ks"].append(0); log["cost_min"].append(0.0); log["cost_mean"].append(0.0); log["n_improved"].append(0); log["cost_start"].append(0.0); log["n_feasible"].append(0); log["viol_sel"].append(0.0)
    t_b = time.perf_counter()
    hv = np.asarray(h2_j(jnp.asarray(z)))
    data.ctrl[:7] = u
    for _ in range(substeps): mujoco.mj_step(model, data)
    t_c = time.perf_counter()
    log["t"].append(t); log["wp"].append(int(wi % len(WAYPOINTS))); log["dist"].append(dist); log["ctrl_ms"].append((t_b - t_a) * 1e3); log["sim_ms"].append((t_c - t_b) * 1e3)
    log["h_min_all"].append(float(hv.min())); log["u_norm"].append(float(np.linalg.norm(u))); log["v_max"].append(float(np.abs(z[7:]).max()))
    for g, sl in groups: log["h_min"][g].append(float(hv[sl].min()))
    trail.append(pee.copy()); trail = trail[-100:]
    if not args.no_video and k % ticks_per_frame == 0:
        renderer.update_scene(data, camera=cam)
        if PS:                                              # the last iteration's sampled rollouts: end-effector paths, the final nominal in yellow
            tr = np.asarray(ee_traj_j(jnp.asarray(ours.traj())))          # (T+1, n_rec, 3), n_rec = K
            ks = log["ks"][-1]; c = ours.costs()[:tr.shape[1]]
            bright = np.clip(1.0 - (c - stat[0, 0]) / max(2.0 * float(stat[0, 2]), 1e-9), 0.0, 1.0)   # 1 at the minimum, 0 at min + 2 std
            others = [kk for kk in range(min(32, tr.shape[1])) if kk != ks][:31]
            for kk in others:
                add_trace(renderer.scene, tr[:, kk], (0.3, 0.55 + 0.45 * bright[kk], 1.0, 0.3 + 0.7 * bright[kk]), width=0.0045)
            add_trace(renderer.scene, tr[:, ks], (1.0, 0.85, 0.1, 1.0), width=0.009)                # the selected rollout = the plan
        add_trace(renderer.scene, np.array(trail), (0.15, 0.9, 0.3, 0.9))
        img = Image.fromarray(renderer.render()); d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([10, 10, int(1000 * fs), int(176 * fs)], fill=(0, 0, 0, 150))
        title = (f"Safe predictive sampling, MJPC style: {n_iter} x best of {args.K} CBF-filtered rollouts, {args.T * CTRL_DT:.1f} s horizon, 50 Hz, no re-filter" if PS
                 else "Filtered OSC (frax demo controller: OSC + one-step CBF-QP), 50 Hz")
        d.text((20 * fs, 16 * fs), title, font=font_b, fill="white")
        if PS: d.text((20 * fs, 150 * fs), f"yellow: the selected rollout (its first filtered torque is applied); blue: {len(others)} of the last iteration's {args.K} samples, brighter = lower cost; green: trail", font=font, fill=(200, 200, 200))
        d.text((20 * fs, 44 * fs), f"t = {t:5.2f} s   waypoint {wi % len(WAYPOINTS) + 1} of {len(WAYPOINTS)} (cycle)   distance {dist:5.3f} m   reached so far {sum(e['reached'] for e in events)}, timed out {sum(not e['reached'] for e in events)}", font=font, fill=(220, 220, 220))
        if PS:
            om = np.median(log["ours_ms"][-ticks_per_frame:]); ni = np.mean(log["n_improved"][-ticks_per_frame:])
            d.text((20 * fs, 70 * fs), f"cusadi2 evaluator (applied): {om:6.2f} ms per tick = {n_iter} x {om / n_iter:5.2f} ms per iteration (budget 20 ms)   plan cost {log['cost_start'][-1]:7.3f} -> {stat[0, 0]:7.3f}, {ni:3.1f} of {n_iter} iterations improved it", font=font, fill=(255, 205, 120))
            if log["frax_ms"]:
                fm = np.median(log["frax_ms"][-ticks_per_frame:])
                d.text((20 * fs, 96 * fs), f"frax + cbfpy evaluator, same driver, same states (not applied): {fm:6.2f} ms per iteration = {fm * n_iter:6.1f} ms for {n_iter}   {fm * n_iter / 20:4.1f}x over budget", font=font, fill=(150, 200, 255))
        hm = log["h_min_all"][-1]
        d.text((20 * fs, 122 * fs), f"min barrier h2 {hm:7.3f}   |v| max {log['v_max'][-1]:4.2f} rad/s   |u| {log['u_norm'][-1]:5.1f} N m", font=font, fill=(255, 110, 90) if hm < 0.0 else (220, 220, 220))
        ff.stdin.write(np.asarray(img).tobytes()); n_frames += 1
    if k % 250 == 0:
        print(f"t={t:5.1f}s wall={time.perf_counter() - twall0:5.1f}s wp {wi % len(WAYPOINTS)} dist {dist:.3f} reached {sum(e['reached'] for e in events)} "
              + (f"ours {np.median(log['ours_ms']):.2f} ms" if log["ours_ms"] else f"ctrl {np.median(log['ctrl_ms']):.2f} ms") + (f" frax {np.median(log['frax_ms']):.2f} ms" if log["frax_ms"] else "")
              + f" h_min {min(log['h_min_all']):.4f}", flush=True)
if not args.no_video:
    ff.stdin.close(); ff.wait(); renderer.close()
reached = [e for e in events if e["reached"]]
summary = dict(controller=args.controller, filter_output_only=bool(args.filter_output_only), select=args.select,
               feasible_candidates_mean=float(np.mean(log["n_feasible"])) if PS else None, ticks_without_feasible_frac=float(np.mean(np.asarray(log["n_feasible"]) == 0)) if PS else None, K=args.K if PS else None, T=args.T if PS else None, n_iter=n_iter if PS else None, seconds=args.seconds, dtype="float", seed=args.seed, noise=args.noise,
               sigma=ours.sigma.reshape(N_KNOT, -1)[:, 0].tolist() if PS else None, knot_scale=args.knot_scale, par=args.par, proj_iters=args.proj_iters, keep_nominal=not args.no_nominal,
               waypoints_reached=len(reached), waypoints_timed_out=len(events) - len(reached),
               time_to_reach_mean_s=float(np.mean([e["t_end"] - e["t_start"] for e in reached])) if reached else None,
               time_to_reach_max_s=float(np.max([e["t_end"] - e["t_start"] for e in reached])) if reached else None,
               h_min_all=float(min(log["h_min_all"])), h_min={g: float(min(v)) for g, v in log["h_min"].items()},
               ours_ms_median=float(np.median(log["ours_ms"])) if log["ours_ms"] else None, ours_ms_p99=float(np.percentile(log["ours_ms"], 99)) if log["ours_ms"] else None,
               ours_ms_max=float(np.max(log["ours_ms"])) if log["ours_ms"] else None,
               frax_ms_median=float(np.median(log["frax_ms"])) if log["frax_ms"] else None, frax_ms_p99=float(np.percentile(log["frax_ms"], 99)) if log["frax_ms"] else None,
               ctrl_wall_ms_median=float(np.median(log["ctrl_ms"])), sim_ms_median=float(np.median(log["sim_ms"])), v_max=float(max(log["v_max"])),
               plan_kept_frac=float(np.mean(np.asarray(log["n_improved"]) == 0)) if PS else None, improved_iterations_mean=float(np.mean(log["n_improved"])) if PS else None,
               cost_min_mean=float(np.mean(log["cost_min"])) if PS else None, cost_start_mean=float(np.mean(log["cost_start"])) if PS else None,
               events=events, n_ticks=n_ticks, frames=n_frames)
common.save_json(out.with_suffix(".json"), summary)
np.savez(out.with_suffix(".npz"), **{k: np.asarray(v) for k, v in log.items() if k != "h_min"}, **{f"h_min_{g}": np.asarray(v) for g, v in log["h_min"].items()})
print({k: v for k, v in summary.items() if k not in ("events", "h_min")}); print("h_min per group:", summary["h_min"])
