"""Closed loop: one simulated Panda (MuJoCo, the cupboard scene) at 50 Hz reaching the pre-grasp pose at the cup, then
closing the gripper and holding.

    python closed_loop.py [--starts 0 1 2 3 4] [--rung 1] [--K 256] [--T 100] [--sigma 0.1 0.1 0.1 0.1] [--temp 0.3]
                          [--par w_fun=20,w_wrist=20] [--seconds 15] [--frax-timing] [--no-video] [--tag name]

The controller is the safe MPPI through the native driver (cusadi2 evaluator, applied). --frax-timing additionally ticks
the frax evaluator on the same plant states (not applied) so its tick time is logged next to ours.

Starts: results/starts.json (make_starts(), the frax default configuration plus 4 perturbed ones, all outside the
cupboard with every barrier > START_MARGIN). Per start: reach phase (until the pre-grasp test holds or REACH_TIMEOUT),
then the grasp phase (gripper command closed, MPPI keeps holding the pose, GRASP_HOLD seconds). The pre-grasp test:
lateral (y, z) TCP error <= LAT_TOL, axial (x) error <= AX_TOL, orientation error <= ROT_TOL. Grasp success: both finger
pads in contact with the cup for the last GRASP_CONTACT seconds, cup tilt < CUP_TILT, cup displacement < CUP_MOVE.
Logged per tick: barrier minimum (controller's 196 rows, CasADi on the CPU), the plant-side filter activity
(|u_applied - u_nom(x_plant, Y0)| > 1e-9), the sample-side activity of the last rollout step (fraction of samples with a
violated row before the projection), MPPI stats, tick times. Outputs results/<tag>/start{i}.{json,mp4} and summary.json.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

p = argparse.ArgumentParser()
p.add_argument("--starts", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--rung", type=int, default=1, help="ladder rung label for the output (the cost shaping itself is --par)")
p.add_argument("--K", type=int, default=256); p.add_argument("--T", type=int, default=100)
p.add_argument("--sigma", type=float, nargs=4, default=[0.1, 0.1, 0.1, 0.1]); p.add_argument("--temp", type=float, default=0.3)
p.add_argument("--par", default="", help="parameter overrides, e.g. w_fun=20,w_wrist=20,w_v=0.5")
p.add_argument("--proj-iters", type=int, default=None)
p.add_argument("--seconds", type=float, default=15.0, help="reach-phase timeout"); p.add_argument("--seed", type=int, default=1)
p.add_argument("--frax-timing", action="store_true"); p.add_argument("--no-video", action="store_true"); p.add_argument("--no-grasp", action="store_true")
p.add_argument("--fps", type=int, default=25); p.add_argument("--size", default="1280x720")
p.add_argument("--tag", default=None)
args = p.parse_args()
common.configure("gpu", x64=True)
common.ensure_frax()
import numpy as np, mujoco, casadi as ca
from model import DT, NX, NV, N_KNOT, PARAMS, PAR_DEFAULT, PROJ_ITERS, build_model_function, build_aux_function
from driver import SafeMppi
import scene

CTRL_DT = DT
LAT_TOL, AX_TOL, ROT_TOL = 0.01, 0.02, 0.1
GRASP_HOLD, GRASP_CONTACT, CUP_TILT, CUP_MOVE = 2.0, 1.0, np.radians(10), 0.02
START_MARGIN = 0.02
tag = args.tag or f"rung{args.rung}"
out = HERE / "results" / tag; out.mkdir(parents=True, exist_ok=True)
par = PAR_DEFAULT.copy()
for kv in filter(None, args.par.split(",")):
    k, v = kv.split("="); par[[n for n, _ in PARAMS].index(k)] = float(v)
PAR_NAMES = [n for n, _ in PARAMS]

# ---- the controller's model on the CPU, for the diagnostics ----
fm, robot, _ = build_model_function(args.T); fa = build_aux_function()
q_sym = ca.SX.sym("q", NV)
from backup_staged import _Model
_m = _Model(); _K = _m.kin(q_sym, ca.SX.zeros(NV))
f_pose = ca.Function("pose", [q_sym], [_K["pos"], _K["rot"]])
groups = scene.group_slices()
lo, hi = np.asarray(robot.actuated_joint_lower_limits), np.asarray(robot.actuated_joint_upper_limits)


def barriers(q):
    x = np.zeros(NX); x[:NV] = q
    return np.asarray(fa(x)[0]).ravel()


def pose_error(q):
    P, R = (np.asarray(v) for v in f_pose(q)); P = P.ravel()
    e = P - scene.P_PREGRASP
    ang = float(np.arccos(np.clip((np.trace(scene.R_PREGRASP.T @ R) - 1) / 2, -1, 1)))
    return P, float(np.linalg.norm(e[1:])), float(abs(e[0])), ang


def make_starts(path=HERE / "results" / "starts.json", n_pert=4, seed=0):
    """The frax default configuration plus n_pert random perturbations (0.3 rad rms), all with every barrier > START_MARGIN
    and the TCP in front of the cupboard (x_tcp < X_OPEN - 0.05)."""
    if path.exists():
        return [np.asarray(s) for s in json.loads(path.read_text())["starts"]]
    rng = np.random.default_rng(seed); q0 = np.asarray(robot.default_configuration); starts = []
    ok = lambda q: barriers(q).min() > START_MARGIN and pose_error(q)[0][0] < scene.X_OPEN - 0.05   # noqa: E731
    if ok(q0):
        starts.append(q0)
    else:
        print(f"default configuration rejected as a start: min barrier {barriers(q0).min():.4f}, x_tcp {pose_error(q0)[0][0]:.3f}", flush=True)
    while len(starts) < n_pert + 1:
        q = np.clip(q0 + rng.normal(0, 0.3, NV), lo + 0.05, hi - 0.05)
        if ok(q):
            starts.append(q)
    path.write_text(json.dumps(dict(starts=[s.tolist() for s in starts], margin=START_MARGIN, seed=seed), indent=1))
    return starts


starts = make_starts()
ours = SafeMppi("cusadi2", K=args.K, T=args.T, sigma=tuple(args.sigma), temp=args.temp, seed=args.seed, par=par, proj_iters=args.proj_iters or PROJ_ITERS)
print(ours.info, flush=True)
fr = SafeMppi("frax", K=args.K, T=args.T, sigma=tuple(args.sigma), temp=args.temp, seed=args.seed, par=par, proj_iters=args.proj_iters or PROJ_ITERS) if args.frax_timing else None

# ---- physics ----
fx = common.FRAX_DIR / "examples/xml"
rs = mujoco.MjSpec.from_file(str(fx / "panda.xml")); ss = mujoco.MjSpec.from_string(scene.mujoco_scene_xml())
ss.attach(rs, prefix="", frame=ss.worldbody.add_frame(pos=[0, 0, 0]))
if not args.no_video:
    W, H = map(int, args.size.split("x"))
    ss.visual.global_.offwidth = W; ss.visual.global_.offheight = H; ss.visual.map.znear = 0.01; ss.visual.quality.shadowsize = 8192
model = ss.compile(); data = mujoco.MjData(model)
qadr = np.array([model.joint(f"joint{i + 1}").qposadr[0] for i in range(NV)]); vadr = np.array([model.joint(f"joint{i + 1}").dofadr[0] for i in range(NV)])
fadr = [model.joint(n).qposadr[0] for n in ("finger_joint1", "finger_joint2")]
act_arm = [model.actuator(f"actuator{i + 1}").id for i in range(NV)]; act_grip = model.actuator("actuator8").id
cup_body = model.body("cup").id; cup_geom = model.geom("cup").id
pad_geoms = {"left": [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("left_finger").id],
             "right": [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("right_finger").id]}
dt = model.opt.timestep; substeps = int(round(CTRL_DT / dt)); assert abs(substeps * dt - CTRL_DT) < 1e-9


def reset(q):
    mujoco.mj_resetData(model, data)
    data.qpos[qadr] = q; data.qpos[fadr] = 0.04; data.ctrl[act_grip] = 255
    mujoco.mj_forward(model, data)
    return data.xpos[cup_body].copy(), data.xmat[cup_body].reshape(3, 3).copy()


def pad_contacts():
    hit = {"left": False, "right": False}
    for c in data.contact[:data.ncon]:
        for side, gs in pad_geoms.items():
            if (c.geom1 == cup_geom and c.geom2 in gs) or (c.geom2 == cup_geom and c.geom1 in gs):
                hit[side] = True
    return hit


# ---- video ----
if not args.no_video:
    renderer = mujoco.Renderer(model, height=H, width=W, max_geom=8000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [scene.X_OPEN + 0.12, 0.0, scene.Z_FLOOR + 0.05]; cam.distance = 1.6; cam.azimuth = -95; cam.elevation = -12
    from PIL import Image, ImageDraw, ImageFont
    try:
        fs_ = H / 720.0; font = ImageFont.truetype("DejaVuSans.ttf", int(17 * fs_)); font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", int(20 * fs_))
    except OSError:
        font = font_b = ImageFont.load_default(); fs_ = 1.0

    def add_trace(scn, pts, rgba, width=0.008):
        for a_, b_ in zip(pts[:-1], pts[1:]):
            if scn.ngeom >= scn.maxgeom: return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a_, b_); scn.ngeom += 1

    def ee_of_traj(tr):
        return np.stack([[np.asarray(f_pose(tr[t, k, :NV])[0]).ravel() for k in range(tr.shape[1])] for t in range(0, tr.shape[0], 5)])


# ---- one start ----
def run_start(si, q_start):
    cup_p0, cup_R0 = reset(q_start)
    x0 = np.zeros((1, NX)); x0[0, :NV] = q_start; ours.set_plant(x0); ours.set_plan(np.tile(q_start, (1, N_KNOT)))
    if fr is not None: fr.set_plant(x0); fr.set_plan(np.tile(q_start, (1, N_KNOT)))
    ticks_per_frame = max(1, int(round(1.0 / (args.fps * CTRL_DT))))
    if not args.no_video:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / f"start{si}.mp4")], stdin=subprocess.PIPE)
    log = dict(t=[], lat=[], ax=[], rot=[], x_tcp=[], h_min=[], h_argmin=[], plant_active=[], sample_active=[], sample_violated=[], ess=[], cmin=[], ours_ms=[], frax_ms=[], v_max=[], grip=[])
    phase, t_reach, t_grasp = "reach", None, None; contact_ticks = 0; trail = []; cup_tilt = cup_move = 0.0; n_frames = 0; twall0 = time.perf_counter()
    k = 0
    while True:
        t = k * CTRL_DT
        q, v = data.qpos[qadr].copy(), data.qvel[vadr].copy()
        P, lat, ax, ang = pose_error(q); hv = barriers(q)
        at_pose = lat <= LAT_TOL and ax <= AX_TOL and ang <= ROT_TOL
        if phase == "reach" and at_pose:
            phase, t_reach = "grasp", t
            if args.no_grasp: break
            data.ctrl[act_grip] = 0.0; t_grasp = t
        elif phase == "reach" and t >= args.seconds:
            break
        elif phase == "grasp" and t - t_grasp >= GRASP_HOLD:
            break
        x = np.zeros((1, NX)); x[0, :NV] = q; x[0, NV:2 * NV] = v
        u, stat, gm, wm = ours.tick(x, scene.P_PREGRASP[None]); u = u[0]
        if fr is not None:
            _, _, gf, _ = fr.tick(x, scene.P_PREGRASP[None]); log["frax_ms"].append(gf)
        # filter activity: plant side (the applied torque against the unfiltered PD torque of the mean plan) and sample side
        Y0 = ours.plan()[0]; xq = np.zeros(NX); xq[:NV] = q; xq[NV:2 * NV] = v
        u_nom = np.asarray(fm(xq, Y0, scene.P_PREGRASP, par)[0]).ravel()
        un_s, G_s, h_s, u_s = ours.filter_data()
        viol = (np.einsum("bij,bj->bi", G_s, un_s) - h_s > 0).any(1); viol_after = (np.einsum("bij,bj->bi", G_s, u_s) - h_s > 1e-6).any(1)
        data.ctrl[act_arm] = u
        for _ in range(substeps): mujoco.mj_step(model, data)
        if phase == "grasp":
            hit = pad_contacts(); contact_ticks = contact_ticks + 1 if hit["left"] and hit["right"] else 0
            Rc = data.xmat[cup_body].reshape(3, 3); cup_tilt = float(np.arccos(np.clip(Rc[2, 2], -1, 1))); cup_move = float(np.linalg.norm(data.xpos[cup_body] - cup_p0))
        log["t"].append(t); log["lat"].append(lat); log["ax"].append(ax); log["rot"].append(ang); log["x_tcp"].append(float(P[0]))
        log["h_min"].append(float(hv.min())); log["h_argmin"].append(int(hv.argmin())); log["plant_active"].append(bool(np.abs(u - u_nom).max() > 1e-9))
        log["sample_active"].append(float(viol.mean())); log["sample_violated"].append(float(viol_after.mean())); log["ess"].append(float(stat[0, 3])); log["cmin"].append(float(stat[0, 0]))
        log["ours_ms"].append(gm); log["v_max"].append(float(np.abs(v).max())); log["grip"].append(float(data.qpos[fadr[0]]))
        trail.append(P.copy()); trail = trail[-150:]
        if not args.no_video and k % ticks_per_frame == 0:
            renderer.update_scene(data, camera=cam)
            tr = ee_of_traj(ours.traj()); wts = ours.weights()[:tr.shape[1]]; wn = wts / max(wts.max(), 1e-12)
            for kk in range(1, tr.shape[1]):
                add_trace(renderer.scene, tr[:, kk], (0.3, 0.55 + 0.45 * wn[kk], 1.0, 0.3 + 0.7 * wn[kk]), width=0.0045)
            add_trace(renderer.scene, tr[:, 0], (1.0, 0.85, 0.1, 1.0), width=0.009)
            add_trace(renderer.scene, np.array(trail), (0.15, 0.9, 0.3, 0.9))
            img = Image.fromarray(renderer.render()); d = ImageDraw.Draw(img, "RGBA")
            d.rectangle([10, 10, int(1000 * fs_), int(176 * fs_)], fill=(0, 0, 0, 150))
            d.text((20 * fs_, 16 * fs_), f"Safe MPPI into a cupboard, K = {args.K} CBF-filtered rollouts x T = {args.T} steps ({args.T * CTRL_DT:.1f} s), 196 barriers, fp64, 50 Hz", font=font_b, fill="white")
            d.text((20 * fs_, 44 * fs_), f"start {si}   t = {t:5.2f} s   phase {phase}   lateral {lat * 100:4.1f} cm  axial {ax * 100:4.1f} cm  rot {np.degrees(ang):4.1f} deg", font=font, fill=(220, 220, 220))
            om = np.median(log["ours_ms"][-ticks_per_frame:])
            d.text((20 * fs_, 70 * fs_), f"cusadi2 evaluator (applied): {om:6.2f} ms per tick" + (f"     frax + cbfpy, same states (not applied): {np.median(log['frax_ms'][-ticks_per_frame:]):6.2f} ms" if log["frax_ms"] else ""), font=font, fill=(255, 205, 120))
            hm = log["h_min"][-1]
            d.text((20 * fs_, 96 * fs_), f"min barrier {hm:7.3f}   filter active: plant {'yes' if log['plant_active'][-1] else 'no '}, samples {100 * log['sample_active'][-1]:3.0f} %   ESS {log['ess'][-1]:5.1f}", font=font, fill=(255, 110, 90) if hm < 0 else (220, 220, 220))
            if phase == "grasp": d.text((20 * fs_, 122 * fs_), f"gripper closing: pad contact {contact_ticks * CTRL_DT:4.2f} s   cup tilt {np.degrees(cup_tilt):4.1f} deg   moved {cup_move * 100:4.1f} cm", font=font, fill=(180, 255, 180))
            d.text((20 * fs_, 150 * fs_), "yellow: mean plan through the CBF filter; blue: sampled plans (brighter = higher weight); green: TCP trail", font=font, fill=(200, 200, 200))
            ff.stdin.write(np.asarray(img).tobytes()); n_frames += 1
        k += 1
        if k % 250 == 0:
            print(f"  start {si} t={t:5.1f}s wall={time.perf_counter() - twall0:5.1f}s {phase} lat {lat * 100:4.1f} cm ax {ax * 100:4.1f} cm rot {np.degrees(ang):4.1f} deg x_tcp {P[0]:.3f} h_min {min(log['h_min']):.4f} ours {np.median(log['ours_ms']):.2f} ms", flush=True)
    if not args.no_video:
        ff.stdin.close(); ff.wait()
    grasp_ok = phase == "grasp" and not args.no_grasp and contact_ticks * CTRL_DT >= GRASP_CONTACT and cup_tilt < CUP_TILT and cup_move < CUP_MOVE
    hmin_i = int(np.argmin(log["h_min"])); row = log["h_argmin"][hmin_i]
    res = dict(start=si, q_start=list(map(float, q_start)), reached=t_reach is not None, t_reach=t_reach, grasp=bool(grasp_ok), contact_s=contact_ticks * CTRL_DT,
               cup_tilt_deg=float(np.degrees(cup_tilt)), cup_move=cup_move, final=dict(lat=log["lat"][-1], ax=log["ax"][-1], rot=log["rot"][-1], x_tcp=log["x_tcp"][-1]),
               h_min=float(min(log["h_min"])), h_min_row=row, h_min_group=next(g for g, s in groups.items() if s.start <= row < s.stop), h_min_t=log["t"][hmin_i],
               plant_active_frac=float(np.mean(log["plant_active"])), sample_active_mean=float(np.mean(log["sample_active"])), sample_violated_mean=float(np.mean(log["sample_violated"])),
               ess_mean=float(np.mean(log["ess"])), ours_ms_median=float(np.median(log["ours_ms"])), ours_ms_p99=float(np.percentile(log["ours_ms"], 99)),
               frax_ms_median=float(np.median(log["frax_ms"])) if log["frax_ms"] else None, v_max=float(max(log["v_max"])), ticks=k, frames=n_frames)
    (out / f"start{si}.json").write_text(json.dumps(res, indent=1)); np.savez(out / f"start{si}.npz", **{kk: np.asarray(vv) for kk, vv in log.items()})
    print(f"start {si}: {'REACHED at %.2f s' % t_reach if res['reached'] else 'NOT reached'} (final lat {res['final']['lat'] * 100:.1f} cm, ax {res['final']['ax'] * 100:.1f} cm, rot {np.degrees(res['final']['rot']):.1f} deg, x_tcp {res['final']['x_tcp']:.3f}); "
          f"grasp {'OK' if grasp_ok else 'no'} (contact {res['contact_s']:.2f} s, tilt {res['cup_tilt_deg']:.1f} deg, moved {cup_move * 100:.1f} cm); h_min {res['h_min']:.4f} ({res['h_min_group']} row {row} at {res['h_min_t']:.2f} s); "
          f"filter active plant {100 * res['plant_active_frac']:.0f} % samples {100 * res['sample_active_mean']:.0f} %; tick {res['ours_ms_median']:.2f} ms", flush=True)
    return res


results = [run_start(si, starts[si]) for si in args.starts]
summary = dict(tag=tag, rung=args.rung, K=args.K, T=args.T, sigma=args.sigma, temp=args.temp, par=dict(zip(PAR_NAMES, map(float, par))), proj_iters=args.proj_iters or PROJ_ITERS,
               seconds=args.seconds, n_reached=sum(r["reached"] for r in results), n_grasp=sum(r["grasp"] for r in results), n=len(results),
               h_min=float(min(r["h_min"] for r in results)), results=results)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(f"{tag}: reached {summary['n_reached']}/{summary['n']}, grasped {summary['n_grasp']}/{summary['n']}, h_min {summary['h_min']:.4f}")
ours.close()
if fr is not None: fr.close()
