"""The comparison controller on the cupboard task: frax's published filtered operational-space controller (OSC nominal
torque + cbfpy's one-step CBF-QP through qpax), with the cupboard scene's 196 barriers (frax_model.CupboardOSCBFConfig),
on the same starts and the same pre-grasp target as closed_loop.py. Qualitative row: a one-step filter has no horizon,
so it is expected to stall at the opening.

    python closed_loop_osc.py [--starts 0 1 2 3 4] [--seconds 15] [--no-video]

Outputs results/osc/start{i}.{json,mp4} and summary.json (same fields as closed_loop.py's reach phase).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

p = argparse.ArgumentParser()
p.add_argument("--starts", type=int, nargs="+", default=[0, 1, 2, 3, 4]); p.add_argument("--seconds", type=float, default=15.0)
p.add_argument("--no-video", action="store_true"); p.add_argument("--fps", type=int, default=25); p.add_argument("--size", default="1280x720")
args = p.parse_args()
common.configure("gpu", x64=True)
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np, mujoco
from cbfpy import CBF
from frax_model import CupboardOSCBFConfig
from panda_problem import PandaProblem
from model import DT, NX, NV, build_aux_function
import scene

CTRL_DT = DT
LAT_TOL, AX_TOL, ROT_TOL = 0.01, 0.02, 0.1
out = HERE / "results" / "osc"; out.mkdir(parents=True, exist_ok=True)
prob = PandaProblem("qpax"); robot = prob.robot
cfg = CupboardOSCBFConfig(robot); cfg.backend = "qpax"; cbf = CBF.from_config(cfg)
ctrl_j = jax.jit(lambda z, zd: cbf.safety_filter(z, prob.nominal(z, zd)))
ee_j = jax.jit(lambda q: robot.ee_transform(q))
z_des = jnp.asarray(np.concatenate([scene.P_PREGRASP, scene.R_PREGRASP.reshape(9), np.zeros(6)]))
fa = build_aux_function()
groups = scene.group_slices()
starts = [np.asarray(s) for s in json.loads((HERE / "results" / "starts.json").read_text())["starts"]]


def barriers(q):
    x = np.zeros(NX); x[:NV] = q
    return np.asarray(fa(x)[0]).ravel()


def pose_error(q):
    T = np.asarray(ee_j(jnp.asarray(q))); P, R = T[:3, 3], T[:3, :3]
    e = P - scene.P_PREGRASP
    return P, float(np.linalg.norm(e[1:])), float(abs(e[0])), float(np.arccos(np.clip((np.trace(scene.R_PREGRASP.T @ R) - 1) / 2, -1, 1)))


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
dt = model.opt.timestep; substeps = int(round(CTRL_DT / dt))
if not args.no_video:
    renderer = mujoco.Renderer(model, height=H, width=W, max_geom=8000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [scene.X_OPEN + 0.12, 0.0, scene.Z_FLOOR + 0.05]; cam.distance = 1.6; cam.azimuth = -95; cam.elevation = -12
    from PIL import Image, ImageDraw, ImageFont
    try:
        fs_ = H / 720.0; font = ImageFont.truetype("DejaVuSans.ttf", int(17 * fs_)); font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", int(20 * fs_))
    except OSError:
        font = font_b = ImageFont.load_default(); fs_ = 1.0

results = []
for si in args.starts:
    q0 = starts[si]
    mujoco.mj_resetData(model, data); data.qpos[qadr] = q0; data.qpos[fadr] = 0.04; data.ctrl[act_grip] = 255; mujoco.mj_forward(model, data)
    ticks_per_frame = max(1, int(round(1.0 / (args.fps * CTRL_DT))))
    if not args.no_video:
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
                               "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out / f"start{si}.mp4")], stdin=subprocess.PIPE)
    log = dict(t=[], lat=[], ax=[], rot=[], x_tcp=[], h_min=[], h_argmin=[], ctrl_ms=[], v_max=[]); t_reach = None; n_ticks = int(round(args.seconds / CTRL_DT)); twall0 = time.perf_counter()
    for k in range(n_ticks):
        t = k * CTRL_DT; q, v = data.qpos[qadr].copy(), data.qvel[vadr].copy()
        P, lat, ax, ang = pose_error(q); hv = barriers(q)
        if lat <= LAT_TOL and ax <= AX_TOL and ang <= ROT_TOL:
            t_reach = t; break
        ta = time.perf_counter(); u = np.asarray(ctrl_j(jnp.asarray(np.concatenate([q, v])), z_des)); tb = time.perf_counter()
        data.ctrl[act_arm] = u
        for _ in range(substeps): mujoco.mj_step(model, data)
        log["t"].append(t); log["lat"].append(lat); log["ax"].append(ax); log["rot"].append(ang); log["x_tcp"].append(float(P[0]))
        log["h_min"].append(float(hv.min())); log["h_argmin"].append(int(hv.argmin())); log["ctrl_ms"].append((tb - ta) * 1e3); log["v_max"].append(float(np.abs(v).max()))
        if not args.no_video and k % ticks_per_frame == 0:
            renderer.update_scene(data, camera=cam); img = Image.fromarray(renderer.render()); d = ImageDraw.Draw(img, "RGBA")
            d.rectangle([10, 10, int(1000 * fs_), int(100 * fs_)], fill=(0, 0, 0, 150))
            d.text((20 * fs_, 16 * fs_), "Filtered OSC (frax demo controller: OSC + one-step CBF-QP, qpax), cupboard barriers, 50 Hz", font=font_b, fill="white")
            d.text((20 * fs_, 44 * fs_), f"start {si}   t = {t:5.2f} s   lateral {lat * 100:4.1f} cm  axial {ax * 100:4.1f} cm  rot {np.degrees(ang):4.1f} deg   min barrier {hv.min():6.3f}", font=font, fill=(220, 220, 220))
            ff.stdin.write(np.asarray(img).tobytes())
    if not args.no_video:
        ff.stdin.close(); ff.wait()
    hmin_i = int(np.argmin(log["h_min"])); row = log["h_argmin"][hmin_i]
    res = dict(start=si, reached=t_reach is not None, t_reach=t_reach, final=dict(lat=log["lat"][-1], ax=log["ax"][-1], rot=log["rot"][-1], x_tcp=log["x_tcp"][-1]),
               h_min=float(min(log["h_min"])), h_min_row=row, h_min_group=next(g for g, s in groups.items() if s.start <= row < s.stop),
               ctrl_ms_median=float(np.median(log["ctrl_ms"][5:])), v_max=float(max(log["v_max"])), ticks=len(log["t"]))
    results.append(res); (out / f"start{si}.json").write_text(json.dumps(res, indent=1)); np.savez(out / f"start{si}.npz", **{kk: np.asarray(vv) for kk, vv in log.items()})
    print(f"osc start {si}: {'REACHED at %.2f s' % t_reach if res['reached'] else 'NOT reached'} (final lat {res['final']['lat'] * 100:.1f} cm, ax {res['final']['ax'] * 100:.1f} cm, rot {np.degrees(res['final']['rot']):.1f} deg, "
          f"x_tcp {res['final']['x_tcp']:.3f}); h_min {res['h_min']:.4f} ({res['h_min_group']}); ctrl {res['ctrl_ms_median']:.2f} ms ({time.perf_counter() - twall0:.0f} s wall)", flush=True)
summary = dict(controller="osc", seconds=args.seconds, n_reached=sum(r["reached"] for r in results), n=len(results), h_min=float(min(r["h_min"] for r in results)), results=results)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(f"osc: reached {summary['n_reached']}/{summary['n']}, h_min {summary['h_min']:.4f}")
