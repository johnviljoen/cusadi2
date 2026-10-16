"""Headless MuJoCo video of the Panda OSC + CBF safety filter (the profiled scenario).

    python profiling/render_panda_video.py [--seconds 20] [--fps 30] [--backend elastiqp]

Same robot, scene, controller, barriers and QP as frax's examples/oscbf_demo.py
(obstacle sphere at [0.5, 0.5, 0.5] r = 0.3, joint limits, singularity, floor,
self-collision), run on CPU in fp64 like the demo. The only difference is that the
mocap target follows a scripted path instead of the mouse, so the arm is driven
into each barrier in turn. Output: results/panda_oscbf.mp4 (+ a barrier-trace png).
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

p = argparse.ArgumentParser()
p.add_argument("--seconds", type=float, default=20.0)
p.add_argument("--fps", type=int, default=30)
p.add_argument("--backend", default="elastiqp", choices=["elastiqp", "qpax"])
p.add_argument("--size", default="1280x720")
p.add_argument("--out", default=str(common.RESULTS / "panda_oscbf.mp4"))
args = p.parse_args()

common.configure("cpu", x64=True)
common.ensure_frax()
import jax, numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
from panda_problem import PandaProblem

W, H = (int(s) for s in args.size.split("x"))
prob = PandaProblem(args.backend)
robot = prob.robot
ctrl = jax.jit(prob.end_to_end)           # nominal OSC -> cbf.safety_filter, as in the demo
nominal = jax.jit(prob.nominal)
h2 = jax.jit(prob.h2_only)
z0, zd0 = np.zeros(14), np.zeros(18)
for f in (ctrl, nominal, h2):
    jax.block_until_ready(f(z0, zd0) if f is not h2 else f(z0))

# ---- MuJoCo scene, exactly as example_utils.ManipulatorEnv builds it (no viewer) ----
frax_dir = common.FRAX_DIR
robot_spec = mujoco.MjSpec.from_file(str(frax_dir / "examples/xml/panda.xml"))
scene_spec = mujoco.MjSpec.from_file(str(frax_dir / "examples/xml/obstacle_scene.xml"))
frame = scene_spec.worldbody.add_frame(pos=[0, 0, 0])
scene_spec.attach(robot_spec, prefix="", frame=frame)
scene_spec.visual.global_.offwidth = W   # offscreen framebuffer for the renderer
scene_spec.visual.global_.offheight = H
model = scene_spec.compile()
data = mujoco.MjData(model)
mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")]
data.qpos[:7] = robot.default_configuration
data.mocap_pos[mocap_id] = [0.5, 0, 0.5]
mujoco.mj_forward(model, data)
dt = model.opt.timestep
des_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # demo's fixed, facing-down orientation

# ---- scripted target path: (time, position, label) ----
WAY = [
    (0.0,  [0.50, 0.00, 0.50], "settle at the demo's start target"),
    (2.0,  [0.50, 0.00, 0.50], "target moves into the obstacle sphere"),
    (5.0,  [0.50, 0.50, 0.50], "target at the obstacle centre (unreachable)"),
    (7.0,  [0.50, 0.50, 0.50], "target moves below the floor"),
    (10.0, [0.45, -0.25, -0.15], "target below the floor (unreachable)"),
    (12.0, [0.45, -0.25, -0.15], "target moves out of reach"),
    (14.5, [1.05, 0.10, 0.35], "target out of reach (singularity / joint limits)"),
    (16.5, [1.05, 0.10, 0.35], "target moves back to the base"),
    (18.5, [0.12, 0.05, 0.25], "target close to the base (self-collision)"),
    (20.0, [0.12, 0.05, 0.25], "hold"),
]
def target(t):
    t = min(t, WAY[-1][0])
    for (t0, p0, lab), (t1, p1, _) in zip(WAY, WAY[1:]):
        if t <= t1:
            s = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            s = s * s * (3 - 2 * s)
            return (1 - s) * np.asarray(p0) + s * np.asarray(p1), lab
    return np.asarray(WAY[-1][1]), WAY[-1][2]

# ---- renderer + HUD ----
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.35, 0.12, 0.40]; cam.distance = 1.95; cam.azimuth = 150; cam.elevation = -16
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 20); font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
except OSError:
    font = font_b = ImageFont.load_default()
groups = []  # (name, slice) in cbfpy's h_2 order
i = 0
for name, n in prob.barrier_groups.items():
    groups.append((name, slice(i, i + n))); i += n

ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
     "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out],
    stdin=subprocess.PIPE)

n_steps = int(round(args.seconds / dt))
steps_per_frame = max(1, int(round(1.0 / (args.fps * dt))))
log = {"t": [], "h_min": {g: [] for g, _ in groups}, "du": [], "ctrl_us": []}
t_wall0 = time.perf_counter()
for k in range(n_steps):
    t = k * dt
    pos, label = target(t)
    data.mocap_pos[mocap_id] = pos
    z = np.concatenate([data.qpos[:7], data.qvel[:7]])
    z_des = np.concatenate([pos, des_rot.ravel(), np.zeros(6)])
    t0 = time.perf_counter()
    u = np.asarray(ctrl(z, z_des))
    log["ctrl_us"].append((time.perf_counter() - t0) * 1e6)
    data.ctrl[:7] = u
    mujoco.mj_step(model, data)
    if k % steps_per_frame == 0:
        u_nom = np.asarray(nominal(z, z_des)); hv = np.asarray(h2(z))
        du = float(np.linalg.norm(u - u_nom))
        log["t"].append(t); log["du"].append(du)
        for g, sl in groups:
            log["h_min"][g].append(float(hv[sl].min()))
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([10, 10, 470, 232], fill=(0, 0, 0, 140))
        d.text((20, 16), f"Panda OSC + CBF safety filter  ({args.backend}, CPU fp64)", font=font_b, fill="white")
        d.text((20, 46), f"t = {t:5.2f} s   {label}", font=font, fill="white")
        d.text((20, 76), f"|u_safe - u_nominal| = {du:6.1f} N·m   filter {'ACTIVE' if du > 0.5 else 'idle'}", font=font,
               fill=(255, 170, 90) if du > 0.5 else (190, 230, 190))
        y = 106
        for g, sl in groups:
            hm = log["h_min"][g][-1]
            col = (255, 110, 90) if hm < 0.05 else (220, 220, 220)
            d.text((20, y), f"min h  {g:<13s} {hm:8.3f}", font=font, fill=col); y += 24
        ff.stdin.write(np.asarray(img).tobytes())
ff.stdin.close(); ff.wait(); renderer.close()
wall = time.perf_counter() - t_wall0
print(f"wrote {args.out}: {len(log['t'])} frames, {n_steps} sim steps of {dt*1e3:.0f} ms, {wall:.1f} s wall")
print(f"controller call median {np.median(log['ctrl_us']):.0f} µs, p99 {np.percentile(log['ctrl_us'], 99):.0f} µs")
print(f"min barrier over the run: " + ", ".join(f"{g} {min(v):.3f}" for g, v in log["h_min"].items()))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
for g, v in log["h_min"].items():
    ax[0].plot(log["t"], v, lw=1.5, label=g)
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_ylabel("min barrier value h₂ per group"); ax[0].legend(fontsize=8, ncol=5, loc="upper right"); ax[0].set_ylim(-0.1, 1.5)
ax[1].plot(log["t"], log["du"], color="#eb6834", lw=1.5); ax[1].set_ylabel("|u_safe − u_nominal| (N·m)"); ax[1].set_xlabel("time (s)")
for a in ax: a.grid(alpha=0.3)
for t0, _, lab in WAY[1::2]: ax[0].axvline(t0, color="gray", lw=0.6, ls=":")
fig.tight_layout(); png = Path(args.out).with_name("panda_oscbf_barriers.png"); fig.savefig(png, dpi=140); print(png)
