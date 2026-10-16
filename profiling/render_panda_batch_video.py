"""Headless video of the profiled GPU scenario: 4096 Pandas, one vmapped qpax fp64
safety-filter call per control tick on the GPU (the configuration where frax model
evaluation takes the largest share of the tick, 62 % at batch 4096).

    python profiling/render_panda_batch_video.py [--batch 4096] [--seconds 20] [--show 36]

Every instance is a full MuJoCo simulation of frax's demo scene (CPU, threaded) with
the demo OSC controller and barriers; the controller for all instances is one
jax.jit(jax.vmap(nominal + cbf.safety_filter)) call with cbfpy's qpax backend in fp64.
Each instance follows its own randomly perturbed, time-shifted version of the scripted
target path. Only `--show` instances are drawn (a grid); all `--batch` are simulated and
controlled. Output: results/panda_oscbf_gpu_b4096.mp4 (+ barrier/activity trace png).
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=4096)
p.add_argument("--show", type=int, default=36)
p.add_argument("--seconds", type=float, default=20.0)
p.add_argument("--fps", type=int, default=30)
p.add_argument("--size", default="1920x1080")
p.add_argument("--threads", type=int, default=12)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
args = p.parse_args()
out = Path(args.out) if args.out else common.RESULTS / f"panda_oscbf_gpu_b{args.batch}.mp4"

common.configure("gpu", x64=True)
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
from panda_problem import PandaProblem

W, H = (int(s) for s in args.size.split("x"))
B, K = args.batch, min(args.show, args.batch)
prob = PandaProblem("qpax")
robot = prob.robot
ctrl = jax.jit(jax.vmap(prob.end_to_end))     # the profiled end_to_end piece, batched
nominal = jax.jit(jax.vmap(prob.nominal))
h2 = jax.jit(jax.vmap(prob.h2_only))
zc = jnp.zeros((B, 14)); zdc = jnp.zeros((B, 18))
t0 = time.perf_counter(); jax.block_until_ready(ctrl(zc, zdc)); compile_s = time.perf_counter() - t0
jax.block_until_ready(nominal(zc, zdc)); jax.block_until_ready(h2(zc))
print(f"batch {B} on {jax.devices()[0]}: end_to_end compile+first call {compile_s:.1f} s")

# ---- physics: B copies of the demo scene (one shared model, B MjData) ----
fx = common.FRAX_DIR / "examples/xml"
def demo_model():
    rs = mujoco.MjSpec.from_file(str(fx / "panda.xml")); ss = mujoco.MjSpec.from_file(str(fx / "obstacle_scene.xml"))
    ss.attach(rs, prefix="", frame=ss.worldbody.add_frame(pos=[0, 0, 0])); return ss.compile()
model = demo_model()
mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")]
datas = [mujoco.MjData(model) for _ in range(B)]
for d in datas:
    d.qpos[:7] = robot.default_configuration; d.mocap_pos[mocap_id] = [0.5, 0, 0.5]; mujoco.mj_forward(model, d)
dt = model.opt.timestep
des_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
pool = ThreadPoolExecutor(args.threads)
chunks = [range(i, min(i + (B + args.threads - 1) // args.threads, B)) for i in range(0, B, (B + args.threads - 1) // args.threads)]
def step_chunk(rng, u, pos):
    for i in rng:
        d = datas[i]; d.mocap_pos[mocap_id] = pos[i]; d.ctrl[:7] = u[i]; mujoco.mj_step(model, d)
def gather():
    z = np.empty((B, 14))
    for i, d in enumerate(datas): z[i, :7] = d.qpos[:7]; z[i, 7:] = d.qvel[:7]
    return z

# ---- per-instance target paths: the scripted path, perturbed and time-shifted ----
WAY = [(0.0, [0.50, 0.00, 0.50]), (2.0, [0.50, 0.00, 0.50]), (5.0, [0.50, 0.50, 0.50]), (7.0, [0.50, 0.50, 0.50]),
       (10.0, [0.45, -0.25, -0.15]), (12.0, [0.45, -0.25, -0.15]), (14.5, [1.05, 0.10, 0.35]), (16.5, [1.05, 0.10, 0.35]),
       (18.5, [0.12, 0.05, 0.25]), (20.0, [0.12, 0.05, 0.25])]
PHASES = ["settle", "into obstacle", "at obstacle", "towards floor", "below floor", "out of reach", "far reach", "back to base", "near base", "hold"]
T_END = WAY[-1][0]
rng = np.random.default_rng(args.seed)
wp = np.array([w[1] for w in WAY])[None].repeat(B, 0) + rng.uniform(-0.12, 0.12, size=(B, len(WAY), 3))
wp[0] = np.array([w[1] for w in WAY])                                # instance 0 follows the unperturbed path
tshift = np.concatenate([[0.0], rng.uniform(0, T_END, size=B - 1)])  # instance 0 unshifted
wt = np.array([w[0] for w in WAY])
def targets(t):
    tt = (t + tshift) % T_END
    k = np.clip(np.searchsorted(wt, tt, side="right") - 1, 0, len(WAY) - 2)
    s = (tt - wt[k]) / (wt[k + 1] - wt[k]); s = s * s * (3 - 2 * s)
    p0 = wp[np.arange(B), k]; p1 = wp[np.arange(B), k + 1]
    return (1 - s)[:, None] * p0 + s[:, None] * p1

# ---- display scene: K robots in a grid, each with its obstacle and target ----
side = int(np.ceil(np.sqrt(K))); spacing = 1.7
disp = mujoco.MjSpec.from_file(str(fx / "scene.xml"))
# remove the single target from scene.xml; we add one per robot
for b in list(disp.worldbody.bodies):
    if b.name == "target": disp.delete(b)
disp.visual.global_.offwidth = W; disp.visual.global_.offheight = H
disp.visual.map.znear = 0.01
disp.visual.quality.shadowsize = 8192
origins = []
for i in range(K):
    ox, oy = (i % side - (side - 1) / 2) * spacing, (i // side - (side - 1) / 2) * spacing
    origins.append((ox, oy))
    rs = mujoco.MjSpec.from_file(str(fx / "panda.xml"))
    disp.attach(rs, prefix=f"r{i}_", frame=disp.worldbody.add_frame(pos=[ox, oy, 0]))
    ob = disp.worldbody.add_body(name=f"obs{i}", pos=[ox + 0.5, oy + 0.5, 0.5])
    ob.add_geom(name=f"obsg{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.3, 0, 0], rgba=[0.2, 0.6, 0.8, 1])
    tb = disp.worldbody.add_body(name=f"tgt{i}", mocap=True, pos=[ox + 0.5, oy, 0.5])
    tb.add_geom(name=f"tgtg{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0], rgba=[1, 0, 0, 0.6], contype=0, conaffinity=0)
dmodel = disp.compile(); ddata = mujoco.MjData(dmodel)
nq_r = model.nq
assert dmodel.nq == K * nq_r, (dmodel.nq, K, nq_r)
tgt_mocap = [dmodel.body_mocapid[mujoco.mj_name2id(dmodel, mujoco.mjtObj.mjOBJ_BODY, f"tgt{i}")] for i in range(K)]
tgt_geom = [mujoco.mj_name2id(dmodel, mujoco.mjtObj.mjOBJ_GEOM, f"tgtg{i}") for i in range(K)]
renderer = mujoco.Renderer(dmodel, height=H, width=W)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.3, 0.2, 0.2]; cam.distance = 1.55 * side * spacing; cam.azimuth = 150; cam.elevation = -32
try:
    fs = H / 1080.0
    font = ImageFont.truetype("DejaVuSans.ttf", int(24 * fs)); font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", int(28 * fs))
except OSError:
    font = font_b = ImageFont.load_default()
groups = []; i0 = 0
for name, n in prob.barrier_groups.items():
    groups.append((name, slice(i0, i0 + n))); i0 += n

ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
                       "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], stdin=subprocess.PIPE)
n_steps = int(round(args.seconds / dt)); steps_per_frame = max(1, int(round(1.0 / (args.fps * dt))))
log = {"t": [], "h_min": {g: [] for g, _ in groups}, "active_frac": [], "near_frac": [], "gpu_us": [], "sim_us": [], "tick_us": []}
twall0 = time.perf_counter(); n_frames = 0
for k in range(n_steps):
    t = k * dt
    t_a = time.perf_counter()
    pos = targets(t); z = gather()
    z_des = np.concatenate([pos, np.broadcast_to(des_rot.ravel(), (B, 9)), np.zeros((B, 6))], axis=1)
    zj, zdj = jax.device_put(z), jax.device_put(z_des); jax.block_until_ready((zj, zdj))
    t_b = time.perf_counter()
    uj = ctrl(zj, zdj); jax.block_until_ready(uj)
    t_c = time.perf_counter()
    u = np.asarray(uj)
    list(pool.map(lambda r: step_chunk(r, u, pos), chunks))
    t_d = time.perf_counter()
    log["gpu_us"].append((t_c - t_b) * 1e6); log["sim_us"].append((t_d - t_c) * 1e6); log["tick_us"].append((t_d - t_a) * 1e6)
    if k % steps_per_frame == 0:
        un = np.asarray(nominal(zj, zdj)); hv = np.asarray(h2(zj))
        du = np.linalg.norm(u - un, axis=1); act = du > 0.5
        log["t"].append(t); log["active_frac"].append(float(act.mean())); log["near_frac"].append(float((hv.min(1) < 0.01).mean()))
        for g, sl in groups: log["h_min"][g].append(float(hv[:, sl].min()))
        for i in range(K):
            ddata.qpos[i * nq_r:(i + 1) * nq_r] = datas[i].qpos
            ddata.mocap_pos[tgt_mocap[i]] = [origins[i][0] + pos[i, 0], origins[i][1] + pos[i, 1], pos[i, 2]]
            dmodel.geom_rgba[tgt_geom[i]] = [1, 0.55, 0, 0.9] if act[i] else [1, 0, 0, 0.6]
        mujoco.mj_forward(dmodel, ddata)
        renderer.update_scene(ddata, camera=cam)
        img = Image.fromarray(renderer.render()); d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([14, 14, int(1300 * fs), int(330 * fs)], fill=(0, 0, 0, 150))
        d.text((28 * fs, 22 * fs), f"Panda OSC + CBF safety filter, batch {B} on GPU  (qpax, fp64, one jit(vmap) call per tick)", font=font_b, fill="white")
        d.text((28 * fs, 62 * fs), f"t = {t:5.2f} s     drawing {K} of the {B} simulated instances; each follows its own perturbed, time-shifted copy of the demo target path", font=font, fill=(220, 220, 220))
        gm = np.median(log["gpu_us"][-steps_per_frame:]) / 1e3
        d.text((28 * fs, 98 * fs), f"GPU controller call: {gm:6.1f} ms per tick for all {B} instances  ({gm * 1e3 / B:4.1f} µs per instance)", font=font, fill=(255, 205, 120))
        d.text((28 * fs, 134 * fs), f"profiled split at this batch: frax model evaluation 62 %, QP solve 38 % of the tick", font=font, fill=(150, 200, 255))
        d.text((28 * fs, 170 * fs), f"filter changes the torque (|u_safe - u_nom| > 0.5 N·m): {act.sum():5d} / {B}   ({100 * act.mean():4.1f} %)   orange targets", font=font, fill=(255, 170, 90))
        d.text((28 * fs, 206 * fs), f"instances with a barrier within 0.01 of its boundary: {(hv.min(1) < 0.01).sum():5d} / {B}", font=font, fill=(220, 220, 220))
        y = 242
        for g, sl in groups:
            hm = log["h_min"][g][-1]
            d.text(((28 + 430 * (groups.index((g, sl)) % 3)) * fs, (y + 36 * (groups.index((g, sl)) // 3)) * fs), f"batch-min h {g} {hm:7.3f}", font=font,
                   fill=(255, 110, 90) if hm < 0.01 else (220, 220, 220))
        ff.stdin.write(np.asarray(img).tobytes()); n_frames += 1
        if n_frames % 30 == 0:
            el = time.perf_counter() - twall0
            print(f"t={t:5.2f}s frames={n_frames} wall={el:6.1f}s  gpu med {np.median(log['gpu_us']):.0f} µs  sim med {np.median(log['sim_us']):.0f} µs", flush=True)
ff.stdin.close(); ff.wait(); renderer.close(); pool.shutdown()
print(f"wrote {out}: {n_frames} frames, {n_steps} ticks, {time.perf_counter() - twall0:.0f} s wall")
print(f"GPU end_to_end call (B={B}): median {np.median(log['gpu_us']):.0f} µs, p99 {np.percentile(log['gpu_us'], 99):.0f} µs; "
      f"MuJoCo step of {B} instances: median {np.median(log['sim_us']):.0f} µs; full tick median {np.median(log['tick_us']):.0f} µs")
print("batch-min barrier over the run: " + ", ".join(f"{g} {min(v):.3f}" for g, v in log["h_min"].items()))
print(f"filter active fraction: mean {np.mean(log['active_frac']):.3f}, max {np.max(log['active_frac']):.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
for g, v in log["h_min"].items(): ax[0].plot(log["t"], v, lw=1.5, label=g)
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_ylabel("batch-min h₂ per group"); ax[0].legend(fontsize=8, ncol=5); ax[0].set_ylim(-0.05, 0.6)
ax[1].plot(log["t"], 100 * np.array(log["active_frac"]), color="#eb6834", label="filter active"); ax[1].plot(log["t"], 100 * np.array(log["near_frac"]), color="#2a78d6", label="barrier within 0.01")
ax[1].set_ylabel(f"% of {B} instances"); ax[1].legend(fontsize=8)
tt = np.arange(n_steps) * dt
ax[2].plot(tt, np.array(log["gpu_us"]) / 1e3, color="#2a78d6", lw=0.8, label="GPU end_to_end call"); ax[2].plot(tt, np.array(log["sim_us"]) / 1e3, color="#898781", lw=0.8, label="MuJoCo step (CPU threads)")
ax[2].set_ylabel("ms per tick"); ax[2].set_xlabel("time (s)"); ax[2].legend(fontsize=8); ax[2].set_ylim(0, None)
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); png = out.with_suffix(".png"); fig.savefig(png, dpi=140); print(png)
common.save_json(out.with_suffix(".json"), {"batch": B, "shown": K, "seconds": args.seconds, "dt": dt, "compile_s": compile_s,
    "gpu_us_median": float(np.median(log["gpu_us"])), "gpu_us_p99": float(np.percentile(log["gpu_us"], 99)),
    "sim_us_median": float(np.median(log["sim_us"])), "tick_us_median": float(np.median(log["tick_us"])),
    "active_frac_mean": float(np.mean(log["active_frac"])), "near_frac_mean": float(np.mean(log["near_frac"])),
    "h_min": {g: float(min(v)) for g, v in log["h_min"].items()}})
