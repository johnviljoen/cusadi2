"""Closed-loop run of the backup-CBF safety filter on 256 simulated Pandas at 50 Hz, with video.

    python closed_loop.py [--batch 256] [--N 20] [--seconds 20] [--show 36] [--no-video]

Every instance is a MuJoCo simulation of frax's demo scene (CPU, threaded); the controller for all
instances is one jitted call per 20 ms control tick: cusadi2 staged model -> CUDA interior-point
solver (the pipeline that fits the period), with a zero-order hold over the 2 ms physics substeps.
At every tick the frax + qpax pipeline is also evaluated on the same states (not applied) so its
tick time appears in the HUD, and the one-step (N = 0) filter is solved on the same states so the
fraction of instances whose torque the predictive rows changed can be reported.

Outputs results/closed_loop_b{B}_N{N}.{mp4,png,json}: video (36 of 256 instances drawn, each with
its obstacle and target; the predicted braking path of every drawn instance is drawn as a capsule
trace, but under the brake the end effector moves only a few centimetres in 0.4 s so it is barely
visible at this camera distance), barrier minima per group over the run, filter activity, tick times.
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=256)
p.add_argument("--N", type=int, default=20)
p.add_argument("--show", type=int, default=36)
p.add_argument("--seconds", type=float, default=20.0)
p.add_argument("--fps", type=int, default=25)
p.add_argument("--size", default="1920x1080")
p.add_argument("--threads", type=int, default=12)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no-video", action="store_true")
p.add_argument("--no-frax", action="store_true", help="skip the frax + qpax reference evaluation per tick")
args = p.parse_args()
out = HERE / "results" / f"closed_loop_b{args.batch}_N{args.N}.mp4"

common.configure("gpu", x64=True)
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np, mujoco
from backup_jax import BackupProblem
from cusadi2_staged import Cusadi2Staged
from ipm import CudaIpm

W, H = (int(s) for s in args.size.split("x"))
B, K, N = args.batch, min(args.show, args.batch), args.N
CTRL_DT = 0.02
prob = BackupProblem(N, loop="unrolled", deriv="jvp")
robot = prob.robot
cus = Cusadi2Staged(N, max_batch=B)
ipm = CudaIpm(prob.n_rows, max_batch=B, penalty=prob.penalty)
prob0 = BackupProblem(0)
ipm0 = CudaIpm(prob0.n_rows, max_batch=B, penalty=prob0.penalty)
pen = jnp.asarray(prob.penalty)

def ours(z, zd):
    u, P, q, G, h = cus.model_part(z, zd)
    x, it, conv = ipm.solve(P, q, G, h)
    slack = h - jnp.einsum("bij,bj->bi", G, x)            # >= 0 at a feasible solution; ~0 on active rows
    return x, u, it, conv, slack
def onestep(z, zd):                                        # the same QP without the predictive rows (frax model, N = 0)
    u, P, q, G, h = jax.vmap(prob0.model_part)(z, zd)
    return ipm0.solve(P, q, G, h)[0]
def frax_qpax(z, zd):
    return jax.vmap(prob.end_to_end)(z, zd)
ours_j = jax.jit(ours); onestep_j = jax.jit(onestep); frax_j = jax.jit(frax_qpax)
h2_j = jax.jit(jax.vmap(prob.cfg.h_2))
roll_j = jax.jit(jax.vmap(prob.rollout_states))
ee_j = jax.jit(jax.vmap(jax.vmap(lambda z: robot.ee_transform(z[:7])[:3, 3])))
zc = jnp.zeros((B, 14)); zdc = jnp.zeros((B, 18))
t0 = time.perf_counter(); jax.block_until_ready(ours_j(zc, zdc)); compile_ours = time.perf_counter() - t0
jax.block_until_ready(onestep_j(zc, zdc)); jax.block_until_ready(h2_j(zc)); jax.block_until_ready(ee_j(roll_j(zc[:K])))
if not args.no_frax:
    t0 = time.perf_counter(); jax.block_until_ready(frax_j(zc, zdc)); compile_frax = time.perf_counter() - t0
else:
    compile_frax = None
print(f"batch {B}, N {N}, {prob.n_rows} rows: ours compile {compile_ours:.1f} s" + (f", frax+qpax compile {compile_frax:.1f} s" if compile_frax else ""), flush=True)

# ---- physics ----
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
substeps = int(round(CTRL_DT / dt)); assert abs(substeps * dt - CTRL_DT) < 1e-9, (dt, CTRL_DT)
des_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
pool = ThreadPoolExecutor(args.threads)
csz = (B + args.threads - 1) // args.threads
chunks = [range(i, min(i + csz, B)) for i in range(0, B, csz)]
def step_chunk(rng, u, pos):
    for i in rng:
        d = datas[i]; d.mocap_pos[mocap_id] = pos[i]; d.ctrl[:7] = u[i]
        for _ in range(substeps): mujoco.mj_step(model, d)
def gather():
    z = np.empty((B, 14))
    for i, d in enumerate(datas): z[i, :7] = d.qpos[:7]; z[i, 7:] = d.qvel[:7]
    return z

# ---- target paths (as in the profiling video) ----
WAY = [(0.0, [0.50, 0.00, 0.50]), (2.0, [0.50, 0.00, 0.50]), (5.0, [0.50, 0.50, 0.50]), (7.0, [0.50, 0.50, 0.50]),
       (10.0, [0.45, -0.25, -0.15]), (12.0, [0.45, -0.25, -0.15]), (14.5, [1.05, 0.10, 0.35]), (16.5, [1.05, 0.10, 0.35]),
       (18.5, [0.12, 0.05, 0.25]), (20.0, [0.12, 0.05, 0.25])]
T_END = WAY[-1][0]
rng = np.random.default_rng(args.seed)
wp = np.array([w[1] for w in WAY])[None].repeat(B, 0) + rng.uniform(-0.12, 0.12, size=(B, len(WAY), 3))
wp[0] = np.array([w[1] for w in WAY]); tshift = np.concatenate([[0.0], rng.uniform(0, T_END, size=B - 1)])
wt = np.array([w[0] for w in WAY])
def targets(t):
    tt = (t + tshift) % T_END
    k = np.clip(np.searchsorted(wt, tt, side="right") - 1, 0, len(WAY) - 2)
    s = (tt - wt[k]) / (wt[k + 1] - wt[k]); s = s * s * (3 - 2 * s)
    return (1 - s)[:, None] * wp[np.arange(B), k] + s[:, None] * wp[np.arange(B), k + 1]

# ---- display scene ----
if not args.no_video:
    side = int(np.ceil(np.sqrt(K))); spacing = 1.7
    disp = mujoco.MjSpec.from_file(str(fx / "scene.xml"))
    for b in list(disp.worldbody.bodies):
        if b.name == "target": disp.delete(b)
    disp.visual.global_.offwidth = W; disp.visual.global_.offheight = H; disp.visual.map.znear = 0.01; disp.visual.quality.shadowsize = 8192
    origins = []
    for i in range(K):
        ox, oy = (i % side - (side - 1) / 2) * spacing, (i // side - (side - 1) / 2) * spacing; origins.append((ox, oy))
        disp.attach(mujoco.MjSpec.from_file(str(fx / "panda.xml")), prefix=f"r{i}_", frame=disp.worldbody.add_frame(pos=[ox, oy, 0]))
        ob = disp.worldbody.add_body(name=f"obs{i}", pos=[ox + 0.5, oy + 0.5, 0.5])
        ob.add_geom(name=f"obsg{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.3, 0, 0], rgba=[0.2, 0.6, 0.8, 1])
        tb = disp.worldbody.add_body(name=f"tgt{i}", mocap=True, pos=[ox + 0.5, oy, 0.5])
        tb.add_geom(name=f"tgtg{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0], rgba=[1, 0, 0, 0.6], contype=0, conaffinity=0)
    dmodel = disp.compile(); ddata = mujoco.MjData(dmodel); nq_r = model.nq
    assert dmodel.nq == K * nq_r
    tgt_mocap = [dmodel.body_mocapid[mujoco.mj_name2id(dmodel, mujoco.mjtObj.mjOBJ_BODY, f"tgt{i}")] for i in range(K)]
    tgt_geom = [mujoco.mj_name2id(dmodel, mujoco.mjtObj.mjOBJ_GEOM, f"tgtg{i}") for i in range(K)]
    renderer = mujoco.Renderer(dmodel, height=H, width=W, max_geom=20000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.3, 0.2, 0.2]; cam.distance = 1.55 * side * spacing; cam.azimuth = 150; cam.elevation = -32
    from PIL import Image, ImageDraw, ImageFont
    try:
        fs = H / 1080.0
        font = ImageFont.truetype("DejaVuSans.ttf", int(24 * fs)); font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", int(28 * fs))
    except OSError:
        font = font_b = ImageFont.load_default(); fs = 1.0
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(args.fps),
                           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], stdin=subprocess.PIPE)
    def add_trace(scene, pts, rgba):
        for a, b in zip(pts[:-1], pts[1:]):
            if scene.ngeom >= scene.maxgeom: return
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, a, b)
            scene.ngeom += 1

groups = []; i0 = 0
for name, n in prob.base.barrier_groups.items():
    groups.append((name, slice(i0, i0 + n))); i0 += n
n_ticks = int(round(args.seconds / CTRL_DT)); ticks_per_frame = max(1, int(round(1.0 / (args.fps * CTRL_DT))))
log = {"t": [], "h_min": {g: [] for g, _ in groups}, "active_frac": [], "pred_frac": [], "pred_rows_frac": [], "near_frac": [],
       "ours_us": [], "frax_us": [], "sim_us": [], "iters_max": [], "conv_frac": [], "h_min_all": []}
twall0 = time.perf_counter(); n_frames = 0
for k in range(n_ticks):
    t = k * CTRL_DT
    pos = targets(t); z = gather()
    z_des = np.concatenate([pos, np.broadcast_to(des_rot.ravel(), (B, 9)), np.zeros((B, 6))], axis=1)
    zj, zdj = jax.device_put(z), jax.device_put(z_des); jax.block_until_ready((zj, zdj))
    t_b = time.perf_counter(); res = ours_j(zj, zdj); jax.block_until_ready(res); t_c = time.perf_counter()
    u, un, it, conv, slack = [np.asarray(a) for a in res]
    if not args.no_frax:
        t_f0 = time.perf_counter(); uf = frax_j(zj, zdj); jax.block_until_ready(uf); log["frax_us"].append((time.perf_counter() - t_f0) * 1e6)
    u1 = np.asarray(onestep_j(zj, zdj))
    hv = np.asarray(h2_j(zj))
    t_d = time.perf_counter(); list(pool.map(lambda r: step_chunk(r, u, pos), chunks)); t_e = time.perf_counter()
    log["ours_us"].append((t_c - t_b) * 1e6); log["sim_us"].append((t_e - t_d) * 1e6)
    act = np.linalg.norm(u - un, axis=1) > 0.5; pred = np.linalg.norm(u - u1, axis=1) > 0.5
    pred_rows = slack[:, 64:64 + 64 * N] < 1e-6
    log["t"].append(t); log["active_frac"].append(float(act.mean())); log["pred_frac"].append(float(pred.mean()))
    log["pred_rows_frac"].append(float(pred_rows.any(1).mean())); log["near_frac"].append(float((hv.min(1) < 0.01).mean()))
    log["iters_max"].append(int(it.max())); log["conv_frac"].append(float(conv.mean())); log["h_min_all"].append(float(hv.min()))
    for g, sl in groups: log["h_min"][g].append(float(hv[:, sl].min()))
    if not args.no_video and k % ticks_per_frame == 0:
        tr = np.asarray(ee_j(roll_j(zj[:K])))                      # (K, N+1, 3) predicted end-effector path
        for i in range(K):
            ddata.qpos[i * nq_r:(i + 1) * nq_r] = datas[i].qpos
            ddata.mocap_pos[tgt_mocap[i]] = [origins[i][0] + pos[i, 0], origins[i][1] + pos[i, 1], pos[i, 2]]
            dmodel.geom_rgba[tgt_geom[i]] = [1, 0.55, 0, 0.9] if act[i] else [1, 0, 0, 0.6]
        mujoco.mj_forward(dmodel, ddata)
        renderer.update_scene(ddata, camera=cam)
        for i in range(K):
            pts = tr[i] + np.array([origins[i][0], origins[i][1], 0.0])
            add_trace(renderer.scene, pts, (1.0, 0.8, 0.1, 1.0) if pred[i] else (0.15, 0.9, 0.3, 0.8))
        img = Image.fromarray(renderer.render()); d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([14, 14, int(1700 * fs), int(372 * fs)], fill=(0, 0, 0, 150))
        om = np.median(log["ours_us"][-ticks_per_frame:]) / 1e3
        d.text((28 * fs, 22 * fs), f"Panda backup-trajectory CBF safety filter, batch {B}, 50 Hz control, horizon N = {N} (0.4 s)", font=font_b, fill="white")
        d.text((28 * fs, 62 * fs), f"t = {t:5.2f} s    {K} of {B} simulated instances drawn, each with its own obstacle and target path; orange target = the filter changed this tick's torque", font=font, fill=(220, 220, 220))
        d.text((28 * fs, 98 * fs), f"cusadi2 model + CUDA interior point (applied):  {om:6.1f} ms per tick for all {B}   budget 20 ms", font=font, fill=(255, 205, 120))
        if log["frax_us"]:
            fm = np.median(log["frax_us"][-ticks_per_frame:]) / 1e3
            d.text((28 * fs, 134 * fs), f"frax model + qpax, same states (not applied):     {fm:6.1f} ms per tick   {fm / 20:4.1f}x over budget", font=font, fill=(150, 200, 255))
        d.text((28 * fs, 170 * fs), f"filter changed the torque: {act.sum():4d} / {B}   predictive rows changed it vs the one-step filter: {pred.sum():4d} / {B}   a predictive row active: {pred_rows.any(1).sum():4d} / {B}", font=font, fill=(255, 170, 90))
        d.text((28 * fs, 206 * fs), f"IPM: max iterations in batch {it.max():2d}, converged {100 * conv.mean():5.1f} %     instances with a barrier within 0.01: {(hv.min(1) < 0.01).sum():4d}", font=font, fill=(220, 220, 220))
        y = 242
        for gi, (g, sl) in enumerate(groups):
            hm = log["h_min"][g][-1]
            d.text(((28 + 430 * (gi % 3)) * fs, (y + 36 * (gi // 3)) * fs), f"batch-min h {g} {hm:7.3f}", font=font, fill=(255, 110, 90) if hm < 0.01 else (220, 220, 220))
        ff.stdin.write(np.asarray(img).tobytes()); n_frames += 1
    if k % 100 == 0:
        print(f"t={t:5.2f}s wall={time.perf_counter() - twall0:6.1f}s  ours med {np.median(log['ours_us']):.0f} µs" +
              (f"  frax med {np.median(log['frax_us']):.0f} µs" if log["frax_us"] else "") + f"  sim med {np.median(log['sim_us']):.0f} µs  h_min {min(log['h_min_all']):.4f}", flush=True)
if not args.no_video:
    ff.stdin.close(); ff.wait(); renderer.close()
pool.shutdown()
summary = {"batch": B, "N": N, "n_rows": prob.n_rows, "seconds": args.seconds, "ctrl_dt": CTRL_DT, "physics_dt": dt, "compile_ours_s": compile_ours, "compile_frax_s": compile_frax,
           "ours_us_median": float(np.median(log["ours_us"])), "ours_us_p99": float(np.percentile(log["ours_us"], 99)), "ours_us_max": float(np.max(log["ours_us"])),
           "frax_us_median": float(np.median(log["frax_us"])) if log["frax_us"] else None, "frax_us_p99": float(np.percentile(log["frax_us"], 99)) if log["frax_us"] else None,
           "sim_us_median": float(np.median(log["sim_us"])), "h_min": {g: float(min(v)) for g, v in log["h_min"].items()}, "h_min_all": float(min(log["h_min_all"])),
           "active_frac_mean": float(np.mean(log["active_frac"])), "pred_changed_frac_mean": float(np.mean(log["pred_frac"])), "pred_changed_frac_max": float(np.max(log["pred_frac"])),
           "pred_row_active_frac_mean": float(np.mean(log["pred_rows_frac"])), "ticks_with_pred_change": float(np.mean(np.array(log["pred_frac"]) > 0)),
           "iters_max_over_run": int(max(log["iters_max"])), "conv_frac_min": float(min(log["conv_frac"])), "n_ticks": n_ticks, "frames": n_frames}
common.save_json(out.with_suffix(".json"), summary)
print({k: v for k, v in summary.items() if k != "h_min"}); print("h_min per group:", summary["h_min"])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
for g, v in log["h_min"].items(): ax[0].plot(log["t"], v, lw=1.5, label=g)
ax[0].axhline(0, color="k", lw=0.8); ax[0].set_ylabel("batch-min h₂ per group"); ax[0].legend(fontsize=8, ncol=5); ax[0].set_ylim(-0.05, 0.6)
ax[1].plot(log["t"], 100 * np.array(log["active_frac"]), color="#eb6834", label="filter changed torque"); ax[1].plot(log["t"], 100 * np.array(log["pred_frac"]), color="#d4a017", label="predictive rows changed torque")
ax[1].plot(log["t"], 100 * np.array(log["near_frac"]), color="#2a78d6", label="barrier within 0.01"); ax[1].set_ylabel(f"% of {B} instances"); ax[1].legend(fontsize=8)
ax[2].plot(log["t"], np.array(log["ours_us"]) / 1e3, color="#2a78d6", lw=0.8, label="cusadi2 + CUDA IPM (applied)")
if log["frax_us"]: ax[2].plot(log["t"], np.array(log["frax_us"]) / 1e3, color="#eb6834", lw=0.8, label="frax + qpax (same states)")
ax[2].axhline(20, color="k", lw=0.8, ls="--", label="20 ms budget"); ax[2].set_ylabel("ms per tick"); ax[2].set_xlabel("time (s)"); ax[2].legend(fontsize=8); ax[2].set_ylim(0, None)
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=140); print(out.with_suffix(".png"))
