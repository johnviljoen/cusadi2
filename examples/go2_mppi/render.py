"""Headless MuJoCo render of a closed-loop run written by mppi_rollout.cu (EGL, MUJOCO_GL=egl): one
frame per control step, so the video plays in real time at the control rate. MuJoCo only draws: the
terrain is a height field sampled from the same course.height the rollouts use, and every robot's
qpos is set from the logged state.

    python render.py <build_dir> <run_dir> <out_stem> [flat|course]     -> <out_stem>.mp4, <out_stem>.png
(TRACK=<robot> follows one robot from the side, for debugging.)

Pinocchio q = [x y z qx qy qz qw joints(12)]  ->  MuJoCo free joint = [x y z qw qx qy qz], same joint order.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from PIL import Image

import course
import go2_model as G

BUILD, RUN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SCEN = sys.argv[4] if len(sys.argv) > 4 else "course"
meta = json.load(open(os.path.join(BUILD, "meta.json")))
dtc = meta["dt"] * meta["substeps"]
rob = np.fromfile(os.path.join(RUN, "robots.f64")).reshape(-1, 6)
R = len(rob)
Q = np.fromfile(os.path.join(RUN, "state.f64")).reshape(-1, R, G.NX)[:, :, :G.NQ]
n_steps = Q.shape[0] - 1
W, H = 1600, 900
COLORS = [(0.84, 0.25, 0.22), (0.95, 0.55, 0.15), (0.20, 0.55, 0.30), (0.45, 0.75, 0.35),
          (0.15, 0.40, 0.75), (0.35, 0.70, 0.85), (0.50, 0.30, 0.65), (0.80, 0.50, 0.75)]

spec = mujoco.MjSpec()
spec.option.gravity = (0, 0, 0)
spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
spec.visual.headlight.diffuse = (0.3, 0.3, 0.3); spec.visual.headlight.ambient = (0.35, 0.35, 0.35)
spec.visual.map.zfar = 80; spec.visual.quality.shadowsize = 8192
spec.worldbody.add_light(pos=(4, -8, 10), dir=(-0.3, 0.6, -0.75), diffuse=(0.6, 0.6, 0.6), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                 rgb1=(0.55, 0.7, 0.9), rgb2=(0.95, 0.97, 1.0), width=256, height=256)
spec.add_texture(name="checker", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=(0.62, 0.64, 0.68), rgb2=(0.8, 0.81, 0.83), width=300, height=300)
if SCEN == "flat":
    centre = np.array([rob[:, [0, 3]].mean(), rob[:, [1, 4]].mean()])
    spec.add_material(name="floor", textures=["", "checker"] + [""] * 8, texrepeat=(40, 40), reflectance=0.0)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(20, 20, 0.1), pos=(centre[0], centre[1], 0), material="floor")
else:
    centre = np.zeros(2)
    HALF, CELL = 6.6, 0.01                         # the height field resolves the tanh risers (course.EDGE)
    n = int(2 * HALF / CELL) + 1
    g = np.linspace(-HALF, HALF, n)
    hh = course.height(g[None, :], g[:, None])     # rows are y, columns x
    # texture over the whole field: 1 m checks, tinted by height, risers darkened by slope
    TPX = 0.005
    nt = int(2 * HALF / TPX)
    t = -HALF + (np.arange(nt) + 0.5) * TPX
    ht = course.height(t[None, :], t[:, None])
    gy, gx = np.gradient(ht, TPX)
    check = ((np.floor(t[None, :]) + np.floor(t[:, None])) % 2)[..., None]
    lvl = np.clip(ht / course.H_SUMMIT, 0, 1)[..., None]
    rgb = (0.66 + 0.14 * check) * ((1 - lvl) * np.array([0.92, 0.95, 1.0]) + lvl * np.array([1.0, 0.80, 0.52]))
    rgb = rgb * (1 - 0.5 * np.clip(np.hypot(gx, gy) / 2, 0, 1))[..., None]
    tex_png = os.path.join(tempfile.mkdtemp(prefix="go2_course_"), "course.png")
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(tex_png)
    spec.add_texture(name="course", type=mujoco.mjtTexture.mjTEXTURE_2D, file=tex_png)
    spec.add_material(name="floor", textures=["", "course"] + [""] * 8, texrepeat=(1, 1), reflectance=0.0)
    spec.add_hfield(name="course", nrow=n, ncol=n, size=(HALF, HALF, course.H_SUMMIT, 0.05), userdata=(hh / course.H_SUMMIT).ravel())
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="course", material="floor")
    for i in range(R):                             # parking spots
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=(0.06, 0.002, 0), rgba=COLORS[i % 8] + (1,),
                                pos=(rob[i, 3], rob[i, 4], float(course.height(rob[i, 3], rob[i, 4])) + 0.002), contype=0, conaffinity=0)
for i in range(R):
    child = mujoco.MjSpec.from_file(G.XML)         # a spec is consumed by attach, so load one per copy
    for gm in child.geoms:
        if gm.classname.name in ("collision", "foot"):
            gm.group = 3
    for mt in child.materials:
        if mt.name == "gray":
            mt.rgba = COLORS[i % 8] + (1,)
    spec.worldbody.add_frame().attach_body(child.worldbody.bodies[0], prefix=f"r{i}_", suffix="")
model = spec.compile(); data = mujoco.MjData(model)
assert model.nq == R * G.NQ, (model.nq, R)


def set_pose(s):
    for i in range(R):
        q = Q[s, i]; b = i * G.NQ
        data.qpos[b:b + 3] = q[:3]
        data.qpos[b + 3:b + 7] = [q[6], q[3], q[4], q[5]]              # xyzw -> wxyz
        data.qpos[b + 7:b + G.NQ] = q[7:]
    mujoco.mj_forward(model, data)


cam = mujoco.MjvCamera(); cam.lookat[:] = (centre[0], centre[1], 0.2)
cam.distance, cam.azimuth, cam.elevation = (11.5, 62, -42) if SCEN == "course" else (14.0, 60, -40)
TRACK = os.environ.get("TRACK")                    # debugging: follow one robot up close
opt = mujoco.MjvOption(); opt.geomgroup[3] = 0
ren = mujoco.Renderer(model, height=H, width=W)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
sheet = []
with tempfile.TemporaryDirectory() as tmp:
    for s in range(n_steps + 1):
        set_pose(s)
        if TRACK is not None:
            cam.lookat[:] = Q[s, int(TRACK), :3]; cam.distance, cam.elevation = 2.2, -20
            cam.azimuth = float(os.environ.get("AZ", 90 + np.rad2deg(rob[int(TRACK), 5])))
        ren.update_scene(data, camera=cam, scene_option=opt); img = ren.render()
        Image.fromarray(img).save(os.path.join(tmp, f"f_{s:05d}.png"))
        if s in [round(j * n_steps / 3) for j in range(4)]:
            sheet.append(img.copy())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{1 / dtc:.6f}", "-i", os.path.join(tmp, "f_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUT + ".mp4"], check=True)
grid = np.concatenate([np.concatenate(sheet[:2], axis=1), np.concatenate(sheet[2:], axis=1)], axis=0)   # t = 0, T/3, 2T/3, T
Image.fromarray(grid).resize((W, H)).save(OUT + ".png")
print(f"RENDER DONE {n_steps + 1} frames at {1 / dtc:.0f} fps, {(n_steps + 1) * dtc:.2f} s -> {OUT}.mp4")
