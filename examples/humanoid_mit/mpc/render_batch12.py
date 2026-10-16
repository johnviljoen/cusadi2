"""Headless MuJoCo render of the 12-instance closed-loop rollout written by mpc_rollout.cu: one frame per
control step, so frames are dt_0 apart and the video plays in real time at 1 / dt_0 fps (EGL, MUJOCO_GL=egl).
Output: results/batch12_closed_loop.mp4, results/batch12_closed_loop.png (contact sheet).

Rows are the yaw-rate commands (-0.5, 0, +0.5 rad/s), columns the forward speeds (0 .. 0.6 m/s); every robot
starts facing +x. The camera is fixed.

Pinocchio q = [x y z qx qy qz qw joints(18)]  ->  MuJoCo free joint = [x y z qw qx qy qz joints]

usage: python render_batch12.py <build_dir>      (URDF and meshes from vendor/, see vendor_setup.py)
"""
import json, os, subprocess, sys, tempfile, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); BUILD = sys.argv[1]; RES = os.path.join(HERE, "results")
URDF = os.path.join(os.environ.get("CUSADI_WIP", os.path.join(HERE, "vendor")), "mit_humanoid_urdf/humanoid_full_sf_toeheel_mj.urdf")
meta = json.load(open(os.path.join(BUILD, "rollout", "meta.json"))); B, n_steps, dt0 = meta["B"], meta["n_steps"], meta["dt0"]
NQ = 25
Q = np.fromfile(os.path.join(BUILD, "rollout", "state.f64")).reshape(n_steps + 1, B, -1)[:, :, :NQ]
NX, NY, DX, DY, W, H = 4, 3, 2.2, 2.0, 1600, 900

def load_child():                            # a spec is consumed by attach, so load one per copy
    child = mujoco.MjSpec.from_file(URDF); child.meshdir = os.path.dirname(URDF)
    for g in child.geoms:                   # draw meshes; put collision boxes in group 3 (hidden)
        g.group = 3 if (g.contype != 0 or g.conaffinity != 0) else 0
        if g.group == 0: g.rgba = (0.16, 0.22, 0.34, 1)
    return child
spec = mujoco.MjSpec()
spec.option.gravity = (0, 0, 0)
spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
spec.visual.headlight.diffuse = (0.25, 0.25, 0.25); spec.visual.headlight.ambient = (0.3, 0.3, 0.3)
spec.worldbody.add_light(pos=(3, -6, 6), dir=(-0.3, 0.6, -0.7), diffuse=(0.55, 0.55, 0.55), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                 rgb1=(0.55, 0.7, 0.9), rgb2=(0.95, 0.97, 1.0), width=256, height=256)
spec.add_texture(name="checker", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=(0.62, 0.64, 0.68), rgb2=(0.8, 0.81, 0.83), width=300, height=300)
spec.add_material(name="floor", textures=["", "checker"] + [""] * 8, texrepeat=(20, 20), reflectance=0.0)   # 1 m checks
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(20, 20, 0.1), material="floor")
OFF = np.zeros((B, 2))
for i in range(B):
    OFF[i] = ((i % NX - (NX - 1) / 2) * DX, (i // NX - (NY - 1) / 2) * DY)
    child = load_child(); roots = child.worldbody.bodies
    assert len(roots) == 1, [b.name for b in roots]
    spec.worldbody.add_frame().attach_body(roots[0], prefix=f"h{i}_", suffix="").add_freejoint()
model = spec.compile(); data = mujoco.MjData(model)
assert model.nq == B * NQ, (model.nq, B, NQ)
PIN_JOINTS = ['a01_right_hip_yaw', 'a02_right_hip_abad', 'a03_right_hip_pitch', 'a04_right_knee', 'a05_right_ankle',
              'a06_left_hip_yaw', 'a07_left_hip_abad', 'a08_left_hip_pitch', 'a09_left_knee', 'a10_left_ankle',
              'a11_right_shoulder_pitch', 'a12_right_shoulder_abad', 'a13_right_shoulder_yaw', 'a14_right_elbow',
              'a15_left_shoulder_pitch', 'a16_left_shoulder_abad', 'a17_left_shoulder_yaw', 'a18_left_elbow']
names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
perm = [PIN_JOINTS.index(n.removeprefix("h0_")) for n in names[1:19]]        # MuJoCo joint j takes Pinocchio joint perm[j]

def set_pose(s):
    for i in range(B):
        q = Q[s, i]; base = i * NQ
        data.qpos[base:base + 3] = q[:3] + [OFF[i, 0], OFF[i, 1], 0.0]   # free joint qpos is a world pose
        data.qpos[base + 3:base + 7] = [q[6], q[3], q[4], q[5]]       # xyzw -> wxyz
        data.qpos[base + 7:base + NQ] = q[7:][perm]
    mujoco.mj_forward(model, data)

cam = mujoco.MjvCamera(); cam.lookat[:] = (1.0, 0.3, 0.3); cam.distance = 10.5; cam.azimuth = 65; cam.elevation = -33
opt = mujoco.MjvOption(); opt.geomgroup[3] = 0
ren = mujoco.Renderer(model, height=H, width=W)
os.makedirs(RES, exist_ok=True)
sheet = []
with tempfile.TemporaryDirectory() as tmp:
    for s in range(n_steps + 1):
        set_pose(s); ren.update_scene(data, camera=cam, scene_option=opt); img = ren.render()
        Image.fromarray(img).save(os.path.join(tmp, f"f_{s:04d}.png"))
        if s in [round(j * n_steps / 3) for j in range(4)]: sheet.append(img.copy())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{1 / dt0:.6f}", "-i", os.path.join(tmp, "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", os.path.join(RES, "batch12_closed_loop.mp4")], check=True)
grid = np.concatenate([np.concatenate(sheet[:2], axis=1), np.concatenate(sheet[2:], axis=1)], axis=0)   # t = 0, T/3, 2T/3, T
Image.fromarray(grid).resize((W, H)).save(os.path.join(RES, "batch12_closed_loop.png"))
print(f"RENDER DONE {n_steps + 1} frames, {dt0 * 1e3:.2f} ms apart, {1 / dt0:.2f} fps, {(n_steps + 1) * dt0:.2f} s")
