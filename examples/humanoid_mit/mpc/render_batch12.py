"""Headless MuJoCo render of a batch of 12 MIT humanoids, one per MPC instance,
each playing its 10-stage predicted trajectory. Runs on a remote machine via EGL
(MUJOCO_GL=egl). Output: batch12_frames/stage_XX.png, batch12_grid.png, batch12.mp4.

Pinocchio q = [x y z qx qy qz qw joints(18)]  ->  MuJoCo free joint = [x y z qw qx qy qz joints]
"""
import os, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "mit_humanoid_urdf/humanoid_full_sf_toeheel_mj.urdf")   # + <mujoco><compiler discardvisual="false"/>
Q = np.load(os.path.join(HERE, "batch12_q.npy")); dt = np.load(os.path.join(HERE, "batch12_dt.npy"))
B, N, NQ = Q.shape
NX, NY, SPACING = 4, 3, 1.0

def load_child():                            # a spec is consumed by attach, so load one per copy
    child = mujoco.MjSpec.from_file(URDF)
    for g in child.geoms:                   # draw meshes; put collision boxes in group 3 (hidden)
        g.group = 3 if (g.contype != 0 or g.conaffinity != 0) else 0     # hide collision geometry
        if g.group == 0: g.rgba = (0.35, 0.4, 0.5, 1)
    return child
spec = mujoco.MjSpec()
spec.option.gravity = (0, 0, 0)
spec.visual.global_.offwidth, spec.visual.global_.offheight = 1600, 900
spec.worldbody.add_light(pos=(0, 0, 5), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=0)
spec.worldbody.add_light(pos=(3, -4, 4), dir=(-0.4, 0.5, -0.6), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                 rgb1=(0.55, 0.7, 0.9), rgb2=(0.95, 0.97, 1.0), width=256, height=256)
spec.add_texture(name="checker", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=(0.75, 0.75, 0.78), rgb2=(0.88, 0.88, 0.9), width=300, height=300)
spec.add_material(name="floor", textures=["", "checker"] + [""] * 8, texrepeat=(12, 12), reflectance=0.1)
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(6, 6, 0.1), material="floor")
OFF = np.zeros((B, 2))
for i in range(B):
    ox, oy = (i % NX - (NX - 1) / 2) * SPACING, (i // NX - (NY - 1) / 2) * SPACING
    OFF[i] = (ox, oy); frame = spec.worldbody.add_frame(pos=(ox, oy, 0))
    child = load_child(); roots = child.worldbody.bodies
    assert len(roots) == 1, [b.name for b in roots]
    root = frame.attach_body(roots[0], prefix=f"h{i}_", suffix="")
    root.add_freejoint()
model = spec.compile(); data = mujoco.MjData(model)
assert model.nq == B * NQ, (model.nq, B, NQ)
PIN_JOINTS = ['a01_right_hip_yaw', 'a02_right_hip_abad', 'a03_right_hip_pitch', 'a04_right_knee', 'a05_right_ankle',
              'a06_left_hip_yaw', 'a07_left_hip_abad', 'a08_left_hip_pitch', 'a09_left_knee', 'a10_left_ankle',
              'a11_right_shoulder_pitch', 'a12_right_shoulder_abad', 'a13_right_shoulder_yaw', 'a14_right_elbow',
              'a15_left_shoulder_pitch', 'a16_left_shoulder_abad', 'a17_left_shoulder_yaw', 'a18_left_elbow']
names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
mj_joints = [n.removeprefix("h0_") for n in names[1:19]]
perm = [PIN_JOINTS.index(n) for n in mj_joints]        # MuJoCo joint j takes Pinocchio joint perm[j]
print("joint order identical:", perm == list(range(18)))

def set_pose(k):
    for i in range(B):
        q = Q[i, k]; base = i * NQ
        data.qpos[base:base + 3] = q[:3] + [OFF[i, 0], OFF[i, 1], 0.0]   # free joint qpos is a world pose
        data.qpos[base + 3:base + 7] = [q[6], q[3], q[4], q[5]]       # xyzw -> wxyz
        data.qpos[base + 7:base + NQ] = q[7:][perm]
    mujoco.mj_forward(model, data)

cam = mujoco.MjvCamera(); cam.lookat[:] = (0, 0, 0.45); cam.distance = 5.2; cam.azimuth = 135; cam.elevation = -20
opt = mujoco.MjvOption(); opt.geomgroup[3] = 0
ren = mujoco.Renderer(model, height=900, width=1600)
out = os.path.join(HERE, "batch12_frames"); os.makedirs(out, exist_ok=True)
from PIL import Image
frames = []
for k in range(N):
    set_pose(k); ren.update_scene(data, camera=cam, scene_option=opt); img = ren.render()
    frames.append(img); Image.fromarray(img).save(os.path.join(out, f"stage_{k:02d}.png"))
# contact sheet: stages 0, 3, 6, 9
sheet = np.concatenate([np.concatenate([frames[0], frames[3]], axis=1), np.concatenate([frames[6], frames[9]], axis=1)], axis=0)
Image.fromarray(sheet).resize((1600, 900)).save(os.path.join(HERE, "batch12_grid.png"))
seq = os.path.join(HERE, "batch12_seq"); os.makedirs(seq, exist_ok=True); n = 0
for k in range(N):                                            # hold each stage for its dt at 30 fps
    for _ in range(max(1, int(round(dt[k] * 30)))):
        Image.fromarray(frames[k]).save(os.path.join(seq, f"f_{n:03d}.png")); n += 1
print("video frames:", n)
print("RENDER DONE", len(frames), "frames ->", out)
