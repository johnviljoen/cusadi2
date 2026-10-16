"""Render the feasibility IK poses in the MuJoCo cupboard world and list mesh contacts (the sphere model is the
controller's; this checks it against the real collision meshes).

    python render_still.py [--feas results/feasibility.json] [--size 1280x960]

Writes results/still_pregrasp.png (two views side by side) and results/still_path.png (path poses overlaid as stills).
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import numpy as np
import mujoco
import common
import scene

ap = argparse.ArgumentParser()
ap.add_argument("--feas", default=str(HERE / "results" / "feasibility.json"))
ap.add_argument("--size", default="960x720")
a = ap.parse_args()
W, H = map(int, a.size.split("x"))


def build_model():
    fx = common.FRAX_DIR / "examples/xml"
    rs = mujoco.MjSpec.from_file(str(fx / "panda.xml"))
    ss = mujoco.MjSpec.from_string(scene.mujoco_scene_xml())
    ss.meshdir = str(fx)          # not used by the scene itself; panda.xml keeps its own asset paths
    ss.attach(rs, prefix="", frame=ss.worldbody.add_frame(pos=[0, 0, 0]))
    ss.visual.global_.offwidth = W; ss.visual.global_.offheight = H; ss.visual.map.znear = 0.01; ss.visual.quality.shadowsize = 8192
    return ss.compile()


def set_pose(model, data, q):
    data.qpos[:] = model.qpos0
    for i in range(7):
        data.qpos[model.joint(f"joint{i + 1}").qposadr[0]] = q[i]
    for n in ("finger_joint1", "finger_joint2"):
        data.qpos[model.joint(n).qposadr[0]] = 0.04   # fingers open
    mujoco.mj_forward(model, data)


def contacts(model, data):
    out = []
    for c in data.contact[:data.ncon]:
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or n1
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or n2
        out.append((f"{n1}/{g1}", f"{n2}/{g2}", round(float(c.dist), 4)))
    return out


def min_dist(model, data, robot_bodies, others, dmax=0.2):
    """Smallest mesh distance between robot geoms and the given scene geoms (mj_geomDistance)."""
    best = (np.inf, None)
    ft = np.zeros(6)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in robot_bodies or model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        for o in others:
            d = mujoco.mj_geomDistance(model, data, g, o, dmax, ft)
            if d < best[0]:
                best = (d, (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]), mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, o)))
    return best


def render(model, data, cams):
    r = mujoco.Renderer(model, H, W)
    imgs = []
    for az, el, dist, look in cams:
        cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = az, el, dist; cam.lookat[:] = look
        r.update_scene(data, cam); imgs.append(r.render().copy())
    r.close()
    return np.concatenate(imgs, axis=1)


feas = json.loads(Path(a.feas).read_text())
model = build_model(); data = mujoco.MjData(model)
robot_bodies = {model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith(("link", "hand", "left_finger", "right_finger"))}
scene_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"slab_{n}") for n in scene.SLAB_NAMES] + [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup")]
look = [scene.X_OPEN + 0.1, 0.0, scene.Z_FLOOR + 0.1]
cams = [(-90, -10, 1.4, look), (90, -89, 1.3, look), (-150, -25, 1.6, look)]   # through the left wall, top, over the shoulder

set_pose(model, data, feas["pregrasp"]["q"])
d, pair = min_dist(model, data, robot_bodies, scene_geoms)
print("pre-grasp q", np.round(feas["pregrasp"]["q"], 3).tolist())
print(f"  mesh contacts: {contacts(model, data)}")
print(f"  min robot-mesh to cupboard/cup distance: {d*100:.2f} cm {pair}")
from PIL import Image
out = HERE / "results"
Image.fromarray(render(model, data, cams)).save(out / "still_pregrasp.png")
frames = []
for p in feas["path"][::3]:
    set_pose(model, data, p["q"])
    d, pair = min_dist(model, data, robot_bodies, scene_geoms)
    print(f"  path x = {p['x']:.3f}: sphere clearance {p['s']*100:5.2f} cm, mesh clearance {d*100:5.2f} cm {pair}, contacts {len(contacts(model, data))}")
    frames.append(render(model, data, cams[:1]))
Image.fromarray(np.concatenate([np.concatenate(frames[i:i + 3], axis=1) for i in range(0, len(frames) - len(frames) % 3, 3)], axis=0)).save(out / "still_path.png")
print("wrote", out / "still_pregrasp.png", out / "still_path.png")
