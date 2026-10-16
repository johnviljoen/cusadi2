"""The cupboard scene: geometry, the barrier list and the MuJoCo world.

A small, deep cupboard straight in front of the Panda (+x) with a cup standing near its back wall. Interior W (y) x H (z)
x D (x), front opening at x = X_OPEN, interior floor at z = Z_FLOOR, five slabs of thickness T_WALL (floor, ceiling,
left, right, back). The cup is a vertical cylinder of radius CUP_R and height CUP_H on the interior floor.

Robot geometry: frax's 21 collision spheres plus 4 finger-pad spheres (2 per finger, open hand) attached to the
end-effector frame. The barriers h2(q) (all relative degree 2), in row order:

    joint limits      q_max - q, q - q_min                                  14
    singularity       sqrt(det J J^T) - SING_TOL                             1
    self-collision    frax's body / root pairs                               7
    floor             p_z - r - FLOOR_HEIGHT, every sphere                    25
    slabs             sdf_box(p) - r, every sphere x every slab (sphere-major) 125
    cup               sdf_cyl(p) - r, every sphere but the finger sphere     24

The box and cylinder signed distances are exact (piecewise smooth); the norms carry SDF_EPS^2 under the square root
so that their derivatives stay finite at a sphere centre on an edge, a face, or the cup axis. The same formulas are
written once in numpy-style code (`sdf_box`, `sdf_cyl`) and evaluated on CasADi SX or jax.numpy arrays.
"""
import os as _os

import numpy as np

# ---- cupboard (metres, robot base frame) ----
# Sized by feasibility.py (results/feasibility.json): a horizontal grasp needs the shelf above the shoulder (joint 6's
# upper limit rules out a forward-pointing horizontal hand at counter height), and 0.32 x 0.28 is the smallest interior
# (1 cm steps from 0.28 x 0.24) that keeps every geometric barrier >= 1 cm at the pre-grasp and along a straight-in path.
W = 0.32                          # interior width (y)
D = float(_os.environ.get("CUPBOARD_D", 0.40))   # interior depth (x); env override: feasibility exploration only
H = float(_os.environ.get("CUPBOARD_H", 0.28))   # interior height (z); env override: feasibility exploration only
T_WALL = 0.02
X_OPEN, Y_MID = 0.42, 0.0
Z_FLOOR = float(_os.environ.get("CUPBOARD_Z_FLOOR", 0.55))      # env override: feasibility exploration only (feasibility.py)
# ---- cup ----
CUP_R, CUP_H = 0.025, 0.12
CUP_BACK_GAP = 0.06               # back wall to cup axis
CUP_X, CUP_Y = X_OPEN + D - CUP_BACK_GAP, Y_MID
CUP_Z0 = Z_FLOOR
# ---- grasp ----
GRASP_DEPTH = 0.013               # cup axis this far beyond the TCP along the approach (keeps the palm off the cup)
GRASP_Z = 0.09                    # grasp height above the shelf (the link-6 wrist sphere, r 8.3 cm, sits at TCP height)
P_PREGRASP = np.array([CUP_X - GRASP_DEPTH, CUP_Y, CUP_Z0 + GRASP_Z])
# hand frame in the world at the pre-grasp: tool axis z_h = +x (into the cupboard), finger axis y_h = -y (fingers close
# horizontally around the vertical cup), x_h = y_h x z_h = +z. (The other finger sense, x_h = -z, needs q7 past its limit.)
R_PREGRASP = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])

# ---- cost shaping (rungs 3 / 4 of the ladder; weights in the model's `par`, zero at rungs 1 / 2) ----
# funnel: w_fun * sigmoid((x_tcp - FUNNEL_X_ON) / FUNNEL_W) * |(y, z)_tcp - (y, z)_axis|^2 pulls the TCP onto the cupboard
# axis as it nears the opening; wrist alignment applies the same funnel to the point WRIST_BACK behind the TCP along the
# tool axis, so the forearm comes in straight.
FUNNEL_X_ON, FUNNEL_W, WRIST_BACK = X_OPEN - 0.10, 0.03, 0.20
# joint-space reference (rung 5): the max-clearance IK pre-grasp configuration of feasibility.py (results/feasibility.json),
# w_q |q - Q_PREGRASP|^2; the arm has to lay its forearm sideways to reach the cup, which no task-space cost suggests
Q_PREGRASP = np.array([0.888772, 0.346545, -0.963051, -1.397969, 2.352187, 3.080356, -1.284875])
AXIS_YZ = np.array([CUP_Y, CUP_Z0 + GRASP_Z])

# slabs as axis-aligned boxes (centre, half extents)
def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return 0.5 * (lo + hi), 0.5 * (hi - lo)

_x0, _x1, _yl, _yr, _z0, _z1 = X_OPEN, X_OPEN + D, Y_MID - W / 2, Y_MID + W / 2, Z_FLOOR, Z_FLOOR + H
SLABS = {
    "floor":   _box([_x0, _yl - T_WALL, _z0 - T_WALL], [_x1 + T_WALL, _yr + T_WALL, _z0]),
    "ceiling": _box([_x0, _yl - T_WALL, _z1], [_x1 + T_WALL, _yr + T_WALL, _z1 + T_WALL]),
    "left":    _box([_x0, _yr, _z0], [_x1, _yr + T_WALL, _z1]),
    "right":   _box([_x0, _yl - T_WALL, _z0], [_x1, _yl, _z1]),
    "back":    _box([_x1, _yl - T_WALL, _z0 - T_WALL], [_x1 + T_WALL, _yr + T_WALL, _z1 + T_WALL]),
}
SLAB_NAMES = list(SLABS)

# ---- finger-pad spheres (end-effector frame = hand frame, origin at the TCP) ----
# MuJoCo's open finger (finger_joint = 0.04): pad inner face at y = +-0.0415, pad centre at hand z 0.1029 (TCP at 0.105).
PAD_R = 0.01
PAD_POS = np.array([[0.0, s * (0.0415 + PAD_R), z] for s in (1.0, -1.0) for z in (-0.011, 0.005)])
FINGER_SPHERE = 19                # frax's sphere (0, 0, 0.184) r 0.04 on link 7: between the fingers, exempt from the cup
N_FRAX_SPHERES = 21
N_SPHERES = N_FRAX_SPHERES + len(PAD_POS)
CUP_SPHERES = [i for i in range(N_SPHERES) if i != FINGER_SPHERE]

SDF_EPS = 1e-9
N_SC = 7
N_ROWS = 14 + 1 + N_SC + N_SPHERES + N_SPHERES * len(SLABS) + len(CUP_SPHERES)
GROUPS = [("joint", 14), ("singularity", 1), ("self", N_SC), ("floor", N_SPHERES), ("slab", N_SPHERES * len(SLABS)),
          ("cup", len(CUP_SPHERES))]


def group_slices():
    out, i = {}, 0
    for name, n in GROUPS:
        out[name] = slice(i, i + n); i += n
    return out


def sdf_box(p, c, e, xp):
    """Exact signed distance from point p (3,) to the box centre c, half extents e. xp: casadi-like namespace with
    fabs, fmax, fmin, sqrt."""
    d = [xp.fabs(p[k] - c[k]) - e[k] for k in range(3)]
    o = [xp.fmax(d[k], 0.0) for k in range(3)]
    outside = xp.sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2] + SDF_EPS * SDF_EPS)
    inside = xp.fmin(xp.fmax(xp.fmax(d[0], d[1]), d[2]), 0.0)
    return outside + inside


def sdf_cyl(p, xp):
    """Exact signed distance from p to the cup (vertical cylinder, axis through (CUP_X, CUP_Y))."""
    dx, dy = p[0] - CUP_X, p[1] - CUP_Y
    dr = xp.sqrt(dx * dx + dy * dy + SDF_EPS * SDF_EPS) - CUP_R
    dz = xp.fabs(p[2] - (CUP_Z0 + 0.5 * CUP_H)) - 0.5 * CUP_H
    o0, o1 = xp.fmax(dr, 0.0), xp.fmax(dz, 0.0)
    return xp.sqrt(o0 * o0 + o1 * o1 + SDF_EPS * SDF_EPS) + xp.fmin(xp.fmax(dr, dz), 0.0)


def scene_rows(sph, rad, xp):
    """Floor, slab and cup barriers for sphere centres sph (list of 3-vectors) and radii rad (list), in row order."""
    from panda_casadi import FLOOR_HEIGHT
    floor = [sph[i][2] - rad[i] - FLOOR_HEIGHT for i in range(N_SPHERES)]
    slab = [sdf_box(sph[i], c, e, xp) - rad[i] for i in range(N_SPHERES) for (c, e) in SLABS.values()]
    cup = [sdf_cyl(sph[i], xp) - rad[i] for i in CUP_SPHERES]
    return floor + slab + cup


class _CaNS:
    """casadi namespace for the SDF formulas."""
    def __init__(self):
        import casadi as ca
        self.fabs, self.fmax, self.fmin, self.sqrt = ca.fabs, ca.fmax, ca.fmin, ca.sqrt


def barriers_casadi(m, q, K):
    """The N_ROWS barriers h2(q) on CasADi SX. m: backup_staged._Model, K: m.kin(q, v)."""
    import casadi as ca
    from panda_casadi import SING_TOL
    D, rad_f = m.D, m.rad
    J, sph_f, pos, rot = K["J"], K["sph"], K["pos"], K["rot"]
    h_jl = ca.vertcat(m.q_max - q, q - m.q_min)
    h_sing = ca.sqrt(ca.det(J @ J.T)) - SING_TOL
    h_sc = []
    for (a, b), tol in zip(D["body_sc_pairs"], D["body_sc_tols"]):
        h_sc.append(ca.norm_2(sph_f[b, :] - sph_f[a, :]) - rad_f[a] - rad_f[b] - tol)
    for (ri, bi), tol in zip(D["root_sc_pairs"], D["root_sc_tols"]):
        h_sc.append(ca.norm_2(sph_f[bi, :].T - ca.DM(D["root_pos"][ri])) - D["root_rad"][ri] - rad_f[bi] - tol)
    assert len(h_sc) == N_SC
    sph = [sph_f[i, :].T for i in range(N_FRAX_SPHERES)] + [pos + rot @ ca.DM(p) for p in PAD_POS]
    rad = [rad_f[i] for i in range(N_FRAX_SPHERES)] + [PAD_R] * len(PAD_POS)
    rows = scene_rows(sph, rad, _CaNS())
    h2 = ca.vertcat(h_jl, h_sing, *h_sc, *rows)
    assert h2.shape[0] == N_ROWS, (h2.shape, N_ROWS)
    return h2


def barriers_jnp(q_min, q_max, q, h_sing, h_sc, sph21, rad21, ee_pos, ee_rot):
    """The N_ROWS barriers on jax.numpy, vectorised over spheres x slabs (the frax / MJX evaluators). Same formulas and
    the same summation order as `sdf_box` / `sdf_cyl`, so the rows agree with the CasADi graph to rounding.
    h_sing (1,), h_sc (N_SC,): the demo's singularity and self-collision barriers; sph21 (21, 3), rad21 (21,): frax's
    sphere centres and radii; ee_pos (3,), ee_rot (3, 3): the end-effector frame the pad spheres are attached to."""
    import jax.numpy as jnp
    from panda_casadi import FLOOR_HEIGHT
    pads = ee_pos[None, :] + jnp.asarray(PAD_POS) @ ee_rot.T
    sph = jnp.concatenate([sph21, pads])                                   # (N_SPHERES, 3)
    rad = jnp.concatenate([jnp.asarray(rad21, sph.dtype), jnp.full(len(PAD_POS), PAD_R, sph.dtype)])
    h_jl = jnp.concatenate([jnp.asarray(q_max) - q, q - jnp.asarray(q_min)])
    floor = sph[:, 2] - rad - FLOOR_HEIGHT
    c = jnp.asarray(np.stack([c for c, _ in SLABS.values()])); e = jnp.asarray(np.stack([e for _, e in SLABS.values()]))
    d = jnp.abs(sph[:, None, :] - c[None]) - e[None]                       # (N_SPHERES, n_slab, 3)
    o = jnp.maximum(d, 0.0)
    outside = jnp.sqrt(o[..., 0] * o[..., 0] + o[..., 1] * o[..., 1] + o[..., 2] * o[..., 2] + SDF_EPS * SDF_EPS)
    inside = jnp.minimum(jnp.maximum(jnp.maximum(d[..., 0], d[..., 1]), d[..., 2]), 0.0)
    slab = (outside + inside - rad[:, None]).reshape(-1)                  # sphere-major
    pc = sph[jnp.asarray(CUP_SPHERES)]
    dx, dy = pc[:, 0] - CUP_X, pc[:, 1] - CUP_Y
    dr = jnp.sqrt(dx * dx + dy * dy + SDF_EPS * SDF_EPS) - CUP_R
    dz = jnp.abs(pc[:, 2] - (CUP_Z0 + 0.5 * CUP_H)) - 0.5 * CUP_H
    o0, o1 = jnp.maximum(dr, 0.0), jnp.maximum(dz, 0.0)
    cup = jnp.sqrt(o0 * o0 + o1 * o1 + SDF_EPS * SDF_EPS) + jnp.minimum(jnp.maximum(dr, dz), 0.0) - rad[jnp.asarray(CUP_SPHERES)]
    return jnp.concatenate([h_jl, h_sing, h_sc, floor, slab, cup])


def mujoco_scene_xml():
    """The cupboard and the cup as MuJoCo bodies (to attach the frax panda.xml to)."""
    boxes = "\n".join(
        f'    <geom name="slab_{n}" type="box" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" size="{e[0]:.4f} {e[1]:.4f} {e[2]:.4f}" '
        f'rgba="0.55 0.42 0.30 {0.35 if n in ("left", "ceiling") else 1.0}"/>' for n, (c, e) in SLABS.items())
    cz = CUP_Z0 + 0.5 * CUP_H
    return f"""<mujoco model="cupboard scene">
  <statistic center="0.4 0 0.4" extent="1"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="150" elevation="-15"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
{boxes}
    <body name="cup" pos="{CUP_X:.4f} {CUP_Y:.4f} {cz:.4f}">
      <freejoint name="cup"/>
      <geom name="cup" type="cylinder" size="{CUP_R} {0.5 * CUP_H}" mass="0.2" rgba="0.85 0.2 0.2 1"/>
    </body>
    <body name="target" mocap="true" pos="{P_PREGRASP[0]:.4f} {P_PREGRASP[1]:.4f} {P_PREGRASP[2]:.4f}">
      <geom name="target_geom" type="sphere" size="0.01" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
    print("rows", N_ROWS, dict(GROUPS))
    for n, (c, e) in SLABS.items():
        print(f"{n:8s} lo {np.round(c - e, 3)} hi {np.round(c + e, 3)}")
    print("cup axis", (CUP_X, CUP_Y), "pre-grasp TCP", P_PREGRASP)
