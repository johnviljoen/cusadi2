"""
Build the H1 compliant-contact forward dynamics CasADi function.

Structural parallel to build_a1_compliant_fwd.py, adapted for the H1
humanoid:

- 2 feet (left_ankle_link / right_ankle_link) × 4 corners each = 8 contact
  points (vs 16 on A1). H1 feet are larger and roughly flat under the ankle;
  the corner offsets approximate a ~18 cm × 8 cm rectangle with the sole
  8 cm below the ankle joint origin.
- K = 5000 N/m (vs 3000 on A1) — heavier robot (~45 kg vs 12 kg) so each
  contact point supports ~55 N at body weight → ~1.1 cm sink at
  equilibrium, same order as A1.
- D = 150 N·s/m (scaled roughly with K).
- α = 200 / m softplus sharpness, μ = 0.7, v_eps = 1e-3 — unchanged from A1.

Inputs/outputs:
    q   (26,)   floating-base + 19 joints (quaternion xyzw)
    v   (25,)
    tau (25,)   first 6 = base wrench, last 19 = actuated torques
  → ddq (25,)   acceleration from unconstrained ABA + contact wrenches
"""
from __future__ import annotations

import os
import sys

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from robot_descriptions.loaders.pinocchio import load_robot_description

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src import CUSADI_FUNCTION_DIR  # noqa: E402
from src.level_parallel.dag import extract_level_plan  # noqa: E402


FOOT_FRAMES = ["left_ankle_link", "right_ankle_link"]

# Four contact corners per foot, in the ankle_link frame (x, y, z) meters.
# x: forward (+toe) / backward (-heel)
# y: lateral (symmetric ±)
# z: down to the sole surface (negative)
CORNERS_LOCAL = np.array([
    [+0.12, +0.04, -0.08],   # toe-outer
    [+0.12, -0.04, -0.08],   # toe-inner
    [-0.06, +0.04, -0.08],   # heel-outer
    [-0.06, -0.04, -0.08],   # heel-inner
])

K_NORMAL = 5000.0
D_NORMAL = 150.0
MU = 0.7
V_EPS = 1e-3
SOFTPLUS_ALPHA = 200.0


def skew_ca(v3: ca.SX) -> ca.SX:
    return ca.vertcat(
        ca.horzcat(0,     -v3[2],  v3[1]),
        ca.horzcat(v3[2],  0,     -v3[0]),
        ca.horzcat(-v3[1], v3[0],  0),
    )


def softplus_penetration(neg_z: ca.SX, alpha: float) -> ca.SX:
    return ca.log1p(ca.exp(alpha * neg_z)) / alpha


def build(fn_name: str = "h1_compliant_sim") -> ca.Function:
    rp = load_robot_description("h1_description", root_joint=pin.JointModelFreeFlyer())
    model = rp.model
    print(f"H1 model: nq={model.nq} nv={model.nv} njoints={model.njoints}")

    foot_ids = [model.getFrameId(f) for f in FOOT_FRAMES]

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    q_sym = ca.SX.sym("q", model.nq)
    v_sym = ca.SX.sym("v", model.nv)
    tau_sym = ca.SX.sym("tau", model.nv)

    cpin.forwardKinematics(cmodel, cdata, q_sym, v_sym)
    cpin.updateFramePlacements(cmodel, cdata)
    cpin.computeJointJacobians(cmodel, cdata, q_sym)

    tau_contact = ca.SX.zeros(model.nv)
    n_points = 0
    for fname, fid in zip(FOOT_FRAMES, foot_ids):
        foot_R = cdata.oMf[fid].rotation
        foot_p = cdata.oMf[fid].translation
        J_foot = cpin.getFrameJacobian(cmodel, cdata, fid,
                                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_lin = J_foot[:3, :]
        J_ang = J_foot[3:6, :]

        for corner_local in CORNERS_LOCAL:
            offset_world = foot_R @ ca.DM(corner_local)
            point_world = foot_p + offset_world

            J_point = J_lin - skew_ca(offset_world) @ J_ang
            v_point = J_point @ v_sym

            z = point_world[2]
            vz = v_point[2]
            vx = v_point[0]
            vy = v_point[1]

            pen = softplus_penetration(-z, SOFTPLUS_ALPHA)
            gate = 1.0 / (1.0 + ca.exp(SOFTPLUS_ALPHA * z))

            f_normal = ca.fmax(0.0, K_NORMAL * pen - D_NORMAL * vz * gate)

            v_tan_norm = ca.sqrt(vx * vx + vy * vy + V_EPS * V_EPS)
            f_fx = -MU * f_normal * vx / v_tan_norm
            f_fy = -MU * f_normal * vy / v_tan_norm

            point_force = ca.vertcat(f_fx, f_fy, f_normal)
            tau_contact = tau_contact + J_point.T @ point_force
            n_points += 1

        print(f"  foot {fname}: 4 corners added ({n_points} contact points so far)")

    tau_total = tau_sym + tau_contact
    ddq = cpin.aba(cmodel, cdata, q_sym, v_sym, tau_total)

    return ca.Function(
        fn_name,
        [q_sym, v_sym, tau_sym],
        [ddq],
        ["q", "v", "tau"],
        ["ddq"],
    )


def main() -> None:
    os.makedirs(CUSADI_FUNCTION_DIR, exist_ok=True)
    print(f"========== H1 compliant (K={K_NORMAL} D={D_NORMAL}) ==========")
    f = build()
    print(f"[build] function: {f.name()}")
    print(f"  n_instructions = {f.n_instructions():,}  sz_w={f.sz_w()}")
    plan = extract_level_plan(f)
    print(f"  LP plan: levels={plan.n_compute_levels}  total_ops={plan.total_compute_ops:,}  "
          f"peak_live_slots={plan.peak_live_slots}  "
          f"({plan.peak_live_slots * 8 / 1024:.1f} KB)")
    fits = plan.peak_live_slots * 8 <= 96 * 1024
    print(f"  fits in untiled LP shared memory (<96KB): {fits}")
    out_path = os.path.join(CUSADI_FUNCTION_DIR, "h1_compliant_sim.casadi")
    f.save(out_path)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
