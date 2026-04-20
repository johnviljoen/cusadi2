"""
Build the A1 compliant-contact forward dynamics CasADi function.

Key design choices (addressing three independent failure modes of a naive
single-contact-point compliant model):

1. FOUR CONTACT POINTS PER FOOT (16 total). A single 3D point contact at
   each foot frame cannot resist yaw, and resists roll/pitch only weakly
   through friction. MuJoCo detects multiple contact points from the foot's
   collision geometry and gets real moment arms. We approximate that with
   four offset points at (±2cm, ±1cm, 0) in the foot frame — four corners
   of a rectangle. The x-separation creates pitch moment arms; y-separation
   creates roll moment arms; and tangent-friction at the y-offsets creates
   yaw moment arms.

2. SOFTPLUS PENETRATION. `ca.fmax(0, -z)` has a derivative discontinuity at
   z=0 — every contact engagement/release produces an impulse-like force
   jump that interacts badly with explicit integration. `log1p(exp(α·(-z)))/α`
   is C∞ smooth, matches fmax within ~1/α away from the transition, and
   has a continuous derivative everywhere.

3. SEMI-IMPLICIT-COMPATIBLE FORMULATION. The caller is expected to use
   semi-implicit Euler (update v first, then integrate q with the new v).
   The CasADi function computes ddq at (q, v, τ); the stability advantage
   over explicit Euler comes from the integration order downstream, not
   from anything in this function.

Inputs/outputs match the constrained forward-dynamics signature:
    q   (19,)    floating-base + 12 joints (quaternion xyzw)
    v   (18,)
    tau (18,)    first 6 = base wrench, last 12 = actuated torques
  → ddq (18,)    acceleration from unconstrained ABA + contact wrenches

Ground: z = 0.
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


FOOT_FRAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Four contact corners per foot, in the foot frame (x, y, z) meters. Matches
# a ~4cm × 2cm rectangular footprint; z=0 means corners lie in the foot frame
# origin's horizontal plane.
CORNERS_LOCAL = np.array([
    [+0.02, +0.01, 0.0],
    [+0.02, -0.01, 0.0],
    [-0.02, +0.01, 0.0],
    [-0.02, -0.01, 0.0],
])

MU = 0.7                  # Coulomb friction coefficient
V_EPS = 1e-3              # friction smoothing
SOFTPLUS_ALPHA = 200.0    # softplus sharpness (1/m); transition width ≈ 5 mm

# Two models compiled: a stiff "sim" model used for advancing the true state
# (with many sub-steps per control period) and a soft "rollout" model used
# inside the MPPI prediction loop at the larger rollout dt. The rollout
# model is coarse (feet sink ~1 cm into the ground under body weight) but
# is numerically stable at dt=0.02 with explicit/semi-implicit Euler. MPPI
# is designed to tolerate planner/sim model mismatch of this flavor.
PRESETS = {
    # Single stiff contact model used for both simulation and MPPI rollouts.
    # Rollouts use linearized-implicit Euler so stiffness no longer breaks
    # stability at the MPPI outer dt.
    "sim":      {"K": 3000.0, "D": 100.0, "Dj": 0.0, "name": "a1_compliant_sim"},
}


def skew_ca(v3: ca.SX) -> ca.SX:
    """3x3 skew-symmetric matrix of a CasADi 3-vector."""
    return ca.vertcat(
        ca.horzcat(0,     -v3[2],  v3[1]),
        ca.horzcat(v3[2],  0,     -v3[0]),
        ca.horzcat(-v3[1], v3[0],  0),
    )


def softplus_penetration(neg_z: ca.SX, alpha: float) -> ca.SX:
    """Smooth approximation to fmax(0, neg_z). C∞ everywhere."""
    return ca.log1p(ca.exp(alpha * neg_z)) / alpha


def build(K_normal: float, D_normal: float, fn_name: str, D_joint: float = 0.0) -> ca.Function:
    rp = load_robot_description("a1_description", root_joint=pin.JointModelFreeFlyer())
    model = rp.model
    print(f"A1 model: nq={model.nq} nv={model.nv} njoints={model.njoints}")

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
        foot_R = cdata.oMf[fid].rotation           # 3×3 SX
        foot_p = cdata.oMf[fid].translation        # 3 SX
        J_foot = cpin.getFrameJacobian(cmodel, cdata, fid,
                                        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)  # 6×nv
        J_lin = J_foot[:3, :]
        J_ang = J_foot[3:6, :]

        for corner_local in CORNERS_LOCAL:
            # World-frame corner position: p_foot + R_foot @ corner_local
            offset_world = foot_R @ ca.DM(corner_local)                     # 3×1
            point_world = foot_p + offset_world                              # 3×1

            # Jacobian of this corner: J_point = J_lin − skew(offset_world) · J_ang
            J_point = J_lin - skew_ca(offset_world) @ J_ang                  # 3×nv
            v_point = J_point @ v_sym                                         # 3×1

            z = point_world[2]
            vz = v_point[2]
            vx = v_point[0]
            vy = v_point[1]

            # Smooth penetration depth (≥ 0, ≈ −z when z < 0).
            pen = softplus_penetration(-z, SOFTPLUS_ALPHA)
            # Smooth contact gate: σ(−α·z) ≈ step(−z), continuous.
            gate = 1.0 / (1.0 + ca.exp(SOFTPLUS_ALPHA * z))

            # Normal force: spring + damping, clipped ≥ 0 so the damper
            # can't pull the foot back into the ground on rebound.
            f_normal = ca.fmax(0.0, K_normal * pen - D_normal * vz * gate)

            # Smoothed Coulomb friction (isotropic, magnitude ≤ μ·f_normal).
            v_tan_norm = ca.sqrt(vx * vx + vy * vy + V_EPS * V_EPS)
            f_fx = -MU * f_normal * vx / v_tan_norm
            f_fy = -MU * f_normal * vy / v_tan_norm

            # Map point force to joint torques via point Jacobian transpose.
            point_force = ca.vertcat(f_fx, f_fy, f_normal)
            tau_contact = tau_contact + J_point.T @ point_force

            n_points += 1

        print(f"  foot {fname}: 4 corners added ({n_points} contact points so far)")

    tau_total = tau_sym + tau_contact
    if D_joint > 0.0:
        # Joint-space viscous damping on the 12 actuated joints only (not
        # the floating-base velocity). Approximates the implicit-integrator
        # suppression of high-frequency joint modes that MuJoCo does
        # automatically; keeps explicit Euler at dt=0.02 stable.
        damp = ca.vertcat(ca.SX.zeros(6), D_joint * v_sym[6:])
        tau_total = tau_total - damp
    ddq = cpin.aba(cmodel, cdata, q_sym, v_sym, tau_total)

    return ca.Function(
        fn_name,
        [q_sym, v_sym, tau_sym],
        [ddq],
        ["q", "v", "tau"],
        ["ddq"],
    )


def build_with_jac(K_normal: float, D_normal: float, fn_name: str, D_joint: float = 0.0) -> ca.Function:
    """Build a function that returns BOTH ddq and A = ∂ddq/∂v.

    Used by the MPPI linearized-implicit-Euler rollout: each rollout step
    needs (ddq, A) so it can solve `(I - dt·A) v_next = v + dt·ddq`. Building
    them in one Function lets CasADi share subexpressions between the value
    and the Jacobian, so the combined kernel is ~2-3× the plain forward
    dynamics (not 2×N) — then the LP compile is a single kernel.
    """
    rp = load_robot_description("a1_description", root_joint=pin.JointModelFreeFlyer())
    model = rp.model
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
    for fid in foot_ids:
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
            z, vz, vx, vy = point_world[2], v_point[2], v_point[0], v_point[1]
            pen = softplus_penetration(-z, SOFTPLUS_ALPHA)
            gate = 1.0 / (1.0 + ca.exp(SOFTPLUS_ALPHA * z))
            f_normal = ca.fmax(0.0, K_normal * pen - D_normal * vz * gate)
            v_tan_norm = ca.sqrt(vx * vx + vy * vy + V_EPS * V_EPS)
            f_fx = -MU * f_normal * vx / v_tan_norm
            f_fy = -MU * f_normal * vy / v_tan_norm
            point_force = ca.vertcat(f_fx, f_fy, f_normal)
            tau_contact = tau_contact + J_point.T @ point_force

    tau_total = tau_sym + tau_contact
    if D_joint > 0.0:
        damp = ca.vertcat(ca.SX.zeros(6), D_joint * v_sym[6:])
        tau_total = tau_total - damp
    ddq = cpin.aba(cmodel, cdata, q_sym, v_sym, tau_total)
    A = ca.jacobian(ddq, v_sym)  # nv × nv

    return ca.Function(
        fn_name,
        [q_sym, v_sym, tau_sym],
        [ddq, A],
        ["q", "v", "tau"],
        ["ddq", "dfdv"],
    )


def main() -> None:
    os.makedirs(CUSADI_FUNCTION_DIR, exist_ok=True)
    for preset_name, cfg in PRESETS.items():
        print(f"\n========== preset: {preset_name} (K={cfg['K']} D={cfg['D']} Dj={cfg['Dj']}) ==========")
        f = build(cfg["K"], cfg["D"], cfg["name"], D_joint=cfg["Dj"])
        print(f"[build] function: {f.name()}")
        print(f"  n_instructions = {f.n_instructions():,}  sz_w={f.sz_w()}")
        plan = extract_level_plan(f)
        print(f"  LP plan: levels={plan.n_compute_levels}  total_ops={plan.total_compute_ops:,}  "
              f"peak_live_slots={plan.peak_live_slots}  "
              f"({plan.peak_live_slots * 8 / 1024:.1f} KB)")
        fits = plan.peak_live_slots * 8 <= 96 * 1024
        print(f"  fits in untiled LP shared memory (<96KB): {fits}")
        out_path = os.path.join(CUSADI_FUNCTION_DIR, f"{cfg['name']}.casadi")
        f.save(out_path)
        print(f"  saved: {out_path}")

    # Also build the (ddq, ∂ddq/∂v) function for implicit-Euler MPPI rollouts.
    print(f"\n========== compliant + velocity Jacobian (for implicit rollout) ==========")
    cfg = PRESETS["sim"]
    fj = build_with_jac(cfg["K"], cfg["D"], "a1_compliant_jac_v", D_joint=cfg["Dj"])
    print(f"[build] function: {fj.name()}")
    print(f"  n_in={fj.n_in()}  n_out={fj.n_out()}  "
          f"nnz_out=[{', '.join(str(fj.nnz_out(i)) for i in range(fj.n_out()))}]")
    print(f"  n_instructions = {fj.n_instructions():,}  sz_w={fj.sz_w()}")
    plan_j = extract_level_plan(fj)
    print(f"  LP plan: levels={plan_j.n_compute_levels}  total_ops={plan_j.total_compute_ops:,}  "
          f"peak_live_slots={plan_j.peak_live_slots}  "
          f"({plan_j.peak_live_slots * 8 / 1024:.1f} KB)")
    fits_j = plan_j.peak_live_slots * 8 <= 96 * 1024
    print(f"  fits in untiled LP shared memory (<96KB): {fits_j}")
    out_path = os.path.join(CUSADI_FUNCTION_DIR, "a1_compliant_jac_v.casadi")
    fj.save(out_path)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
