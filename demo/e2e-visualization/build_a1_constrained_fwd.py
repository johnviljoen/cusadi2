"""
Build the FORWARD-ONLY constrained dynamics CasADi function for A1.

Uses the same 4 × CONTACT_3D setup as exploration/constrained_dynamics (FL/FR/RL/RR_foot,
LOCAL_WORLD_ALIGNED), but returns only cdata.ddq — not the 3-output Jacobian.
For MPPI rollouts the controller needs ddq for state propagation, not derivatives.

Saves src/casadi_functions/a1_constrained_fwd.casadi.
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


CONTACTS = [
    ("FL_foot", pin.ContactType.CONTACT_3D),
    ("FR_foot", pin.ContactType.CONTACT_3D),
    ("RL_foot", pin.ContactType.CONTACT_3D),
    ("RR_foot", pin.ContactType.CONTACT_3D),
]


def build() -> ca.Function:
    rp = load_robot_description("a1_description", root_joint=pin.JointModelFreeFlyer())
    model = rp.model
    data = rp.data
    print(f"A1 model: nq={model.nq} nv={model.nv} njoints={model.njoints}")

    # Home configuration used as the Baumgarte target (feet welded at their
    # home-config world positions). RigidConstraintModel.desired_contact_placement
    # defaults to SE3.Identity (translation = 0); without overriding, Baumgarte
    # would pull feet toward the origin with enormous force.
    HOME_JOINT = np.array([0.0, 0.8, -1.6] * 4, dtype=np.float64)
    q_home = np.concatenate([
        [0.0, 0.0, 0.30],    # base xyz
        [0.0, 0.0, 0.0, 1.0], # base quaternion xyzw (identity)
        HOME_JOINT,
    ])
    pin.forwardKinematics(model, data, q_home)
    pin.updateFramePlacements(model, data)

    # NOTE: Pinocchio's RigidConstraintModel supports Baumgarte stabilization
    # via `.m_baumgarte_parameters.{Kp,Kd}` and `.desired_contact_placement`,
    # but the numeric targets set on a pin.RigidConstraintModel do NOT
    # propagate through the `cpin.RigidConstraintModel(cm)` conversion that
    # pinocchio.casadi uses to build the symbolic form. Setting Kp=400 Kd=40
    # with target-home vs target-origin produces bit-identical symbolic ddq
    # (|ddq| = 676 in both), because both collapse to target=identity in the
    # cpin copy. Without a working target setter on the symbolic side, any
    # non-zero Baumgarte gains just push the feet toward the origin very
    # hard. Setting Kp=Kd=0 disables the corrector; position-level
    # stabilization is instead applied on the Python side after integration
    # (see projection logic in a1_mppi.py).

    pcontact_models = []
    for frame_name, ctype in CONTACTS:
        fid = model.getFrameId(frame_name)
        pj = model.frames[fid].parentJoint
        placement = model.frames[fid].placement
        cm = pin.RigidConstraintModel(
            ctype, model, pj, placement, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        pcontact_models.append(cm)
        print(f"  contact @ {frame_name:<10s}  parentJoint={pj}  CONTACT_3D  "
              f"home_target_xyz={np.round(data.oMf[fid].translation, 4)}")

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    ccontact_models = cpin.StdVec_RigidConstraintModel()
    for cm in pcontact_models:
        ccontact_models.append(cpin.RigidConstraintModel(cm))
    ccontact_data = cpin.StdVec_RigidConstraintData()
    for cm in ccontact_models:
        ccontact_data.append(cpin.RigidConstraintData(cm))
    prox = cpin.ProximalSettings(1e-10, 1e-10, 1)

    q_sym = ca.SX.sym("q", model.nq)
    v_sym = ca.SX.sym("v", model.nv)
    tau_sym = ca.SX.sym("tau", model.nv)

    cpin.initConstraintDynamics(cmodel, cdata, ccontact_models, ccontact_data)
    cpin.constraintDynamics(cmodel, cdata, q_sym, v_sym, tau_sym,
                            ccontact_models, ccontact_data, prox)

    f = ca.Function(
        "a1_constrained_fwd",
        [q_sym, v_sym, tau_sym],
        [cdata.ddq],
        ["q", "v", "tau"],
        ["ddq"],
    )
    return f


def main() -> None:
    f = build()
    print(f"\n[build] function: {f.name()}")
    print(f"  n_in  = {f.n_in()}   ({', '.join(f.name_in(i) for i in range(f.n_in()))})")
    print(f"  n_out = {f.n_out()}  ({', '.join(f.name_out(i) for i in range(f.n_out()))})")
    for i in range(f.n_out()):
        print(f"    {f.name_out(i)}: shape={f.sparsity_out(i).shape}  nnz={f.nnz_out(i)}")
    print(f"  n_instructions = {f.n_instructions():,}")
    print(f"  sz_w           = {f.sz_w()}")

    plan = extract_level_plan(f)
    print(f"  LP plan: levels={plan.n_compute_levels}  total_ops={plan.total_compute_ops:,}  "
          f"peak_live_slots={plan.peak_live_slots}  "
          f"(shared memory needed = {plan.peak_live_slots * 8 / 1024:.1f} KB)")
    smem_fit = plan.peak_live_slots * 8 <= 96 * 1024  # L40S per-SM smem ~100KB (99KB usable)
    print(f"  fits in untiled LP shared memory (<96KB): {smem_fit}")

    os.makedirs(CUSADI_FUNCTION_DIR, exist_ok=True)
    out_path = os.path.join(CUSADI_FUNCTION_DIR, "a1_constrained_fwd.casadi")
    f.save(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
