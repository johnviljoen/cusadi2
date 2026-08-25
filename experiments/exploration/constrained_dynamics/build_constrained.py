"""
Build the constrained ABA-Jacobian CasADi function (ddq/dq, ddq/dv, ddq/dtau)
for A1 and H1 with contacts active.

A1  : 4 × CONTACT_3D at FL/FR/RL/RR_foot, LOCAL_WORLD_ALIGNED
H1  : 2 × CONTACT_6D at left/right ankle_link, LOCAL_WORLD_ALIGNED

Saves build/casadi_functions/{robot}_constrained_aba_jac.casadi.

Usage:
    python exploration/constrained_dynamics/build_constrained.py --robot a1
    python exploration/constrained_dynamics/build_constrained.py --robot h1
"""

from __future__ import annotations

import argparse
import os
import sys

import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from robot_descriptions.loaders.pinocchio import load_robot_description


ROBOTS = {
    "a1": {
        "description": "a1_description",
        "contacts": [
            ("FL_foot", pin.ContactType.CONTACT_3D),
            ("FR_foot", pin.ContactType.CONTACT_3D),
            ("RL_foot", pin.ContactType.CONTACT_3D),
            ("RR_foot", pin.ContactType.CONTACT_3D),
        ],
    },
    "h1": {
        "description": "h1_description",
        "contacts": [
            ("left_ankle_link",  pin.ContactType.CONTACT_6D),
            ("right_ankle_link", pin.ContactType.CONTACT_6D),
        ],
    },
}


def build(robot: str) -> ca.Function:
    cfg = ROBOTS[robot]
    rp = load_robot_description(cfg["description"], root_joint=pin.JointModelFreeFlyer())
    model = rp.model
    print(f"[build] {robot}: nq={model.nq}  nv={model.nv}  njoints={model.njoints}")

    pcontact_models = []
    for frame_name, ctype in cfg["contacts"]:
        if not model.existFrame(frame_name):
            raise RuntimeError(f"frame {frame_name!r} not in model")
        fid = model.getFrameId(frame_name)
        pj = model.frames[fid].parentJoint
        placement = model.frames[fid].placement
        cm = pin.RigidConstraintModel(
            ctype, model, pj, placement, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        pcontact_models.append(cm)
        print(f"  contact @ {frame_name:<20s} parentJoint={pj}  "
              f"type={'CONTACT_3D' if ctype == pin.ContactType.CONTACT_3D else 'CONTACT_6D'}")

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
    cpin.computeConstraintDynamicsDerivatives(
        cmodel, cdata, ccontact_models, ccontact_data, prox
    )

    f = ca.Function(
        f"{robot}_constrained_aba_jac",
        [q_sym, v_sym, tau_sym],
        [cdata.ddq_dq, cdata.ddq_dv, cdata.ddq_dtau],
        ["q", "v", "tau"],
        ["ddq_dq", "ddq_dv", "ddq_dtau"],
    )
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=list(ROBOTS.keys()), required=True)
    ap.add_argument("--out_dir", default="build/casadi_functions")
    args = ap.parse_args()

    f = build(args.robot)
    print(f"\n[build] function: {f.name()}")
    print(f"  n_in  = {f.n_in()}   ({', '.join(f.name_in(i) for i in range(f.n_in()))})")
    print(f"  n_out = {f.n_out()}  ({', '.join(f.name_out(i) for i in range(f.n_out()))})")
    for i in range(f.n_out()):
        print(f"    {f.name_out(i)}: shape={f.sparsity_out(i).shape}  nnz={f.nnz_out(i)}")
    print(f"  n_instructions = {f.n_instructions():,}")
    print(f"  sz_w           = {f.sz_w()}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.robot}_constrained_aba_jac.casadi")
    f.save(out_path)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
