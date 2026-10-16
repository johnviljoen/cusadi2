"""CasADi (Pinocchio-casadi) re-implementation of the Panda OSC + CBF model side.

`build_oscbf_function()` returns a casadi.Function

    oscbf_panda(z[14], z_des[18]) -> (u_nom[7], P[7x7], q[7], G[78x7], h[78])

that evaluates exactly what frax + cbfpy evaluate before the QP solve in the
profiled Panda scenario (examples/oscbf_demo.py's DemoOSCBFConfig + OSC controller):
the nominal operational-space torque and cbfpy's (P, q, G, h) for the l1-relaxed
CBF-QP. Kinematics/dynamics come from Pinocchio's CasADi bindings on frax's own
panda.urdf; barriers (joint limits, singularity, obstacle, self-collision, floor)
and their Lie derivatives are built symbolically with casadi.jacobian instead of
jax.jvp. `verify()` checks every output against frax/cbfpy on random states.

This function is what cusadi2 compiles into a batched CUDA kernel in
`bench_cusadi2_swap.py`.
"""
import sys
from pathlib import Path

import numpy as np
import casadi as ca

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

URDF = common.FRAX_DIR / "frax/assets/franka_panda/panda.urdf"
OBSTACLE_POS = np.array([[0.5, 0.5, 0.5]])
OBSTACLE_RAD = np.array([0.3])
SING_TOL = 1e-3
FLOOR_HEIGHT = 0.01
ALPHA = 10.0


def _frax_model_data():
    """Numerical constants taken from the (unmodified) frax Panda object."""
    common.ensure_frax()
    from frax import load_panda

    r = load_panda()
    d = dict(
        q_min=np.asarray(r.actuated_joint_lower_limits), q_max=np.asarray(r.actuated_joint_upper_limits),
        u_max=np.asarray(r.actuated_joint_max_forces), ee_offset=np.asarray(r.ee_offset),
        des_q=np.asarray(r.default_configuration),
        sphere_pos=[np.asarray(p) for p in r.collision_positions],   # per joint, local frame
        sphere_rad=np.asarray(r.flat_collision_radii),
        body_sc_pairs=np.asarray(r.body_sc_pairs), body_sc_tols=np.asarray(r.body_sc_tols),
        root_sc_pairs=np.asarray(r.root_sc_pairs) if r.has_root_collision_data else np.zeros((0, 2), int),
        root_sc_tols=np.asarray(r.root_sc_tols) if r.has_root_collision_data else np.zeros(0),
        root_pos=np.asarray(r.root_collision_positions) if r.has_root_collision_data else np.zeros((0, 3)),
        root_rad=np.asarray(r.root_collision_radii) if r.has_root_collision_data else np.zeros(0),
    )
    return r, d


def build_oscbf_function(name="oscbf_panda", cse=True):
    import pinocchio as pin
    import pinocchio.casadi as cpin

    robot, D = _frax_model_data()
    model = pin.buildModelFromUrdf(str(URDF))
    assert model.nq == 7, model.nq
    j7 = model.getJointId("panda_joint7")
    ee_fid = model.addFrame(pin.Frame("ee", j7, model.getFrameId("panda_link7"), pin.SE3(D["ee_offset"]), pin.FrameType.OP_FRAME))
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    nv = 7

    z = ca.SX.sym("z", 2 * nv)
    z_des = ca.SX.sym("z_des", 18)
    q, v = z[:nv], z[nv:]
    des_pos = z_des[:3]
    des_rot = ca.reshape(z_des[3:12], 3, 3).T   # row-major (3,3) as jnp.reshape does
    des_vel, des_omega = z_des[12:15], z_des[15:18]

    # ---- kinematics / dynamics (same quantities as frax.torque_control_matrices) ----
    cpin.forwardKinematics(cmodel, cdata, q, v)
    cpin.updateFramePlacements(cmodel, cdata)
    cpin.computeJointJacobians(cmodel, cdata, q)
    M = cpin.crba(cmodel, cdata, q)
    M = ca.triu(M) + ca.triu(M, False).T
    M_inv = ca.solve(M, ca.SX.eye(nv))
    g = cpin.computeGeneralizedGravity(cmodel, cdata, q)
    c = cpin.rnea(cmodel, cdata, q, v, ca.SX.zeros(nv)) - g
    J = cpin.getFrameJacobian(cmodel, cdata, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)  # [Jv; Jw]
    T = cdata.oMf[ee_fid]
    pos, rot = T.translation, T.rotation

    # ---- nominal OSC (oscbf_demo.operational_space_control minus the filter) ----
    kp_task = ca.DM([50.0] * 3 + [20.0] * 3)
    kd_task = ca.DM([20.0] * 3 + [10.0] * 3)
    kp_joint, kd_joint = 10.0, 5.0
    twist = J @ v
    rot_error = -0.5 * (ca.cross(rot[:, 0], des_rot[:, 0]) + ca.cross(rot[:, 1], des_rot[:, 1]) + ca.cross(rot[:, 2], des_rot[:, 2]))
    task_p_error = ca.vertcat(pos - des_pos, rot_error)
    task_d_error = ca.vertcat(twist[:3] - des_vel, twist[3:] - des_omega)
    task_inertia_inv = J @ M_inv @ J.T
    task_inertia = ca.solve(task_inertia_inv, ca.SX.eye(6))
    J_bar = M_inv @ J.T @ task_inertia
    task_accel = -kp_task * task_p_error - kd_task * task_d_error
    tau = J.T @ (task_inertia @ task_accel) + g + c
    NT = ca.SX.eye(nv) - J.T @ J_bar.T
    joint_accel = -kp_joint * (q - ca.DM(D["des_q"])) - kd_joint * v
    u_nom = tau + NT @ (M @ joint_accel)

    # ---- cbfpy / cbf_utils: f, g, P, q ----
    f_z = ca.vertcat(v, -M_inv @ (g + c))
    g_z = ca.vertcat(ca.SX.zeros(nv, nv), M_inv)
    P = NT.T @ M_inv.T @ M_inv @ NT + M_inv.T @ J.T @ J @ M_inv     # W = I
    q_qp = -(P.T @ u_nom)                                           # (-u_des^T P)^T

    # ---- DemoOSCBFConfig.h_2 ----
    h_jl = ca.vertcat(ca.DM(D["q_max"]) - q, q - ca.DM(D["q_min"]))
    h_sing = ca.sqrt(ca.det(J @ J.T)) - SING_TOL                     # prod(sigma) = sqrt(det(J J^T))
    sph = []
    for ji in range(nv):
        oMi = cdata.oMi[ji + 1]
        for pl in D["sphere_pos"][ji]:
            sph.append(oMi.rotation @ ca.DM(pl) + oMi.translation)
    sph = ca.horzcat(*sph).T                                         # (n_spheres, 3)
    rad = ca.DM(D["sphere_rad"])
    h_obs = []
    for k in range(sph.shape[0]):
        for o in range(OBSTACLE_POS.shape[0]):
            dlt = sph[k, :].T - ca.DM(OBSTACLE_POS[o])
            h_obs.append(ca.norm_2(dlt) - (rad[k] + OBSTACLE_RAD[o]))
    h_obs = ca.vertcat(*h_obs)
    h_floor = sph[:, 2] - rad - FLOOR_HEIGHT
    h_sc = []
    for (a, b), tol in zip(D["body_sc_pairs"], D["body_sc_tols"]):
        h_sc.append(ca.norm_2(sph[b, :] - sph[a, :]) - rad[a] - rad[b] - tol)
    for (ri, bi), tol in zip(D["root_sc_pairs"], D["root_sc_tols"]):
        h_sc.append(ca.norm_2(sph[bi, :].T - ca.DM(D["root_pos"][ri])) - D["root_rad"][ri] - rad[bi] - tol)
    h_sc = ca.vertcat(*h_sc)
    h2 = ca.vertcat(h_jl, h_sing, h_obs, h_sc, h_floor)

    # ---- cbfpy CBF.h / h_and_Lfh / Lgh / G_qp / h_qp ----
    # cbfpy evaluates these with jax.jvp (forward-mode directional derivatives); ca.jtimes is
    # the CasADi equivalent. (Forming the full ca.jacobian of h and multiplying gives the same
    # numbers but a 6x larger and 7x deeper expression graph; see results/cusadi2_swap.md.)
    h = ca.jtimes(h2, z, f_z) + ALPHA * h2          # RD2 -> RD1: dh2/dt + alpha_2(h2)
    Lfh = ca.jtimes(h, z, f_z)
    Lgh = ca.jtimes(h, z, g_z)                       # one forward sweep per column of g
    G = ca.vertcat(-Lgh, ca.SX.eye(nv), -ca.SX.eye(nv))
    h_qp = ca.vertcat(ALPHA * h + Lfh, ca.DM(D["u_max"]), ca.DM(D["u_max"]))   # u_min = -u_max

    outs = [u_nom, P, q_qp, G, h_qp]
    if cse:
        outs = ca.cse(outs)                          # common-subexpression elimination
    fn = ca.Function(name, [z, z_des], outs, ["z", "z_des"], ["u_nom", "P", "q", "G", "h"])
    return fn, robot, dict(n_cbf=int(h2.shape[0]), n_rows=int(G.shape[0]), n_spheres=int(sph.shape[0]))


def verify(n=64, seed=0, verbose=True):
    """Compare against frax + cbfpy (JAX, CPU, fp64) on random states."""
    common.ensure_frax()
    import jax
    from panda_problem import PandaProblem

    fn, robot, info = build_oscbf_function()
    prob = PandaProblem("qpax")
    rng = np.random.default_rng(seed)
    qs = common.sample_q(robot, rng, n); vs = common.sample_v(rng, n, 7)
    zs = np.concatenate([qs, vs], 1)
    zdes = np.asarray(prob.desired_from_q(common.sample_q(robot, rng, n)))
    nominal = jax.jit(prob.nominal); qp_data = jax.jit(prob.qp_data)
    worst = {}
    for i in range(n):
        u_c, P_c, q_c, G_c, h_c = [np.asarray(x) for x in fn(zs[i], zdes[i])]
        u_j = np.asarray(nominal(zs[i], zdes[i]))
        P_j, q_j, _, _, G_j, h_j = [np.asarray(x) for x in qp_data(zs[i], u_j)]
        for k, (a, b) in dict(u_nom=(u_c.ravel(), u_j), P=(P_c, P_j), q=(q_c.ravel(), q_j), G=(G_c, G_j), h=(h_c.ravel(), h_j)).items():
            rel = np.abs(a - b).max() / max(1.0, np.abs(b).max())
            worst[k] = max(worst.get(k, 0.0), rel)
    if verbose:
        print("max relative error casadi vs frax+cbfpy over", n, "states:", {k: f"{v:.2e}" for k, v in worst.items()})
        print("sizes:", info, " SX nodes:", fn.n_nodes())
    return worst, info


if __name__ == "__main__":
    common.configure("cpu", x64=True)
    verify()
