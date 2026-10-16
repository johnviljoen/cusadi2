"""CasADi model of the Panda backup-trajectory CBF safety filter (model side only).

    build_backup_function(N) -> casadi.Function

        backup_panda_N{N}(z[14], z_des[18]) -> (u_nom[7], P[7x7], q[7], G[(78+64N)x7], h[78+64N])

The QP is cbfpy's l1-relaxed CBF-QP with the same objective (P, q) and torque-bound
rows as the one-step filter in profiling/panda_casadi.py, plus the predictive rows:

  * k = 0   the 64 relative-degree-2 rows exactly as cbfpy builds them (h = dh2/dt + alpha h2,
            row: -Lgh u <= alpha h + Lfh);
  * k = 1..N  the state is rolled forward under the backup policy
            u_b(z) = sat(g(q) - M(q) v / tau) (gravity compensation plus inertia-shaped damping,
            every joint's velocity decays with time constant tau; clipped to the torque limits;
            constant per-joint damping is unstable at dt = 20 ms on the wrist joints, and a tanh
            saturation under-compensates gravity enough for the arm to sag) with semi-implicit
            Euler at dt, and every barrier at the predicted state, H_k(z) = h2(phi_k(z)), is a
            relative-degree-1 constraint on the current torque through the flow:
            -LgH_k u <= alpha H_k + LfH_k, with LfH_k = dH_k/dz f(z), LgH_k = dH_k/dz g(z).

Row order: [k=0 rows (64); k=1..N rows (64 each); u <= u_max (7); -u <= u_max (7)], so at N = 0
the function is the verified one-step model.  All Lie derivatives are one forward-mode sweep
(ca.jtimes with the 8 seeds [f, g]) through the whole rollout graph, as jax.jvp does on the
frax side (backup_jax.py); the mass-matrix solves inside the rollout are a hand-unrolled
Cholesky (CasADi's symbolic solve is ~15x more nodes).  verify.py checks this against frax +
cbfpy (N = 0) and against the JAX rollout (N > 0).
"""
import sys
from pathlib import Path

import numpy as np
import casadi as ca

HERE = Path(__file__).resolve().parent
PROFILING = HERE.parent.parent / "profiling"
sys.path.insert(0, str(PROFILING))
import common  # noqa: E402
from panda_casadi import URDF, OBSTACLE_POS, OBSTACLE_RAD, SING_TOL, FLOOR_HEIGHT, ALPHA, _frax_model_data  # noqa: E402

DT = 0.02          # rollout step = control period at 50 Hz
TAU = 0.1          # backup braking time constant [s]: u_b = clip(g - M v / tau, -u_max, u_max)
NV = 7


def chol_solve(M, B):
    """x = M^-1 B for symmetric positive definite SX M (n x n), B (n x m), by unrolled Cholesky."""
    n = M.shape[0]
    L = [[None] * n for _ in range(n)]
    for j in range(n):
        s = M[j, j]
        for k in range(j):
            s = s - L[j][k] * L[j][k]
        L[j][j] = ca.sqrt(s)
        for i in range(j + 1, n):
            s = M[i, j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
            L[i][j] = s / L[j][j]
    cols = []
    for c in range(B.shape[1]):
        y = [None] * n
        for i in range(n):
            s = B[i, c]
            for k in range(i):
                s = s - L[i][k] * y[k]
            y[i] = s / L[i][i]
        x = [None] * n
        for i in reversed(range(n)):
            s = y[i]
            for k in range(i + 1, n):
                s = s - L[k][i] * x[k]
            x[i] = s / L[i][i]
        cols.append(ca.vertcat(*x))
    return ca.horzcat(*cols)


def build_backup_function(N, dt=DT, tau=TAU, name=None, cse=True, chol=True):
    import pinocchio as pin
    import pinocchio.casadi as cpin

    robot, D = _frax_model_data()
    model = pin.buildModelFromUrdf(str(URDF))
    assert model.nq == NV, model.nq
    j7 = model.getJointId("panda_joint7")
    ee_fid = model.addFrame(pin.Frame("ee", j7, model.getFrameId("panda_link7"), pin.SE3(D["ee_offset"]), pin.FrameType.OP_FRAME))
    cmodel = cpin.Model(model)
    nv = NV
    u_max = ca.DM(D["u_max"])
    rad = ca.DM(D["sphere_rad"])
    q_min, q_max, des_q = ca.DM(D["q_min"]), ca.DM(D["q_max"]), ca.DM(D["des_q"])
    I7 = ca.SX.eye(nv)

    def kin(q, v):
        """Everything the model needs at one state (fresh Pinocchio data each call)."""
        cd = cmodel.createData()
        cpin.forwardKinematics(cmodel, cd, q, v)
        cpin.updateFramePlacements(cmodel, cd)
        cpin.computeJointJacobians(cmodel, cd, q)
        M = cpin.crba(cmodel, cd, q)
        M = ca.triu(M) + ca.triu(M, False).T
        g = cpin.computeGeneralizedGravity(cmodel, cd, q)
        c = cpin.rnea(cmodel, cd, q, v, ca.SX.zeros(nv)) - g
        J = cpin.getFrameJacobian(cmodel, cd, ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        T = cd.oMf[ee_fid]
        sph = []
        for ji in range(nv):
            oMi = cd.oMi[ji + 1]
            for pl in D["sphere_pos"][ji]:
                sph.append(oMi.rotation @ ca.DM(pl) + oMi.translation)
        sph = ca.horzcat(*sph).T                                     # (n_spheres, 3)
        return dict(M=M, g=g, c=c, J=J, pos=T.translation, rot=T.rotation, sph=sph)

    def barriers(q, K):
        """DemoOSCBFConfig.h_2 at (q, kinematics K)."""
        J, sph = K["J"], K["sph"]
        h_jl = ca.vertcat(q_max - q, q - q_min)
        h_sing = ca.sqrt(ca.det(J @ J.T)) - SING_TOL
        h_obs = []
        for k in range(sph.shape[0]):
            for o in range(OBSTACLE_POS.shape[0]):
                dlt = sph[k, :].T - ca.DM(OBSTACLE_POS[o])
                h_obs.append(ca.norm_2(dlt) - (rad[k] + OBSTACLE_RAD[o]))
        h_floor = sph[:, 2] - rad - FLOOR_HEIGHT
        h_sc = []
        for (a, b), tol in zip(D["body_sc_pairs"], D["body_sc_tols"]):
            h_sc.append(ca.norm_2(sph[b, :] - sph[a, :]) - rad[a] - rad[b] - tol)
        for (ri, bi), tol in zip(D["root_sc_pairs"], D["root_sc_tols"]):
            h_sc.append(ca.norm_2(sph[bi, :].T - ca.DM(D["root_pos"][ri])) - D["root_rad"][ri] - rad[bi] - tol)
        return ca.vertcat(h_jl, h_sing, ca.vertcat(*h_obs), ca.vertcat(*h_sc), h_floor)

    z = ca.SX.sym("z", 2 * nv)
    z_des = ca.SX.sym("z_des", 18)
    q, v = z[:nv], z[nv:]
    des_pos = z_des[:3]
    des_rot = ca.reshape(z_des[3:12], 3, 3).T
    des_vel, des_omega = z_des[12:15], z_des[15:18]

    # ---- current state: nominal OSC, cbfpy P / q / f / g (as profiling/panda_casadi.py) ----
    K0 = kin(q, v)
    M, g, c, J = K0["M"], K0["g"], K0["c"], K0["J"]
    M_inv = chol_solve(M, I7) if chol else ca.solve(M, I7)
    kp_task = ca.DM([50.0] * 3 + [20.0] * 3)
    kd_task = ca.DM([20.0] * 3 + [10.0] * 3)
    kp_joint, kd_joint = 10.0, 5.0
    twist = J @ v
    rot, pos = K0["rot"], K0["pos"]
    rot_error = -0.5 * (ca.cross(rot[:, 0], des_rot[:, 0]) + ca.cross(rot[:, 1], des_rot[:, 1]) + ca.cross(rot[:, 2], des_rot[:, 2]))
    task_p_error = ca.vertcat(pos - des_pos, rot_error)
    task_d_error = ca.vertcat(twist[:3] - des_vel, twist[3:] - des_omega)
    task_inertia_inv = J @ M_inv @ J.T
    task_inertia = ca.solve(task_inertia_inv, ca.SX.eye(6))
    J_bar = M_inv @ J.T @ task_inertia
    task_accel = -kp_task * task_p_error - kd_task * task_d_error
    tau_nom = J.T @ (task_inertia @ task_accel) + g + c
    NT = I7 - J.T @ J_bar.T
    joint_accel = -kp_joint * (q - des_q) - kd_joint * v
    u_nom = tau_nom + NT @ (M @ joint_accel)

    f_z = ca.vertcat(v, -M_inv @ (g + c))
    g_z = ca.vertcat(ca.SX.zeros(nv, nv), M_inv)
    P = NT.T @ M_inv.T @ M_inv @ NT + M_inv.T @ J.T @ J @ M_inv
    q_qp = -(P.T @ u_nom)

    # ---- k = 0: cbfpy's relative-degree-2 rows ----
    h2_0 = barriers(q, K0)
    h0 = ca.jtimes(h2_0, z, f_z) + ALPHA * h2_0

    # ---- k = 1..N: backup rollout ----
    H = []
    zk = z
    for k in range(N):
        qk, vk = zk[:nv], zk[nv:]
        Kk = K0 if k == 0 else kin(qk, vk)
        u_b = ca.fmin(ca.fmax(Kk["g"] - Kk["M"] @ vk / tau, -u_max), u_max)
        a = chol_solve(Kk["M"], u_b - Kk["g"] - Kk["c"]) if chol else ca.solve(Kk["M"], u_b - Kk["g"] - Kk["c"])
        vk1 = vk + dt * a
        qk1 = qk + dt * vk1
        zk = ca.vertcat(qk1, vk1)
        H.append(barriers(qk1, kin(qk1, vk1)))
    hall = ca.vertcat(h0, *H)                                        # (64 (N+1),)
    seeds = ca.horzcat(f_z, g_z)                                     # (14, 8): [f, g_1..g_7]
    Lh = ca.jtimes(hall, z, seeds)                                   # one forward sweep, 8 directions
    Lfh, Lgh = Lh[:, 0], Lh[:, 1:]
    G = ca.vertcat(-Lgh, I7, -I7)
    h_qp = ca.vertcat(ALPHA * hall + Lfh, u_max, u_max)              # u_min = -u_max

    outs = [u_nom, P, q_qp, G, h_qp]
    if cse:
        outs = ca.cse(outs)
    name = name or f"backup_panda_N{N}"
    fn = ca.Function(name, [z, z_des], outs, ["z", "z_des"], ["u_nom", "P", "q", "G", "h"])
    info = dict(N=N, dt=dt, tau=tau, n_cbf=int(hall.shape[0]), n_cbf_per_step=int(h2_0.shape[0]), n_rows=int(G.shape[0]),
                n_spheres=int(K0["sph"].shape[0]), sx_nodes=fn.n_nodes(), chol=chol)
    return fn, robot, info


if __name__ == "__main__":
    import time
    for N in (0, 1, 4):
        t0 = time.perf_counter()
        fn, _, info = build_backup_function(N)
        print(N, info, f"{time.perf_counter() - t0:.1f} s")
