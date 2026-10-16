"""Staged CasADi model of the backup-CBF filter: a head function plus a per-step function.

A single CasADi function for the whole rollout (backup_casadi.build_backup_function) does not
suit cusadi2's level kernel beyond N = 8: the ASAP level schedule keeps too many values live
(N = 12 only fits shared memory by windowing into 24k levels, N = 16 does not fit at all).
The staged form keeps every plan small and independent of N; the horizon is the number of
step launches.

    head(z[14], z_des[18]) -> (u_nom[7], P[7x7], q[7], G0[78x7], h0[78], T0[14x8])
        current-state quantities exactly as the monolithic function: nominal OSC torque,
        cbfpy's P, q, the 64 relative-degree-2 rows at k = 0 and the 14 torque-bound rows
        (G0 = [-Lgh; I; -I], h0 = [alpha h + Lfh; u_max; u_max]), and the 8 tangent seeds
        T0 = [f(z), g(z)] (14 x 8) that the rollout propagates.

    step(z_k[14], T_k[14x8]) -> (z_{k+1}[14], T_{k+1}[14x8], Gk[64x7], hk[64])
        one semi-implicit Euler step under the backup policy, its forward-mode push-forward
        of the 8 tangents, T_{k+1} = dF/dz(z_k) T_k, and the 64 predictive rows at the new
        state: [LfH, LgH] = dh2/dz(z_{k+1}) T_{k+1}, Gk = -LgH, hk = alpha H + LfH.

Composing head + N steps reproduces the monolithic function's rows (verify_staged()).
"""
import sys
from pathlib import Path

import numpy as np
import casadi as ca

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
from panda_casadi import URDF, OBSTACLE_POS, OBSTACLE_RAD, SING_TOL, FLOOR_HEIGHT, ALPHA, _frax_model_data  # noqa: E402
from backup_casadi import DT, TAU, NV, chol_solve  # noqa: E402


class _Model:
    def __init__(self):
        import pinocchio as pin
        import pinocchio.casadi as cpin
        self.pin, self.cpin = pin, cpin
        self.robot, D = _frax_model_data()
        self.D = D
        model = pin.buildModelFromUrdf(str(URDF))
        j7 = model.getJointId("panda_joint7")
        self.ee_fid = model.addFrame(pin.Frame("ee", j7, model.getFrameId("panda_link7"), pin.SE3(D["ee_offset"]), pin.FrameType.OP_FRAME))
        self.cmodel = cpin.Model(model)
        self.u_max = ca.DM(D["u_max"]); self.rad = ca.DM(D["sphere_rad"])
        self.q_min, self.q_max, self.des_q = ca.DM(D["q_min"]), ca.DM(D["q_max"]), ca.DM(D["des_q"])

    def kin(self, q, v):
        cpin, pin, nv = self.cpin, self.pin, NV
        cd = self.cmodel.createData()
        cpin.forwardKinematics(self.cmodel, cd, q, v)
        cpin.updateFramePlacements(self.cmodel, cd)
        cpin.computeJointJacobians(self.cmodel, cd, q)
        M = cpin.crba(self.cmodel, cd, q)
        M = ca.triu(M) + ca.triu(M, False).T
        g = cpin.computeGeneralizedGravity(self.cmodel, cd, q)
        c = cpin.rnea(self.cmodel, cd, q, v, ca.SX.zeros(nv)) - g
        J = cpin.getFrameJacobian(self.cmodel, cd, self.ee_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        T = cd.oMf[self.ee_fid]
        sph = []
        for ji in range(nv):
            oMi = cd.oMi[ji + 1]
            for pl in self.D["sphere_pos"][ji]:
                sph.append(oMi.rotation @ ca.DM(pl) + oMi.translation)
        return dict(M=M, g=g, c=c, J=J, pos=T.translation, rot=T.rotation, sph=ca.horzcat(*sph).T)

    def barriers(self, q, K):
        D, rad = self.D, self.rad
        J, sph = K["J"], K["sph"]
        h_jl = ca.vertcat(self.q_max - q, q - self.q_min)
        h_sing = ca.sqrt(ca.det(J @ J.T)) - SING_TOL
        h_obs = []
        for k in range(sph.shape[0]):
            for o in range(OBSTACLE_POS.shape[0]):
                h_obs.append(ca.norm_2(sph[k, :].T - ca.DM(OBSTACLE_POS[o])) - (rad[k] + OBSTACLE_RAD[o]))
        h_floor = sph[:, 2] - rad - FLOOR_HEIGHT
        h_sc = []
        for (a, b), tol in zip(D["body_sc_pairs"], D["body_sc_tols"]):
            h_sc.append(ca.norm_2(sph[b, :] - sph[a, :]) - rad[a] - rad[b] - tol)
        for (ri, bi), tol in zip(D["root_sc_pairs"], D["root_sc_tols"]):
            h_sc.append(ca.norm_2(sph[bi, :].T - ca.DM(D["root_pos"][ri])) - D["root_rad"][ri] - rad[bi] - tol)
        return ca.vertcat(h_jl, h_sing, ca.vertcat(*h_obs), ca.vertcat(*h_sc), h_floor)

    def step_state(self, z, dt, tau):
        """Semi-implicit Euler step under the backup brake u_b = clip(g - M v / tau)."""
        q, v = z[:NV], z[NV:]
        K = self.kin(q, v)
        u_b = ca.fmin(ca.fmax(K["g"] - K["M"] @ v / tau, -self.u_max), self.u_max)
        a = chol_solve(K["M"], u_b - K["g"] - K["c"])
        v1 = v + dt * a
        return ca.vertcat(q + dt * v1, v1)


def build_head_function(name="backup_head", cse=True):
    m = _Model()
    nv, I7 = NV, ca.SX.eye(NV)
    z = ca.SX.sym("z", 2 * nv); z_des = ca.SX.sym("z_des", 18)
    q, v = z[:nv], z[nv:]
    des_pos = z_des[:3]; des_rot = ca.reshape(z_des[3:12], 3, 3).T; des_vel, des_omega = z_des[12:15], z_des[15:18]
    K0 = m.kin(q, v)
    M, g, c, J = K0["M"], K0["g"], K0["c"], K0["J"]
    M_inv = chol_solve(M, I7)
    kp_task = ca.DM([50.0] * 3 + [20.0] * 3); kd_task = ca.DM([20.0] * 3 + [10.0] * 3); kp_joint, kd_joint = 10.0, 5.0
    twist = J @ v; rot, pos = K0["rot"], K0["pos"]
    rot_error = -0.5 * (ca.cross(rot[:, 0], des_rot[:, 0]) + ca.cross(rot[:, 1], des_rot[:, 1]) + ca.cross(rot[:, 2], des_rot[:, 2]))
    task_p_error = ca.vertcat(pos - des_pos, rot_error); task_d_error = ca.vertcat(twist[:3] - des_vel, twist[3:] - des_omega)
    task_inertia_inv = J @ M_inv @ J.T; task_inertia = ca.solve(task_inertia_inv, ca.SX.eye(6)); J_bar = M_inv @ J.T @ task_inertia
    task_accel = -kp_task * task_p_error - kd_task * task_d_error
    tau_nom = J.T @ (task_inertia @ task_accel) + g + c
    NT = I7 - J.T @ J_bar.T
    u_nom = tau_nom + NT @ (M @ (-kp_joint * (q - m.des_q) - kd_joint * v))
    f_z = ca.vertcat(v, -M_inv @ (g + c)); g_z = ca.vertcat(ca.SX.zeros(nv, nv), M_inv)
    P = NT.T @ M_inv.T @ M_inv @ NT + M_inv.T @ J.T @ J @ M_inv
    q_qp = -(P.T @ u_nom)
    h2_0 = m.barriers(q, K0)
    h0 = ca.jtimes(h2_0, z, f_z) + ALPHA * h2_0
    T0 = ca.horzcat(f_z, g_z)
    L = ca.jtimes(h0, z, T0)
    G0 = ca.vertcat(-L[:, 1:], I7, -I7)
    hq0 = ca.vertcat(ALPHA * h0 + L[:, 0], m.u_max, m.u_max)
    outs = [u_nom, P, q_qp, G0, hq0, T0]
    if cse: outs = ca.cse(outs)
    fn = ca.Function(name, [z, z_des], outs, ["z", "z_des"], ["u_nom", "P", "q", "G0", "h0", "T0"])
    return fn, m.robot, dict(n_rows0=int(G0.shape[0]), n_cbf_per_step=int(h2_0.shape[0]), sx_nodes=fn.n_nodes())


def build_step_function(dt=DT, tau=TAU, name="backup_step", cse=True):
    m = _Model()
    nv = NV
    z = ca.SX.sym("z", 2 * nv); T = ca.SX.sym("T", 2 * nv, 8)
    z1 = m.step_state(z, dt, tau)
    T1 = ca.jtimes(z1, z, T)                        # dF/dz T, 8 directions in one sweep
    K1 = m.kin(z1[:nv], z1[nv:])
    H = m.barriers(z1[:nv], K1)                     # h2 at the predicted state
    L = ca.jtimes(H, z, T)                          # dh2/dz1 dF/dz T  (= dh2/dz1 T1 by the chain rule)
    Gk = -L[:, 1:]
    hk = ALPHA * H + L[:, 0]
    outs = [z1, T1, Gk, hk]
    if cse: outs = ca.cse(outs)
    fn = ca.Function(name, [z, T], outs, ["z", "T"], ["z1", "T1", "G", "h"])
    return fn, dict(dt=dt, tau=tau, n_cbf_per_step=int(H.shape[0]), sx_nodes=fn.n_nodes())


def compose(head, step, N, z, z_des):
    """Numerically evaluate head + N steps and assemble (u_nom, P, q, G, h) in the monolithic row order."""
    u, P, q, G0, h0, T = [np.asarray(x) for x in head(z, z_des)]
    h0 = h0.ravel(); n0 = 64
    Gs, hs = [G0[:n0]], [h0[:n0]]
    zk = np.asarray(z)
    for _ in range(N):
        zk, T, Gk, hk = [np.asarray(x) for x in step(zk, T)]
        Gs.append(Gk); hs.append(hk.ravel())
    Gs.append(G0[n0:]); hs.append(h0[n0:])
    return u.ravel(), P, q.ravel(), np.concatenate(Gs), np.concatenate(hs)


def verify_staged(N=4, n=8, seed=0):
    import common
    from backup_casadi import build_backup_function
    head, robot, hi = build_head_function(); step, si = build_step_function()
    mono, _, mi = build_backup_function(N)
    print("head:", hi, " step:", si, " mono:", {k: mi[k] for k in ("n_rows", "sx_nodes")}, flush=True)
    rng = np.random.default_rng(seed)
    zs = np.concatenate([common.sample_q(robot, rng, n), common.sample_v(rng, n, 7)], 1)
    zdes = rng.normal(size=(n, 18)); zdes[:, 3:12] = np.eye(3).ravel()
    worst = {}
    for i in range(n):
        a = compose(head, step, N, zs[i], zdes[i])
        b = [np.asarray(x) for x in mono(zs[i], zdes[i])]
        for k, x, y in zip(["u_nom", "P", "q", "G", "h"], a, b):
            y = y.reshape(x.shape)
            worst[k] = max(worst.get(k, 0.0), np.abs(x - y).max() / max(1.0, np.abs(y).max()))
    print(f"staged vs monolithic N={N}:", {k: f"{v:.1e}" for k, v in worst.items()})
    return worst


if __name__ == "__main__":
    import common
    common.configure("cpu", x64=True)
    verify_staged()
