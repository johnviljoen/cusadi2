"""The cupboard safe-MPPI model functions on CasADi (the cusadi2 evaluator; CusADi-1 and CasADi codegen compile the same graphs).

    model(x[16], Y[N_KNOT*7], rp[3], par[NPAR])  -> (u_nom[7], G[N_CBF x 7], h[N_CBF])
    integrate(x[16], u[7], rp[3], par[NPAR])     -> x1[16]

State x = [q (7), v (7), tau (plan clock), c (accumulated cost)]. One control period DT = 20 ms:

  1. joint target  q_des = sum_j w_j(tau) Y_j, hat-function interpolation of the knots (knot j at j * knot_dt)
  2. nominal torque (inertia-shaped PD + gravity/Coriolis compensation)
         u_nom = clip(M(q) (kp (q_des - q) - kd v) + g(q) + c(q, v), -u_max, u_max)
  3. cbfpy's relative-degree-2 CBF rows of the cupboard scene's N_CBF barriers h2(q) (scene.py) at the current state:
         h = Lf h2 + alpha h2,   G = -Lg h,   h_q = alpha h + Lf h,   so that  G u <= h_q
     (CasADi jtimes, forward-mode directional derivatives, as cbfpy builds them)
  -- the projection of u_nom onto {G u <= h_q, |u| <= u_max} is the driver's shared kernel (k_project in safe_mppi.cu,
     reference project_greedy_np below), not part of either function --
  4. semi-implicit Euler: a = M^-1 (u - g - c), v1 = v + DT a, q1 = q + DT v1
  5. running cost accumulated in the state: c1 = c + DT * l(x, u),
         l = w_pos |p_ee - rp|^2 + w_rot |e_rot|^2 + w_v |v|^2 + w_u |u|^2
             + w_fun s(p_ee) |(y, z)_ee - axis|^2 + w_wrist s(p_wr) |(y, z)_wr - axis|^2
     (e_rot against the pre-grasp orientation scene.R_PREGRASP, rp the pre-grasp TCP position; the last two are the
     cost-shaping terms of the ladder's rungs 3 / 4, s a sigmoid in x switching on near the opening, p_wr the point
     WRIST_BACK behind the TCP along the tool axis; w_fun = w_wrist = 0 at rungs 1 / 2), + w_q |q - Q_PREGRASP|^2 (rung 5).

`par` holds the gains and weights so that tuning never rebuilds the plans.
"""
import sys
from pathlib import Path

import numpy as np
import casadi as ca

HERE = Path(__file__).resolve().parent
BACKUP = HERE.parent / "panda_backup_cbf"
PROFILING = HERE.parent.parent / "profiling"
for p in (HERE, BACKUP, PROFILING):
    sys.path.insert(0, str(p))
from panda_casadi import ALPHA  # noqa: E402
from backup_casadi import NV, chol_solve  # noqa: E402
from backup_staged import _Model  # noqa: E402
import scene  # noqa: E402

DT = 0.02                       # control period = rollout step
N_KNOT = 4
NX = 2 * NV + 2
I_TAU, I_COST = 2 * NV, 2 * NV + 1
ND = N_KNOT * NV
PROJ_ITERS = 48          # greedy iterations of the shared projection kernel (inherited from the sphere scene)
N_CBF = scene.N_ROWS
DES_ROT = scene.R_PREGRASP       # tool into the cupboard (+x), palm up, fingers along -y (the pre-grasp orientation)
PARAMS = [("kp", 50.0), ("kd", 20.0), ("w_pos", 50.0), ("w_rot", 2.0), ("w_v", 1.0), ("w_u", 1e-4), ("w_fun", 0.0), ("w_wrist", 0.0), ("w_q", 0.0)]
PAR_DEFAULT = np.array([v for _, v in PARAMS])
NPAR = len(PARAMS)


def knot_dt(T):
    """Knot spacing for a horizon of T control periods: the last knot sits at the horizon end."""
    return T * DT / (N_KNOT - 1)


def interp_weights(tau, kdt):
    s = ca.fmin(ca.fmax(tau / kdt, 0), N_KNOT - 1)
    return [ca.fmax(0, 1 - ca.fabs(s - j)) for j in range(N_KNOT)]


def _nominal_and_rows(m, q, v, tau, Y, par, T, rows):
    nv, I7 = NV, ca.SX.eye(NV)
    P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
    z = ca.vertcat(q, v)
    K = m.kin(q, v)
    M, g, cor = K["M"], K["g"], K["c"]
    w = interp_weights(tau, knot_dt(T))
    q_des = sum(w[j] * Y[j * nv:(j + 1) * nv] for j in range(N_KNOT))
    u_nom = M @ (P["kp"] * (q_des - q) - P["kd"] * v) + g + cor
    u_nom = ca.fmin(ca.fmax(u_nom, -m.u_max), m.u_max)
    M_inv = chol_solve(M, I7)
    f_z = ca.vertcat(v, -M_inv @ (g + cor)); g_z = ca.vertcat(ca.SX.zeros(nv, nv), M_inv)
    assert rows == "all", "the cupboard scene has one row set"
    h2 = scene.barriers_casadi(m, q, K)
    h = ca.jtimes(h2, z, f_z) + ALPHA * h2
    L = ca.jtimes(h, z, ca.horzcat(f_z, g_z))
    return u_nom, -L[:, 1:], ALPHA * h + L[:, 0], K


def build_model_function(T, name="safe_model", cse=True, rows="all"):
    """model(x[16], Y[28], rp[3], par[6]) -> (u_nom[7], G[n_cbf x 7], h[n_cbf]): the nominal PD torque and cbfpy's
    one-step rows at the current state. The projection onto {G u <= h, |u| <= u_max} is the MPPI's shared kernel."""
    m = _Model()
    q = ca.SX.sym("q", NV); v = ca.SX.sym("v", NV); tau = ca.SX.sym("tau"); c = ca.SX.sym("c")
    Y = ca.SX.sym("Y", ND); rp = ca.SX.sym("rp", 3); par = ca.SX.sym("par", NPAR)
    x = ca.vertcat(q, v, tau, c)
    u_nom, G, hq, _ = _nominal_and_rows(m, q, v, tau, Y, par, T, rows)
    outs = [u_nom, G, hq]
    if cse:
        outs = ca.cse(outs)
    fn = ca.Function(name, [x, Y, rp, par], outs, ["x", "Y", "rp", "par"], ["u_nom", "G", "h"])
    return fn, m.robot, dict(T=T, rows=rows, n_cbf=int(G.shape[0]), sx_nodes=fn.n_nodes())


def build_integrate_function(dt=DT, name="safe_integrate", cse=True):
    """integrate(x[16], u[7], rp[3], par[6]) -> x1[16]: semi-implicit Euler under the (already filtered) torque u and
    the running cost accumulated into the state."""
    m = _Model()
    q = ca.SX.sym("q", NV); v = ca.SX.sym("v", NV); tau = ca.SX.sym("tau"); c = ca.SX.sym("c")
    u = ca.SX.sym("u", NV); rp = ca.SX.sym("rp", 3); par = ca.SX.sym("par", NPAR)
    P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
    x = ca.vertcat(q, v, tau, c)
    K = m.kin(q, v)
    M, g, cor = K["M"], K["g"], K["c"]
    a = chol_solve(M, u - g - cor)
    v1 = v + dt * a; q1 = q + dt * v1
    rot, pos = K["rot"], K["pos"]
    des_rot = ca.DM(DES_ROT)
    e_rot = -0.5 * (ca.cross(rot[:, 0], des_rot[:, 0]) + ca.cross(rot[:, 1], des_rot[:, 1]) + ca.cross(rot[:, 2], des_rot[:, 2]))
    ell = P["w_pos"] * ca.sumsqr(pos - rp) + P["w_rot"] * ca.sumsqr(e_rot) + P["w_v"] * ca.sumsqr(v) + P["w_u"] * ca.sumsqr(u)
    # cost shaping (scene.py): funnel on the TCP and on the wrist point, switched on by a sigmoid in x near the opening
    axis = ca.DM(scene.AXIS_YZ); wrist = pos - scene.WRIST_BACK * rot[:, 2]
    s_tcp = 1.0 / (1.0 + ca.exp(-(pos[0] - scene.FUNNEL_X_ON) / scene.FUNNEL_W)); s_wr = 1.0 / (1.0 + ca.exp(-(wrist[0] - scene.FUNNEL_X_ON) / scene.FUNNEL_W))
    ell = ell + P["w_fun"] * s_tcp * ca.sumsqr(pos[1:3] - axis) + P["w_wrist"] * s_wr * ca.sumsqr(wrist[1:3] - axis)
    ell = ell + P["w_q"] * ca.sumsqr(q - ca.DM(scene.Q_PREGRASP))                      # joint-space reference (rung 5)
    x1 = ca.vertcat(q1, v1, tau + dt, c + dt * ell)
    outs = [x1]
    if cse:
        outs = ca.cse(outs)
    fn = ca.Function(name, [x, u, rp, par], outs, ["x", "u", "rp", "par"], ["x1"])
    return fn, m.robot, dict(dt=dt, sx_nodes=fn.n_nodes())


def project_greedy_np(u0, G, h, u_max, iters):
    """numpy reference of the MPPI's projection kernel (k_project in safe_mppi.cu): `iters` times project onto the
    most violated row of G u <= h (signed distance), then clip to the torque box."""
    import numpy as np
    u = np.array(u0, dtype=np.float64); G = np.asarray(G, dtype=np.float64); h = np.asarray(h, dtype=np.float64)
    gg = (G * G).sum(1) + 1e-12; inv = 1.0 / np.sqrt(gg)
    for _ in range(iters):
        r = (G @ u - h) * inv
        i = int(np.argmax(r))                      # first index on ties, like the kernel
        if r[i] > 0:
            u = u - (G[i] @ u - h[i]) / gg[i] * G[i]
        u = np.clip(u, -u_max, u_max)
    return u


def build_aux_function(name="safe_aux"):
    """(x) -> (h2[64], p_ee[3]) for diagnostics (barrier minima, waypoint distance) on the CPU."""
    m = _Model()
    q = ca.SX.sym("q", NV); v = ca.SX.sym("v", NV); tau = ca.SX.sym("tau"); c = ca.SX.sym("c")
    x = ca.vertcat(q, v, tau, c)
    K = m.kin(q, v)
    return ca.Function(name, [x], [scene.barriers_casadi(m, q, K), K["pos"]], ["x"], ["h2", "p_ee"])


if __name__ == "__main__":
    fn, robot, info = build_model_function(T=100)
    print(info)
