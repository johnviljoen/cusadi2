"""The safe-MPPI rollout step of the Panda as one CasADi function (the cusadi2 evaluator).

    safe_step(x[16], Y[N_KNOT*7], rp[3], par[NPAR]) -> (x1[16], u[7])

State x = [q (7), v (7), tau (plan clock), c (accumulated cost)]. One control period (DT = 20 ms):

  1. joint target  q_des = sum_j w_j(tau) Y_j, linear interpolation of the knots (knot j at j * KNOT_DT)
  2. nominal torque (inertia-shaped PD + gravity/Coriolis compensation, the nominal controller)
         u_nom = clip(M(q) (kp (q_des - q) - kd v) + g(q) + c(q, v), -u_max, u_max)
  3. the one-step CBF rows of frax's demo scene at the current state, exactly cbfpy's relative-degree-2
     rows for the 64 barriers (joint limits, singularity, obstacle, self-collision, floor):
         G = -Lg h,  hq = alpha h + Lf h,   h = Lf h2 + alpha h2
  4. closed-form safety filter: PROJ_ROUNDS rounds of an exact small active-set projection onto the 78 rows
     [G; I; -I] u <= [hq; u_max; u_max]: each round takes the three most violated rows and solves that 3-row
     projection QP exactly (the 8 candidate active sets with closed-form Gram inverses; KKT is sufficient).
     No QP solver, no linear solve beyond the 7 x 7 Cholesky of the forward dynamics that every pipeline shares.
     (projection_study2.py compares it with cbfpy's elastic QP on 20-step rollouts from initially safe states.)
  5. semi-implicit Euler: a = M^-1 (u - g - c), v1 = v + DT a, q1 = q + DT v1
  6. running cost accumulated in the state: c1 = c + DT * l(x, u),
         l = w_pos |p_ee - rp|^2 + w_rot |e_rot|^2 + w_v |v|^2 + w_u |u|^2
     (e_rot against the fixed tool-down orientation of the frax demo).

The function is the whole per-step model; MPPI (sampling, weighting, plant) lives in the driver.
`par` holds the gains and weights so that tuning never rebuilds the plan.
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
from panda_casadi import ALPHA, OBSTACLE_POS  # noqa: E402
from backup_casadi import NV, chol_solve  # noqa: E402
from backup_staged import _Model  # noqa: E402

DT = 0.02                       # control period = rollout step
N_KNOT = 4
NX = 2 * NV + 2
I_TAU, I_COST = 2 * NV, 2 * NV + 1
ND = N_KNOT * NV
PROJ_ROUNDS = 3
PROJ_ITERS = 48          # greedy iterations of the shared projection kernel (study: 32 already matches the QP)
N_CBF = 64
DES_ROT = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])      # tool pointing down (frax demo target)
PARAMS = [("kp", 50.0), ("kd", 20.0), ("w_pos", 50.0), ("w_rot", 2.0), ("w_v", 1.0), ("w_u", 1e-4)]
PAR_DEFAULT = np.array([v for _, v in PARAMS])
NPAR = len(PARAMS)


def knot_dt(T):
    """Knot spacing for a horizon of T control periods: the last knot sits at the horizon end."""
    return T * DT / (N_KNOT - 1)


def interp_weights(tau, kdt):
    s = ca.fmin(ca.fmax(tau / kdt, 0), N_KNOT - 1)
    return [ca.fmax(0, 1 - ca.fabs(s - j)) for j in range(N_KNOT)]


def tree_reduce(xs, op):
    xs = list(xs)
    while len(xs) > 1:
        nxt = [op(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]
        if len(xs) % 2:
            nxt.append(xs[-1])
        xs = nxt
    return xs[0]


def tree_sum(xs):
    """Balanced sum (CasADi's mtimes sums serially, one level per term; the level kernel pays one barrier per level)."""
    return tree_reduce(xs, lambda a, b: a + b)


def tree_max(xs):
    return tree_reduce(xs, ca.fmax)


BIG = 1e30
PROJ_TOL = 1e-7
TIE_EPS = 1e-6
DET_REL = 1e-3          # Gram determinant / product of row norms: sin^2 of the row angles, rejects near-parallel sets
TRUST = 9.0             # |u_S - u|^2 <= TRUST * sum of the single-row projection distances^2 (orthogonal rows give 1)


def project(u0, A, b, rounds=PROJ_ROUNDS):
    """Exact small-active-set projection of u0 onto {u : A u <= b} (A n x 7 as list of lists, b list), `rounds` times.

    Each round takes the three most violated rows (by signed distance), solves the 3-row projection QP exactly by
    enumerating the 8 candidate active sets with closed-form Gram inverses (the KKT conditions of a convex QP are
    sufficient, so a candidate with lambda >= 0 whose other rows are satisfied is the optimum), and falls back to
    the single-row projection if every candidate is degenerate. The box rows are part of A, so no clip is needed
    in between. Written with balanced reductions: about 65 levels per round for cusadi2's level kernel.
    """
    n = len(A)
    inv_norm = [1.0 / ca.sqrt(tree_sum(A[i][j] * A[i][j] for j in range(NV)) + 1e-12) for i in range(n)]
    u = u0
    for _ in range(rounds):
        r = [(tree_sum(A[i][j] * u[j] for j in range(NV)) - b[i]) * inv_norm[i] for i in range(n)]
        # deterministic tie-break (exact ties are structural: box rows at distance 0 when u sits on a bound): lower index wins
        rt = [r[i] - TIE_EPS * i * (1.0 + ca.fabs(r[i])) for i in range(n)]
        sels, ms, rr = [], [], rt
        for _k in range(3):
            m = tree_max(rr)
            sel = [ca.if_else(rr[i] >= m, 1.0, 0.0) for i in range(n)]
            sels.append(sel); ms.append(m)
            rr = [rr[i] - BIG * sel[i] for i in range(n)]
        a = [[tree_sum(sels[k][i] * A[i][j] for i in range(n)) for j in range(NV)] for k in range(3)]
        bb = [tree_sum(sels[k][i] * b[i] for i in range(n)) for k in range(3)]
        g = [[tree_sum(a[k][j] * a[l][j] for j in range(NV)) for l in range(3)] for k in range(3)]   # Gram
        c = [tree_sum(a[k][j] * u[j] for j in range(NV)) - bb[k] for k in range(3)]                   # A_S u - b_S
        def cand(S):
            """(u_S, valid) for the active set S (tuple of row slots)."""
            if len(S) == 1:
                (k,) = S; det = g[k][k]; scale = g[k][k]; lam = [c[k] / det]
            elif len(S) == 2:
                k, l = S; det = g[k][k] * g[l][l] - g[k][l] * g[k][l]; scale = g[k][k] * g[l][l]
                lam = [(g[l][l] * c[k] - g[k][l] * c[l]) / det, (-g[k][l] * c[k] + g[k][k] * c[l]) / det]
            else:
                k, l, m_ = S
                G = [[g[k][k], g[k][l], g[k][m_]], [g[l][k], g[l][l], g[l][m_]], [g[m_][k], g[m_][l], g[m_][m_]]]
                det = (G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1]) - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
                       + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]))
                adj = [[G[1][1] * G[2][2] - G[1][2] * G[2][1], G[0][2] * G[2][1] - G[0][1] * G[2][2], G[0][1] * G[1][2] - G[0][2] * G[1][1]],
                       [G[1][2] * G[2][0] - G[1][0] * G[2][2], G[0][0] * G[2][2] - G[0][2] * G[2][0], G[0][2] * G[1][0] - G[0][0] * G[1][2]],
                       [G[1][0] * G[2][1] - G[1][1] * G[2][0], G[0][1] * G[2][0] - G[0][0] * G[2][1], G[0][0] * G[1][1] - G[0][1] * G[1][0]]]
                cs = [c[k], c[l], c[m_]]; scale = g[k][k] * g[l][l] * g[m_][m_]
                lam = [tree_sum(adj[p][q] * cs[q] for q in range(3)) / det for p in range(3)]
            uS = [u[j] - tree_sum(lam[p] * a[S[p]][j] for p in range(len(S))) for j in range(NV)]
            ok = det > DET_REL * scale            # Gram determinant relative to the row norms: rejects near-parallel active sets
            move2 = tree_sum((uS[j] - u[j]) * (uS[j] - u[j]) for j in range(NV))
            ref2 = tree_sum(ca.fmax(0, c[S[p]]) * ca.fmax(0, c[S[p]]) / g[S[p]][S[p]] for p in range(len(S)))
            ok = ca.logic_and(ok, move2 <= TRUST * ref2 + 1e-12)
            for p in range(len(S)):
                ok = ca.logic_and(ok, lam[p] >= -PROJ_TOL)
            for k in range(3):
                if k not in S:
                    ok = ca.logic_and(ok, (tree_sum(a[k][j] * uS[j] for j in range(NV)) - bb[k]) * inv_norm_slot(k) <= PROJ_TOL)
            return uS, ok
        def inv_norm_slot(k):
            return 1.0 / ca.sqrt(g[k][k] + 1e-12)
        # fallback: single projection onto the most violated row; empty set: nothing violated
        u_new = [u[j] - ca.fmax(0, c[0]) / g[0][0] * a[0][j] for j in range(NV)]
        none_viol = tree_max(r) <= PROJ_TOL
        u_new = [ca.if_else(none_viol, u[j], u_new[j]) for j in range(NV)]
        for S in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
            uS, ok = cand(S)
            ok = ca.logic_and(ok, ca.logic_not(none_viol))
            u_new = [ca.if_else(ok, uS[j], u_new[j]) for j in range(NV)]
        u = ca.vertcat(*u_new)
    return u


def build_step_function(T, dt=DT, rounds=PROJ_ROUNDS, name="safe_step", cse=True, rows="all"):
    """rows: "all" (the 64 barriers of the demo) or "scene" (joint limits, obstacle, floor; no self-collision / singularity)."""
    m = _Model()
    nv, I7 = NV, ca.SX.eye(NV)
    q = ca.SX.sym("q", nv); v = ca.SX.sym("v", nv); tau = ca.SX.sym("tau"); c = ca.SX.sym("c")
    Y = ca.SX.sym("Y", ND); rp = ca.SX.sym("rp", 3); par = ca.SX.sym("par", NPAR)
    P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
    x = ca.vertcat(q, v, tau, c); z = ca.vertcat(q, v)
    K = m.kin(q, v)
    M, g, cor = K["M"], K["g"], K["c"]
    # 1. joint target from the knots
    w = interp_weights(tau, knot_dt(T))
    q_des = sum(w[j] * Y[j * nv:(j + 1) * nv] for j in range(N_KNOT))
    # 2. nominal torque
    u_nom = M @ (P["kp"] * (q_des - q) - P["kd"] * v) + g + cor
    u_nom = ca.fmin(ca.fmax(u_nom, -m.u_max), m.u_max)
    # 3. one-step rows (cbfpy's RD2 rows, as in backup_staged.build_head_function)
    M_inv = chol_solve(M, I7)
    f_z = ca.vertcat(v, -M_inv @ (g + cor)); g_z = ca.vertcat(ca.SX.zeros(nv, nv), M_inv)
    h2 = m.barriers(q, K)
    if rows == "scene":
        n_sph = K["sph"].shape[0]; n_obs = n_sph * OBSTACLE_POS.shape[0]; n_sc = len(m.D["body_sc_pairs"]) + len(m.D["root_sc_pairs"])
        h2 = ca.vertcat(h2[:2 * nv], h2[2 * nv + 1:2 * nv + 1 + n_obs], h2[2 * nv + 1 + n_obs + n_sc:])
    h = ca.jtimes(h2, z, f_z) + ALPHA * h2
    L = ca.jtimes(h, z, ca.horzcat(f_z, g_z))
    G = -L[:, 1:]; hq = ALPHA * h + L[:, 0]
    # 4. closed-form filter
    A = [[G[i, j] for j in range(nv)] for i in range(G.shape[0])] + [[1.0 if j == k else 0.0 for j in range(nv)] for k in range(nv)] + [[-1.0 if j == k else 0.0 for j in range(nv)] for k in range(nv)]
    b = [hq[i] for i in range(hq.shape[0])] + [m.u_max[k] for k in range(nv)] + [m.u_max[k] for k in range(nv)]
    u = project(u_nom, A, b, rounds)
    u = ca.fmin(ca.fmax(u, -m.u_max), m.u_max)
    # 5. dynamics
    a = chol_solve(M, u - g - cor)
    v1 = v + dt * a; q1 = q + dt * v1
    # 6. cost
    rot, pos = K["rot"], K["pos"]
    des_rot = ca.DM(DES_ROT)
    e_rot = -0.5 * (ca.cross(rot[:, 0], des_rot[:, 0]) + ca.cross(rot[:, 1], des_rot[:, 1]) + ca.cross(rot[:, 2], des_rot[:, 2]))
    ell = P["w_pos"] * ca.sumsqr(pos - rp) + P["w_rot"] * ca.sumsqr(e_rot) + P["w_v"] * ca.sumsqr(v) + P["w_u"] * ca.sumsqr(u)
    x1 = ca.vertcat(q1, v1, tau + dt, c + dt * ell)
    outs = [x1, u]
    if cse:
        outs = ca.cse(outs)
    fn = ca.Function(name, [x, Y, rp, par], outs, ["x", "Y", "rp", "par"], ["x1", "u"])
    return fn, m.robot, dict(T=T, dt=dt, proj_rounds=rounds, rows=rows, n_cbf=int(h2.shape[0]), sx_nodes=fn.n_nodes())


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
    h2 = m.barriers(q, K)
    if rows == "scene":
        n_sph = K["sph"].shape[0]; n_obs = n_sph * OBSTACLE_POS.shape[0]; n_sc = len(m.D["body_sc_pairs"]) + len(m.D["root_sc_pairs"])
        h2 = ca.vertcat(h2[:2 * nv], h2[2 * nv + 1:2 * nv + 1 + n_obs], h2[2 * nv + 1 + n_obs + n_sc:])
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
    return ca.Function(name, [x], [m.barriers(q, K), K["pos"]], ["x"], ["h2", "p_ee"])


if __name__ == "__main__":
    fn, robot, info = build_step_function(T=20)
    print(info)
