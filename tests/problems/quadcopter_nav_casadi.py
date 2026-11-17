"""The same quadcopter NMPC written by hand in CasADi SX, copied from cuipm's
codegen/generate.py (build_quadcopter_nav_sx / build_nlp_functions_from_sx)
WITHOUT the `eps` pattern-preservation input: with the JAX side now also going
through CasADi SX, both sides fold constant zeros the same way and the hack is
unnecessary. This is the sparsity reference for tests/test_quadcopter_nav.py.
"""
import casadi as ca
import numpy as np

from tests.problems.quadcopter_nav import OBSTACLES, quad_params


def _f_ca(x, u, qp, d, g):
    th0 = qp["kTh"] * u[0] ** 2; th1 = qp["kTh"] * u[1] ** 2
    th2 = qp["kTh"] * u[2] ** 2; th3 = qp["kTh"] * u[3] ** 2
    to0 = qp["kTo"] * u[0] ** 2; to1 = qp["kTo"] * u[1] ** 2
    to2 = qp["kTo"] * u[2] ** 2; to3 = qp["kTo"] * u[3] ** 2
    dx, dy, dz = x[7], x[8], x[9]
    dq0 = -0.5 * x[10] * x[4] - 0.5 * x[11] * x[5] - 0.5 * x[6] * x[12]
    dq1 = 0.5 * x[10] * x[3] - 0.5 * x[11] * x[6] + 0.5 * x[5] * x[12]
    dq2 = 0.5 * x[10] * x[6] + 0.5 * x[11] * x[3] - 0.5 * x[4] * x[12]
    dq3 = -0.5 * x[10] * x[5] + 0.5 * x[11] * x[4] + 0.5 * x[3] * x[12]
    vx_rel = d[0] * ca.cos(d[1]) * ca.cos(d[2]) - x[7]
    vy_rel = d[0] * ca.sin(d[1]) * ca.cos(d[2]) - x[8]
    vz_rel = d[0] * ca.sin(d[2]) + x[9]
    total_thrust = th0 + th1 + th2 + th3
    ddx = (qp["Cd"] * ca.sign(vx_rel) * vx_rel ** 2 - 2.0 * (x[3] * x[5] + x[4] * x[6]) * total_thrust) / qp["mB"]
    ddy = (qp["Cd"] * ca.sign(vy_rel) * vy_rel ** 2 + 2.0 * (x[3] * x[4] - x[5] * x[6]) * total_thrust) / qp["mB"]
    ddz = (-qp["Cd"] * ca.sign(vz_rel) * vz_rel ** 2
           - total_thrust * (x[3] ** 2 - x[4] ** 2 - x[5] ** 2 + x[6] ** 2) + g * qp["mB"]) / qp["mB"]
    prec_coef = float(qp["usePrecession"]) * qp["IRzz"] * (u[0] - u[1] + u[2] - u[3])
    dp = ((qp["IB"][1, 1] - qp["IB"][2, 2]) * x[11] * x[12] - prec_coef * x[11]
          + (th0 - th1 - th2 + th3) * qp["dym"]) / qp["IB"][0, 0]
    dq = ((qp["IB"][2, 2] - qp["IB"][0, 0]) * x[10] * x[12] + prec_coef * x[10]
          + (th0 + th1 - th2 - th3) * qp["dxm"]) / qp["IB"][1, 1]
    dr = ((qp["IB"][0, 0] - qp["IB"][1, 1]) * x[10] * x[11] - to0 + to1 - to2 + to3) / qp["IB"][2, 2]
    return ca.vertcat(dx, dy, dz, dq0, dq1, dq2, dq3, ddx, ddy, ddz, dp, dq, dr)


def build_quadcopter_nav_sx(N=30, Ts=0.1):
    nx_dim, nu_dim = 13, 4
    nx = N * nx_dim + (N - 1) * nu_dim
    x = ca.SX.sym("x", nx)
    p = ca.SX.sym("p", nx_dim)
    xs = ca.reshape(x[: N * nx_dim], nx_dim, N).T
    us = ca.reshape(x[N * nx_dim:], nu_dim, N - 1).T
    Q = ca.DM([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    f_sx = ca.SX(0.0)
    for k in range(N):
        f_sx = f_sx + ca.sum1(Q * xs[k, :].T ** 2)
    eq_list = [xs[0, :].T - p]
    for k in range(N - 1):
        x_next = xs[k, :].T + _f_ca(xs[k, :].T, us[k, :].T, quad_params, [0.0, 0.0, 0.0], 9.81) * Ts
        eq_list.append(xs[k + 1, :].T - x_next)
    c_sx = ca.vertcat(*eq_list)
    r__ = [r + OBSTACLES["quad_radius"] for r in OBSTACLES["r_"]]
    ineq_list = []
    for k in range(N - 1):
        multiplier = 1.0 + k * Ts * 0.1
        for xc, yc, r in zip(OBSTACLES["xc_"], OBSTACLES["yc_"], r__):
            ineq_list.append((xs[k, 0] - xc) ** 2 + (xs[k, 1] - yc) ** 2 - r ** 2 * multiplier)
    d_sx = ca.vertcat(*ineq_list)
    return x, p, f_sx, c_sx, d_sx


def reference_nnz(N=30, Ts=0.1):
    """nnz of the nine quantities cuipm's codegen emits, from the hand-written SX."""
    x, p, f_sx, c_sx, d_sx = build_quadcopter_nav_sx(N, Ts)
    lam_c = ca.SX.sym("lam_c", c_sx.numel())
    lam_d = ca.SX.sym("lam_d", d_sx.numel())
    return {
        "jac_f": ca.jacobian(f_sx, x).nnz(),
        "jac_c": ca.jacobian(c_sx, x).nnz(),
        "jac_d": ca.jacobian(d_sx, x).nnz(),
        "hess_f": ca.hessian(f_sx, x)[0].nnz(),
        "hess_c": ca.hessian(ca.dot(lam_c, c_sx), x)[0].nnz(),
        "hess_d": ca.hessian(ca.dot(lam_d, d_sx), x)[0].nnz(),
        "nx": int(x.numel()), "nyc": int(c_sx.numel()), "nyd": int(d_sx.numel()),
    }
