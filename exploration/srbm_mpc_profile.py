"""
Build one SRBM-MPC SQP iteration symbolically, measure the LP DAG.

Mirrors the CusADi paper's MPC-parallelization structure for the MIT Humanoid:
 * state x ∈ R^12   (pos, Euler angles, lin vel, ang vel)
 * input u ∈ R^6    (3D force per foot × 2 feet — H1)
 * horizon T stages (10 default)
 * dynamics = SRBM Newton-Euler with Euler integration
 * cost     = quadratic tracking on x_ref, quadratic regularization on u
 * constraints: x_0 = x_init (equality), dynamics defects (equality),
                friction-cone / force-bound (inequality)
 * one iteration: build gradient g and Hessian H of cost at the current
   guess, build Jacobians A_eq / A_ineq, form the penalty-regularized QP
   (P̃ = H + μ_eq A_eq^T A_eq + μ_ineq A_ineq^T A_ineq,
    c̃ = g + μ_eq A_eq^T h_eq + μ_ineq A_ineq^T p_ineq),
   solve via an explicit scalar Cholesky, update z.

The solver is written with scalar SX expressions so every arithmetic
operation surfaces in the instruction stream — no OP_SOLVE / OP_MTIMES
nodes — giving a fair DAG to the LP planner. Report:
  - n_instructions
  - LP levels (depth)
  - mean / p50 / p95 / max level width
  - peak live slots (shared-memory pressure)
  - ops-per-level histogram (to see shape: wide-shallow vs narrow-deep)
"""
from __future__ import annotations

import os
import sys

import casadi as ca
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.level_parallel.dag import extract_level_plan  # noqa: E402


# ---- SRBM parameters (H1-like) --------------------------------------------
MASS = 51.6
INERTIA = np.diag([5.0, 5.0, 2.0])          # body inertia (kg·m²), rough
G_VEC = np.array([0.0, 0.0, -9.81])
DT_MPC = 0.02                                # MPC stage length (s)
MU_FRIC = 0.7                                # friction coefficient
F_Z_MAX = 500.0                              # max normal force per foot
# Foot positions (gait scheduler output; fixed here for a standing posture)
P_FOOT_L = np.array([0.00, +0.15, 0.00])
P_FOOT_R = np.array([0.00, -0.15, 0.00])

NX = 12
NU = 6    # [f_L(3), f_R(3)]


def srbm_step(x_sx: ca.SX, u_sx: ca.SX) -> ca.SX:
    """One Euler step of the SRBM. x = [p, θ, v, ω], u = [f_L, f_R]."""
    p    = x_sx[0:3]
    th   = x_sx[3:6]
    v    = x_sx[6:9]
    w    = x_sx[9:12]
    fL   = u_sx[0:3]
    fR   = u_sx[3:6]
    F    = fL + fR + ca.DM(MASS * G_VEC)                        # (3,)
    rL   = ca.DM(P_FOOT_L) - p
    rR   = ca.DM(P_FOOT_R) - p
    tau  = ca.cross(rL, fL) + ca.cross(rR, fR)                  # (3,)
    # Use a FIXED diagonal inertia; no cross coupling from ω×Iω in this
    # simplified SRBM (matches paper-level fidelity, keeps the DAG clean).
    I_diag = ca.DM(np.diag(INERTIA))
    alpha = tau / I_diag
    a     = F / MASS
    return ca.vertcat(p + v * DT_MPC,
                      th + w * DT_MPC,
                      v + a * DT_MPC,
                      w + alpha * DT_MPC)


def build_one_sqp_iteration(T: int) -> ca.Function:
    """One Gauss-Newton / penalty-SQP iteration as a closed-form SX function.

    Inputs: (z_current, x_init, x_ref)
    Output: z_next
    """
    n_dec = (T + 1) * NX + T * NU

    # Decision variable as a single flat SX
    z = ca.SX.sym("z", n_dec)
    x_init = ca.SX.sym("x_init", NX)
    x_ref = ca.SX.sym("x_ref", NX)

    # Slice z into stage variables
    x_stage = [z[i * NX : (i + 1) * NX] for i in range(T + 1)]
    u_stage = [z[(T + 1) * NX + i * NU : (T + 1) * NX + (i + 1) * NU] for i in range(T)]

    # ---- Cost: sum of stage + terminal tracking, input regularization ----
    Q = np.eye(NX)
    Q_terminal = 10.0 * np.eye(NX)
    R = 1e-4 * np.eye(NU)
    cost = ca.SX(0)
    for i in range(T):
        e = x_stage[i] - x_ref
        cost = cost + 0.5 * e.T @ ca.DM(Q) @ e
        cost = cost + 0.5 * u_stage[i].T @ ca.DM(R) @ u_stage[i]
    eT = x_stage[T] - x_ref
    cost = cost + 0.5 * eT.T @ ca.DM(Q_terminal) @ eT

    # ---- Equality constraints (initial + dynamics defects) ----
    eq_parts = [x_stage[0] - x_init]
    for i in range(T):
        eq_parts.append(x_stage[i + 1] - srbm_step(x_stage[i], u_stage[i]))
    h_eq = ca.vertcat(*eq_parts)

    # ---- Inequality constraints: friction pyramid + normal bounds ----
    # Per foot (3D force f = (fx, fy, fz)):
    #   fz >= 0            → -fz <= 0
    #   fz <= F_Z_MAX      → fz - F_Z_MAX <= 0
    #   |fx| <= μ fz       → fx - μ fz <= 0,  -fx - μ fz <= 0
    #   |fy| <= μ fz       → fy - μ fz <= 0,  -fy - μ fz <= 0
    ineq_parts = []
    for i in range(T):
        for foot in (u_stage[i][0:3], u_stage[i][3:6]):
            fx, fy, fz = foot[0], foot[1], foot[2]
            ineq_parts.extend([
                -fz,
                fz - F_Z_MAX,
                fx - MU_FRIC * fz,
                -fx - MU_FRIC * fz,
                fy - MU_FRIC * fz,
                -fy - MU_FRIC * fz,
            ])
    g_ineq = ca.vertcat(*ineq_parts)
    p_ineq = ca.fmax(g_ineq, 0.0)      # penalty residual: 0 when feasible

    # ---- Linearization at z (Gauss-Newton): gradient + Hessian of cost ----
    grad = ca.gradient(cost, z)
    # Exact Hessian of the quadratic tracking cost (constant w.r.t. z, but
    # CasADi will still write the scalar entries — fine for DAG profiling).
    H = ca.hessian(cost, z)[0]
    A_eq = ca.jacobian(h_eq, z)
    A_ineq = ca.jacobian(g_ineq, z)

    # ---- Penalty QP: P̃ dz = -c̃  (symmetric positive-definite system) ----
    mu_eq = 1e4
    mu_ineq = 1e3
    # Build only the active-set contribution symbolically: multiply by
    # p_ineq > 0 via the soft indicator max(g, 0)²/... — simplest is to use
    # the penalty residual directly, letting g_ineq itself pass through.
    P_tilde = H + mu_eq * (A_eq.T @ A_eq) + mu_ineq * (A_ineq.T @ A_ineq)
    c_tilde = grad + mu_eq * (A_eq.T @ h_eq) + mu_ineq * (A_ineq.T @ p_ineq)

    # ---- Scalar Cholesky: P̃ = L L^T, solve L y = -c̃, L^T dz = y ----
    # Operate entirely on SX scalars so every FP op is in the instruction
    # stream (nothing hides behind OP_SOLVE / OP_MTIMES).
    n = n_dec
    P_mat = [[P_tilde[i, j] for j in range(n)] for i in range(n)]
    L = [[ca.SX(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = P_mat[i][j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
            if i == j:
                L[i][i] = ca.sqrt(s)
            else:
                L[i][j] = s / L[j][j]

    neg_c = [-c_tilde[i] for i in range(n)]
    y = [ca.SX(0)] * n
    for i in range(n):
        s = neg_c[i]
        for k in range(i):
            s = s - L[i][k] * y[k]
        y[i] = s / L[i][i]
    dz_list = [ca.SX(0)] * n
    for i in reversed(range(n)):
        s = y[i]
        for k in range(i + 1, n):
            s = s - L[k][i] * dz_list[k]
        dz_list[i] = s / L[i][i]
    dz = ca.vertcat(*dz_list)

    z_new = z + dz
    return ca.Function("srbm_mpc_step",
                       [z, x_init, x_ref], [z_new],
                       ["z", "x_init", "x_ref"], ["z_new"])


def profile(T: int) -> dict:
    print(f"\n========== SRBM MPC SQP iteration, T={T} ==========")
    f = build_one_sqp_iteration(T)
    print(f"  n_dec   = {(T + 1) * NX + T * NU}")
    print(f"  n_instructions = {f.n_instructions():,}")
    print(f"  sz_w           = {f.sz_w()}")

    plan = extract_level_plan(f)
    widths = np.array([lv.width for lv in plan.compute_levels])
    ops = int(widths.sum())
    print(f"  LP levels       = {plan.n_compute_levels}  (depth)")
    print(f"  total compute ops = {ops:,}")
    print(f"  peak_live_slots = {plan.peak_live_slots}  "
          f"({plan.peak_live_slots * 8 / 1024:.1f} KB)")
    print(f"  fits in 96 KB   = {plan.peak_live_slots * 8 <= 96 * 1024}")
    print(f"  level width:  mean={widths.mean():.1f}  p50={np.median(widths):.0f}  "
          f"p95={np.percentile(widths, 95):.0f}  max={widths.max()}")
    # histogram of level widths
    bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 100000]
    hist, edges = np.histogram(widths, bins=bins)
    print(f"  width histogram (levels falling in each bin):")
    for i, h in enumerate(hist):
        if h > 0:
            print(f"    [{edges[i]:5d}, {edges[i+1]:5d})  →  {h:5d} levels")
    # ideal vs sequential
    ideal_ms_at_X_ops_per_ms = ops / 10000.0     # 10M ops/ms is a handy unit
    print(f"  sequential op count = {ops:,}")
    print(f"  levels × 1  (perfect-parallel lower bound) = {plan.n_compute_levels} steps")
    return {
        "T": T,
        "n_dec": (T + 1) * NX + T * NU,
        "n_instructions": f.n_instructions(),
        "levels": plan.n_compute_levels,
        "total_ops": ops,
        "peak_live_slots": plan.peak_live_slots,
        "peak_live_kb": plan.peak_live_slots * 8 / 1024,
        "width_mean": float(widths.mean()),
        "width_p50": float(np.median(widths)),
        "width_p95": float(np.percentile(widths, 95)),
        "width_max": int(widths.max()),
    }


def main():
    print("Profiling SRBM MPC SQP-iteration DAG for various horizon lengths.")
    print("State: R^12 (pos+Euler+v+ω)  Input: R^6 (2 feet × 3 forces)")
    print("Dynamics: Euler SRBM, Cost: quadratic, Penalty: equality + friction-cone.")
    results = []
    for T in [5, 8, 10]:
        results.append(profile(T))

    print("\n========== SUMMARY ==========")
    print(f"{'T':>4}  {'n_dec':>6}  {'n_instr':>10}  {'levels':>7}  "
          f"{'ops':>10}  {'peak_KB':>8}  {'w_mean':>7}  {'w_p95':>6}  {'w_max':>6}")
    for r in results:
        print(f"{r['T']:>4}  {r['n_dec']:>6}  {r['n_instructions']:>10,}  "
              f"{r['levels']:>7}  {r['total_ops']:>10,}  "
              f"{r['peak_live_kb']:>7.1f}  "
              f"{r['width_mean']:>7.1f}  {r['width_p95']:>6.0f}  {r['width_max']:>6}")


if __name__ == "__main__":
    main()
