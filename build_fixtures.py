"""
Build CasADi fixture functions for validation and debugging.

Usage:
    python build_fixtures.py

Generates .casadi files in src/casadi_functions/. These are gitignored
(build artifacts); this script is the authoritative source.

Current fixtures:
    double_integrator          — f(x[2], u[1]) -> xdot[2], dense 2-vector
    double_integrator_jac      — f(x[2], u[1]) -> jac[2x3], sparse 2-nnz Jacobian
    stage_cost_hess            — f(x[2], u[1]) -> H[3x3], cost Hessian (diag)
    dynamics_defect_jac        — f(x_t[2], u_t[1], x_tp1[2]) -> J[2x5], defect Jacobian
"""

import os
from casadi import SX, Function, hessian, jacobian, vertcat, diag

FUNCTION_DIR = os.path.join(os.path.dirname(__file__), "src", "casadi_functions")


def build_double_integrator():
    x = SX.sym("x", 2)
    u = SX.sym("u", 1)
    xdot = vertcat(x[1], u[0])

    f_dyn = Function("double_integrator", [x, u], [xdot], ["x", "u"], ["xdot"])
    path_dyn = os.path.join(FUNCTION_DIR, "double_integrator.casadi")
    f_dyn.save(path_dyn)
    print(f"Saved: {path_dyn}")
    print(f"  {f_dyn}")
    print(f"  n_instructions={f_dyn.n_instructions()}, sz_w={f_dyn.sz_w()}")
    for i in range(f_dyn.n_out()):
        sp = f_dyn.sparsity_out(i)
        rows, cols = sp.get_triplet()
        print(f"  out[{i}]: shape=({f_dyn.size1_out(i)},{f_dyn.size2_out(i)}) "
              f"nnz={f_dyn.nnz_out(i)} dense={sp.is_dense()}")

    xu = vertcat(x, u)
    J = jacobian(xdot, xu)
    f_jac = Function("double_integrator_jac", [x, u], [J], ["x", "u"], ["jac"])
    path_jac = os.path.join(FUNCTION_DIR, "double_integrator_jac.casadi")
    f_jac.save(path_jac)
    print(f"\nSaved: {path_jac}")
    print(f"  {f_jac}")
    print(f"  n_instructions={f_jac.n_instructions()}, sz_w={f_jac.sz_w()}")
    for i in range(f_jac.n_out()):
        sp = f_jac.sparsity_out(i)
        rows, cols = sp.get_triplet()
        print(f"  out[{i}]: shape=({f_jac.size1_out(i)},{f_jac.size2_out(i)}) "
              f"nnz={f_jac.nnz_out(i)} dense={sp.is_dense()}")
        print(f"    triplet: rows={list(rows)} cols={list(cols)}")


def _print_fn_info(f):
    print(f"  {f}")
    print(f"  n_instructions={f.n_instructions()}, sz_w={f.sz_w()}")
    for i in range(f.n_out()):
        sp = f.sparsity_out(i)
        rows, cols = sp.get_triplet()
        print(f"  out[{i}]: shape=({f.size1_out(i)},{f.size2_out(i)}) "
              f"nnz={f.nnz_out(i)} dense={sp.is_dense()}")
        if f.nnz_out(i) <= 20:
            print(f"    triplet: rows={list(rows)} cols={list(cols)}")


def build_trajectory_patterns():
    """Build stage cost Hessian and dynamics defect Jacobian for C4."""
    x = SX.sym("x", 2)
    u = SX.sym("u", 1)
    z = vertcat(x, u)

    Q = diag(SX([1.0, 1.0]))
    R = SX(0.1)
    cost = 0.5 * (x.T @ Q @ x + u.T * R * u)
    H, _ = hessian(cost, z)
    f_hess = Function("stage_cost_hess", [x, u], [H], ["x", "u"], ["H"])
    path = os.path.join(FUNCTION_DIR, "stage_cost_hess.casadi")
    f_hess.save(path)
    print(f"Saved: {path}")
    _print_fn_info(f_hess)

    x_t = SX.sym("x_t", 2)
    u_t = SX.sym("u_t", 1)
    x_tp1 = SX.sym("x_tp1", 2)
    dt = 0.1
    xdot = vertcat(x_t[1], u_t[0])
    defect = x_tp1 - x_t - dt * xdot
    z_full = vertcat(x_t, u_t, x_tp1)
    J = jacobian(defect, z_full)
    f_jac = Function("dynamics_defect_jac", [x_t, u_t, x_tp1], [J],
                     ["x_t", "u_t", "x_tp1"], ["J"])
    path = os.path.join(FUNCTION_DIR, "dynamics_defect_jac.casadi")
    f_jac.save(path)
    print(f"\nSaved: {path}")
    _print_fn_info(f_jac)


if __name__ == "__main__":
    build_double_integrator()
    print()
    build_trajectory_patterns()
