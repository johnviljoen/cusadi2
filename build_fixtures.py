"""
Build CasADi fixture functions for validation and debugging.

Usage:
    python build_fixtures.py

Generates .casadi files in src/casadi_functions/. These are gitignored
(build artifacts); this script is the authoritative source.

Current fixtures:
    double_integrator      — f(x[2], u[1]) -> xdot[2], dense 2-vector output
    double_integrator_jac  — f(x[2], u[1]) -> jac[2x3], sparse 2-nnz Jacobian
"""

import os
from casadi import SX, Function, jacobian, vertcat

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


if __name__ == "__main__":
    build_double_integrator()
