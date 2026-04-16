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
    panda_aba                  — f(q[9], v[9], tau[9]) -> a[9], Pinocchio ABA (3234 instrs)
    panda_aba_jac              — f(q[9], v[9], tau[9]) -> J[9x27, 234nz], ABA Jacobian (48k instrs)
    panda_cost_hess            — f(q[9], v[9], tau[9]) -> H[27x27, 27nz], quadratic cost Hessian
    a1_aba                     — f(q[19], v[18], tau[18]) -> a[18], floating-base ABA (5260 instrs)
    a1_aba_jac                 — f(q[19], v[18], tau[18]) -> J[18x55, 936nz], ABA Jacobian (150k instrs)
    a1_cost_hess               — f(q[19], v[18], tau[18]) -> H[55x55, 55nz], quadratic cost Hessian
    a1_joint_limit_jac         — f(q[19], v[18], tau[18]) -> J[19x55, 19nz], joint limit Jacobian
    h1_aba                     — f(q[26], v[25], tau[25]) -> a[25], floating-base ABA (9467 instrs)
    h1_aba_jac                 — f(q[26], v[25], tau[25]) -> J[25x76, 1825nz], ABA Jacobian (364k instrs)
    h1_cost_hess               — f(q[26], v[25], tau[25]) -> H[76x76, 76nz], quadratic cost Hessian
    h1_joint_limit_jac         — f(q[26], v[25], tau[25]) -> J[26x76, 26nz], joint limit Jacobian
"""

import os
from casadi import SX, Function, hessian, jacobian, vertcat, diag

HAS_PINOCCHIO = False
try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
    import pinocchio.casadi as cpin
    HAS_PINOCCHIO = True
except ImportError:
    pass

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


def build_panda_fixtures():
    """Build ABA forward dynamics + Jacobian + cost Hessian for the Panda 7-DOF.

    Requires pinocchio with CasADi support and robot_descriptions.
    """
    if not HAS_PINOCCHIO:
        print("Skipping Panda fixtures: pinocchio.casadi or robot_descriptions not available")
        return

    robot = load_robot_description("panda_description")
    model = robot.model
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    print(f"Panda model: nq={model.nq}, nv={model.nv}")

    q = SX.sym("q", model.nq)
    v = SX.sym("v", model.nv)
    tau = SX.sym("tau", model.nv)

    a = cpin.aba(cmodel, cdata, q, v, tau)
    f_aba = Function("panda_aba", [q, v, tau], [a], ["q", "v", "tau"], ["a"])
    path = os.path.join(FUNCTION_DIR, "panda_aba.casadi")
    f_aba.save(path)
    print(f"Saved: {path}")
    _print_fn_info(f_aba)

    z = vertcat(q, v, tau)
    J_a = jacobian(a, z)
    f_jac = Function("panda_aba_jac", [q, v, tau], [J_a], ["q", "v", "tau"], ["J"])
    path = os.path.join(FUNCTION_DIR, "panda_aba_jac.casadi")
    f_jac.save(path)
    print(f"\nSaved: {path}")
    _print_fn_info(f_jac)

    nz = model.nq + model.nv + model.nv
    cost = 0.5 * (q.T @ q + v.T @ v + 0.01 * tau.T @ tau)
    H, _ = hessian(cost, z)
    f_cost = Function("panda_cost_hess", [q, v, tau], [H], ["q", "v", "tau"], ["H"])
    path = os.path.join(FUNCTION_DIR, "panda_cost_hess.casadi")
    f_cost.save(path)
    print(f"\nSaved: {path}")
    _print_fn_info(f_cost)


def build_a1_fixtures():
    """Build ABA + Jacobian + cost Hessian + joint limits for the Unitree A1 quadruped.

    Floating-base model: nq=19 (3 pos + 4 quat + 12 joints), nv=18 (6 + 12).
    Requires pinocchio with CasADi support and robot_descriptions.
    """
    if not HAS_PINOCCHIO:
        print("Skipping A1 fixtures: pinocchio.casadi or robot_descriptions not available")
        return

    import pinocchio as pin

    robot = load_robot_description("a1_description", root_joint=pin.JointModelFreeFlyer())
    model = robot.model
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    print(f"A1 model: nq={model.nq}, nv={model.nv} (floating base)")

    q = SX.sym("q", model.nq)
    v = SX.sym("v", model.nv)
    tau = SX.sym("tau", model.nv)

    a = cpin.aba(cmodel, cdata, q, v, tau)
    f_aba = Function("a1_aba", [q, v, tau], [a], ["q", "v", "tau"], ["a"])
    path = os.path.join(FUNCTION_DIR, "a1_aba.casadi")
    f_aba.save(path)
    print(f"Saved: {path}")
    _print_fn_info(f_aba)

    z = vertcat(q, v, tau)
    J_a = jacobian(a, z)
    f_jac = Function("a1_aba_jac", [q, v, tau], [J_a], ["q", "v", "tau"], ["J"])
    path = os.path.join(FUNCTION_DIR, "a1_aba_jac.casadi")
    f_jac.save(path)
    print(f"\nSaved: {path}")
    _print_fn_info(f_jac)

    cost = 0.5 * (q.T @ q + v.T @ v + 0.01 * tau.T @ tau)
    H, _ = hessian(cost, z)
    f_cost = Function("a1_cost_hess", [q, v, tau], [H], ["q", "v", "tau"], ["H"])
    path = os.path.join(FUNCTION_DIR, "a1_cost_hess.casadi")
    f_cost.save(path)
    print(f"\nSaved: {path}")
    _print_fn_info(f_cost)

    J_limit = jacobian(q, z)
    f_limit = Function("a1_joint_limit_jac", [q, v, tau], [J_limit],
                       ["q", "v", "tau"], ["J"])
    path = os.path.join(FUNCTION_DIR, "a1_joint_limit_jac.casadi")
    f_limit.save(path)
    print(f"\nSaved: {path}")
    _print_fn_info(f_limit)


def build_h1_fixtures():
    """Build ABA + Jacobian + cost Hessian + joint limits for the Unitree H1 humanoid.

    Floating-base model: nq=26 (7 free-flyer + 19 joints), nv=25 (6 + 19).
    ABA Jacobian: ~364k instructions, ~25 min nvcc compile.
    Talos (nq=51, nv=50) was infeasible — CasADi symbolic Jacobian construction
    timed out. H1 is the largest humanoid-class model tractable with single-file
    codegen.
    """
    if not HAS_PINOCCHIO:
        print("Skipping H1 fixtures: pinocchio.casadi or robot_descriptions not available")
        return

    import pinocchio as pin

    robot = load_robot_description("h1_description", root_joint=pin.JointModelFreeFlyer())
    model = robot.model
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    print(f"H1 model: nq={model.nq}, nv={model.nv} (floating base)")

    q = SX.sym("q", model.nq)
    v = SX.sym("v", model.nv)
    tau = SX.sym("tau", model.nv)

    a = cpin.aba(cmodel, cdata, q, v, tau)
    f_aba = Function("h1_aba", [q, v, tau], [a], ["q", "v", "tau"], ["a"])
    path = os.path.join(FUNCTION_DIR, "h1_aba.casadi")
    f_aba.save(path)
    print(f"Saved: {path}")
    _print_fn_info(f_aba)

    z = vertcat(q, v, tau)
    J_a = jacobian(a, z)
    f_jac = Function("h1_aba_jac", [q, v, tau], [J_a], ["q", "v", "tau"], ["J"])
    path = os.path.join(FUNCTION_DIR, "h1_aba_jac.casadi")
    f_jac.save(path)
    print(f"\nSaved: {path}")
    _print_fn_info(f_jac)

    cost = 0.5 * (q.T @ q + v.T @ v + 0.01 * tau.T @ tau)
    H, _ = hessian(cost, z)
    f_cost = Function("h1_cost_hess", [q, v, tau], [H], ["q", "v", "tau"], ["H"])
    path = os.path.join(FUNCTION_DIR, "h1_cost_hess.casadi")
    f_cost.save(path)
    print(f"\nSaved: {path}")
    _print_fn_info(f_cost)

    J_limit = jacobian(q, z)
    f_limit = Function("h1_joint_limit_jac", [q, v, tau], [J_limit],
                       ["q", "v", "tau"], ["J"])
    path = os.path.join(FUNCTION_DIR, "h1_joint_limit_jac.casadi")
    f_limit.save(path)
    print(f"\nSaved: {path}")
    _print_fn_info(f_limit)


if __name__ == "__main__":
    build_double_integrator()
    print()
    build_trajectory_patterns()
    print()
    build_panda_fixtures()
    print()
    build_a1_fixtures()
    print()
    build_h1_fixtures()
