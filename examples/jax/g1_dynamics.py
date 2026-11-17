#!/usr/bin/env python
"""Unitree G1 rigid-body dynamics from the Menagerie MJCF, through cusadi2.jax.rbd,
evaluated by cusadi2 with exact-sparsity Jacobians and Hessians.

`rbd` parses the model with MuJoCo (MJCF here; a URDF works the same way) and
gives Featherstone's recursive Newton-Euler and articulated-body algorithms in
body coordinates as plain JAX functions, reproducing MuJoCo's mj_inverse /
mj_forward (constraints and actuation disabled) to round-off. Their jaxpr is
what cusadi2 interprets into CasADi SX, differentiates symbolically and
compiles into GPU plans.

Dependencies: jax, numpy, mujoco (parser and reference values), cusadi2 with its
JAX extension.

    uv run --extra examples examples/jax/g1_dynamics.py            # GPU: plans, derivatives, timing
    uv run --extra examples examples/jax/g1_dynamics.py --jax-only # only check the JAX dynamics vs MuJoCo
    options: --batch 1024  --xml path/to/model.xml
             --true-hessian  also the full (35 x 106 x 106) Hessian tensor of the inverse dynamics
             --fd-hessian    also the Lagrangian Hessian of the forward dynamics (dense through M^-1)

Functions (q: 36 = pelvis position, quaternion wxyz, 29 hinge angles;
qd / qdd / tau: 35, MuJoCo conventions):

    inverse_dynamics(q, qd, qdd) -> qfrc    tree-sparse Jacobian (35 x 106) and Lagrangian Hessian
                                            (106 x 106): the constraint objects of an inverse-dynamics
                                            transcription (qdd a decision variable), what cuipm consumes
    forward_dynamics(q, qd, tau) -> qdd     Jacobian dense through M^-1 for any formulation
"""
import argparse
import os

# cusadi2 plans live outside XLA's memory pool; leave the GPU room for them instead of
# letting JAX grab 75% up front (set before JAX is imported, only if the user hasn't)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "mujoco_menagerie", "unitree_g1", "g1.xml")


# ----------------------------------------------------------------------------- MuJoCo reference
def random_states(m, n, seed=0):
    import mujoco
    rng = np.random.default_rng(seed)
    q = np.tile(m.qpos0, (n, 1)) + rng.standard_normal((n, m.nq)) * 0.5
    for k in range(n):
        mujoco.mj_normalizeQuat(m, q[k])
    return q, rng.standard_normal((n, m.nv)), rng.standard_normal((n, m.nv)) * 5.0


def mujoco_reference(m, q, qd, tau_or_qdd, kind):
    import mujoco
    # no contacts, limits, friction loss; and no actuators: the Menagerie G1 uses position
    # actuators whose affine bias (-kp q - kv qd) acts even at zero control
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
    d = mujoco.MjData(m)
    out = np.zeros_like(tau_or_qdd)
    for k in range(q.shape[0]):
        d.qpos[:], d.qvel[:] = q[k], qd[k]
        if kind == "forward":
            d.qfrc_applied[:] = tau_or_qdd[k]
            mujoco.mj_forward(m, d)
            out[k] = d.qacc
        else:
            d.qacc[:] = tau_or_qdd[k]
            mujoco.mj_inverse(m, d)
            out[k] = d.qfrc_inverse
    return out


def rel_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - b)) / (np.max(np.abs(b)) + 1e-300))


def timeit(fn, *args, n=20):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n):
        t = time.perf_counter(); jax.block_until_ready(fn(*args)); ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e6


def dense_from_coo(coo, values, shape):
    out = np.zeros(shape)
    out[tuple(np.asarray(coo).T)] = np.asarray(values)
    return out


def numerical_nnz(A, tol=1e-10):
    A = np.asarray(A)
    return int(np.count_nonzero(np.abs(A) > tol * np.max(np.abs(A))))


def stack_blocks(H, nd):
    """Nested tuple of blocks from jax.jacfwd / jax.hessian with argnums=(0,1,2) -> one array."""
    if nd == 2:
        return np.block([[np.asarray(H[i][j]) for j in range(3)] for i in range(3)])
    return np.concatenate([np.concatenate([np.asarray(H[i][j]) for j in range(3)], axis=2) for i in range(3)], axis=1)


def report_derivatives(fn, f, x0, label, weighted, true_hessian, lam):
    """Build the Jacobian (and Hessian) plans of a jax_function and check them against JAX autodiff."""
    args = [jnp.asarray(a) for a in x0]
    t = time.perf_counter(); jac = fn.jacobian(); coo = jac.coo(*x0); t_j = time.perf_counter() - t
    J_ref = np.hstack([np.asarray(b) for b in jax.jacfwd(f, argnums=(0, 1, 2))(*args)])
    print(f"[cusadi2] {label}: Jacobian plan in {t_j:.1f}s: {coo.shape[0]} nnz structural / {numerical_nnz(J_ref)} "
          f"numerical of {J_ref.size} ({100 * coo.shape[0] / J_ref.size:.1f}%), vs jax.jacfwd max rel err "
          f"{rel_err(dense_from_coo(coo, jac(*x0), jac.shape(*x0)), J_ref):.1e}", flush=True)
    out = {"jac": jac}
    if weighted:
        t = time.perf_counter(); whes = fn.hessian(weighted=True); coo2 = whes.coo(*x0); t_h = time.perf_counter() - t
        lag = lambda q, v, a: jnp.dot(jnp.asarray(lam), f(q, v, a))
        H_ref = stack_blocks(jax.hessian(lag, argnums=(0, 1, 2))(*args), 2)
        mode = "serial" if fn._plans[("whess", fn._trace(x0)["sig"])].serial else "level-parallel"
        print(f"[cusadi2] {label}: Lagrangian Hessian plan ({mode}) in {t_h:.1f}s: {coo2.shape[0]} nnz structural / "
              f"{numerical_nnz(H_ref)} numerical of {H_ref.size} ({100 * coo2.shape[0] / H_ref.size:.1f}%), vs jax.hessian "
              f"max rel err {rel_err(dense_from_coo(coo2, whes(*x0, lam=jnp.asarray(lam)), whes.shape(*x0)), H_ref):.1e}",
              flush=True)
        out["whess"] = whes
    if true_hessian:
        t = time.perf_counter(); hes = fn.hessian(); coo3 = hes.coo(*x0); t_h = time.perf_counter() - t
        H_ref = stack_blocks(jax.hessian(f, argnums=(0, 1, 2))(*args), 3)
        mode = "serial" if fn._plans[("hess", fn._trace(x0)["sig"])].serial else "level-parallel"
        print(f"[cusadi2] {label}: full Hessian tensor plan ({mode}) in {t_h:.1f}s: {coo3.shape[0]} nnz structural / "
              f"{numerical_nnz(H_ref)} numerical of {H_ref.size} ({100 * coo3.shape[0] / H_ref.size:.2f}%), vs jax.hessian "
              f"max rel err {rel_err(dense_from_coo(coo3, hes(*x0), hes.shape(*x0)), H_ref):.1e}", flush=True)
        out["hess"] = hes
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--jax-only", action="store_true", help="check the JAX dynamics against MuJoCo and stop")
    ap.add_argument("--true-hessian", action="store_true", help="also the full Hessian tensor of the inverse dynamics")
    ap.add_argument("--fd-hessian", action="store_true", help="also the Lagrangian Hessian of the forward dynamics")
    args = ap.parse_args()

    import mujoco
    from cusadi2.jax import rbd
    m = mujoco.MjModel.from_xml_path(args.xml)
    model = rbd.RigidBodyModel.from_mujoco(m)
    print(f"model: {model.nbody - 1} bodies, nq={model.nq}, nv={model.nv}")
    fd = lambda q, qd, tau: model.forward_dynamics(q, qd, tau)
    inv = lambda q, qd, qdd: model.inverse_dynamics(q, qd, qdd)

    q, qd, u = random_states(m, 4)
    ref_tau = mujoco_reference(m, q, qd, qd, "inverse")        # qdd := qd for the inverse check
    ref_qdd = mujoco_reference(m, q, qd, u, "forward")
    print(f"[jax]     inverse dynamics vs mj_inverse: max rel err {rel_err(jax.jit(jax.vmap(inv))(q, qd, qd), ref_tau):.1e}")
    print(f"[jax]     forward dynamics vs mj_forward: max rel err {rel_err(jax.jit(jax.vmap(fd))(q, qd, u), ref_qdd):.1e}")
    if args.jax_only:
        return

    from cusadi2.jax import jax_function
    lam = np.random.default_rng(3).standard_normal(model.nv)
    for name, f, inputs, ref in (("inverse_dynamics", inv, (q, qd, qd), ref_tau), ("forward_dynamics", fd, (q, qd, u), ref_qdd)):
        x0 = [a[0] for a in inputs]
        fn = jax_function(f, name=f"g1_{name}")
        t = time.perf_counter(); y = jax.jit(jax.vmap(fn))(*inputs); jax.block_until_ready(y)
        print(f"[cusadi2] {name}: value plan in {time.perf_counter() - t:.1f}s; vs MuJoCo max rel err {rel_err(y, ref):.1e}")
        is_inv = name == "inverse_dynamics"
        plans = report_derivatives(fn, f, x0, name, weighted=is_inv or args.fd_hessian,
                                   true_hessian=is_inv and args.true_hessian, lam=lam)

        # batched evaluation on the GPU
        Q, QD, U = random_states(m, args.batch, seed=1)
        L = np.tile(lam, (args.batch, 1))
        line = (f"[cusadi2] {name}: batch {args.batch}: value {timeit(jax.jit(jax.vmap(fn)), Q, QD, U):.0f} us, "
                f"Jacobian {timeit(jax.jit(jax.vmap(plans['jac'])), Q, QD, U):.0f} us")
        if "whess" in plans:
            whess = plans["whess"]
            line += f", Lagrangian Hessian {timeit(jax.jit(jax.vmap(lambda a, b, c, l: whess(a, b, c, lam=l))), Q, QD, U, L):.0f} us"
        if "hess" in plans:
            line += f", full Hessian {timeit(jax.jit(jax.vmap(plans['hess'])), Q, QD, U):.0f} us"
        print(line + " per batched call", flush=True)


if __name__ == "__main__":
    main()
