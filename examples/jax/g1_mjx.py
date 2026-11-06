#!/usr/bin/env python
"""Comparison baseline: the Unitree G1 inverse dynamics through MuJoCo MJX instead
of cusadi2.jax.rbd, evaluated by cusadi2.

`mjx.inverse` is a JAX function and its jaxpr interprets into CasADi SX like any
other (no loops once constraints, contacts and actuation are disabled), so this
works. It is kept as the measurement of what MuJoCo's formulation costs:
MuJoCo expresses spatial quantities at the subtree centre of mass (com_pos /
cdof), which depends on every joint angle. That dependence cancels in the final
torques, but symbolic sparsity cannot see cancellation, so the structural
patterns are denser than the numerical ones (both are printed), and the graphs
are several times larger than the body-frame recursions of `rbd`
(examples/jax/g1_dynamics.py). The forward dynamics are not built here: dense
through M^-1 for any formulation, and through MJX the most expensive way.

Dependencies: jax, numpy, mujoco, mujoco-mjx, cusadi2 with its JAX extension.

    uv run --extra examples examples/jax/g1_mjx.py            # GPU: plans, derivatives, timing
    uv run --extra examples examples/jax/g1_mjx.py --jax-only # only check MJX vs MuJoCo C
    options: --batch 1024  --xml path/to/model.xml
"""
import argparse
import os

# cusadi2 plans live outside XLA's memory pool; leave the GPU room for them instead of
# letting JAX grab 75% up front (set before JAX is imported, only if the user hasn't)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "..", "mujoco_menagerie", "unitree_g1", "g1.xml")


def load(xml):
    """MjModel with smooth dynamics only, its MJX (JAX backend) twin, and the inverse dynamics."""
    import mujoco
    from mujoco import mjx
    m = mujoco.MjModel.from_xml_path(xml)
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_CONTACT
                           | mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    m.geom_contype[:] = 0            # no collision pairs: MJX would otherwise reject the
    m.geom_conaffinity[:] = 0        # cylinder-mesh pairs of the G1 at put_model
    mx = mjx.put_model(m, impl="jax")
    dx = mjx.make_data(mx)

    def inverse(q, qd, qdd):
        return mjx.inverse(mx, dx.replace(qpos=q, qvel=qd, qacc=qdd)).qfrc_inverse

    return m, inverse


def main():
    from g1_dynamics import mujoco_reference, random_states, rel_err, report_derivatives, timeit
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--jax-only", action="store_true", help="check MJX against MuJoCo C and stop")
    args = ap.parse_args()

    m, inv = load(args.xml)
    print(f"model: {m.nbody - 1} bodies, nq={m.nq}, nv={m.nv}")
    q, qd, _ = random_states(m, 4)
    ref_tau = mujoco_reference(m, q, qd, qd, "inverse")        # qdd := qd
    print(f"[mjx]     inverse vs mj_inverse: max rel err {rel_err(jax.jit(jax.vmap(inv))(q, qd, qd), ref_tau):.1e}")
    if args.jax_only:
        return

    from cusadi2.jax import jax_function
    x0 = [q[0], qd[0], qd[0]]
    lam = np.random.default_rng(3).standard_normal(m.nv)
    fn = jax_function(inv, name="g1_mjx_inverse")
    t = time.perf_counter(); y = jax.jit(jax.vmap(fn))(q, qd, qd); jax.block_until_ready(y)
    print(f"[cusadi2] mjx inverse: value plan in {time.perf_counter() - t:.1f}s; vs MuJoCo max rel err {rel_err(y, ref_tau):.1e}")
    plans = report_derivatives(fn, inv, x0, "mjx inverse", weighted=True, true_hessian=False, lam=lam)
    Q, QD, U = random_states(m, args.batch, seed=1)
    L = np.tile(lam, (args.batch, 1))
    whess = plans["whess"]
    print(f"[cusadi2] mjx inverse: batch {args.batch}: value {timeit(jax.jit(jax.vmap(fn)), Q, QD, U):.0f} us, "
          f"Jacobian {timeit(jax.jit(jax.vmap(plans['jac'])), Q, QD, U):.0f} us, Lagrangian Hessian "
          f"{timeit(jax.jit(jax.vmap(lambda a, b, c, l: whess(a, b, c, lam=l))), Q, QD, U, L):.0f} us per batched call")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    main()
