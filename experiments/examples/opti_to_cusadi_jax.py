import jax

jax.config.update("jax_enable_x64", True)  # cusadi kernels are fp64

import os

import casadi as ca
import jax.numpy as jnp
import numpy as np

from cusadi import CUSADI_FUNCTION_DIR
from cusadi.jax_ffi.function import CusadiJaxFunction

N = 10        # horizon
DT = 0.1      # timestep
FN_NAME = "opti_con_jac_jax"


def build_function() -> ca.Function:
    """Steps 1-2: formulate the double-integrator OCP with Opti, extract the
    sparse constraint Jacobian dg/dz, and collapse the (z, p) pair into a single
    input w = [z; p] so it satisfies the JAX backend's 1-in/1-out arity."""
    opti = ca.Opti()
    X = opti.variable(2, N + 1)      # state trajectory  [pos; vel]
    U = opti.variable(1, N)          # control trajectory
    x0 = opti.parameter(2)           # initial state -- varies per batch instance

    J = 0
    for k in range(N):
        xdot = ca.vertcat(X[1, k], U[0, k])
        opti.subject_to(X[:, k + 1] == X[:, k] + DT * xdot)   # dynamics defect
        J = J + X[0, k] ** 2 + U[0, k] ** 2
    opti.subject_to(X[:, 0] == x0)                            # initial condition
    opti.minimize(J)

    z = opti.x                       # decision vector
    p = opti.p                       # parameters
    Jg = ca.jacobian(opti.g, z)      # sparse, fixed structure

    # Single-input form: w = [z; p]. substitute() rewrites Jg (expressed in the
    # composite z, p) onto the fresh primitive w.
    w = ca.MX.sym("w", z.numel() + p.numel())
    Jg_w = ca.substitute(Jg, ca.vertcat(z, p), w)

    f = ca.Function(FN_NAME, [w], [Jg_w], ["w"], ["Jg"])

    # Opti builds an MXFunction; the level-parallel interpreter reads the scalar
    # SX instruction tape, so expand MX -> SX before handing it to cusadi.
    return f.expand()


def main():
    os.makedirs(CUSADI_FUNCTION_DIR, exist_ok=True)
    path = os.path.join(CUSADI_FUNCTION_DIR, f"{FN_NAME}.casadi")

    # --- Steps 1-2: build + save -----------------------------------------
    f = build_function()
    f.save(path)
    print(f"Saved: {path}")
    print(f"  {f}")
    print(f"  n_in={f.n_in()} n_out={f.n_out()} "
          f"n_instructions={f.n_instructions()} "
          f"nnz_out={f.nnz_out(0)} (sparse {f.size1_out(0)}x{f.size2_out(0)})")

    # --- Step 3: batch on the GPU via JAX --------------------------------
    f = ca.Function.load(path)        # reload to prove .casadi is the only input
    gfn = CusadiJaxFunction(f, plan_id=1)   # extracts plan, uploads to GPU

    batch = 4096
    nnz_in = f.nnz_in(0)
    W = jax.random.normal(jax.random.PRNGKey(0), (batch, nnz_in), dtype=jnp.float64)

    # XLA does the batching: vmap maps axis 0 -> kernel grid=(B,); jit compiles.
    batched = jax.jit(jax.vmap(gfn))
    vals = batched(W)                 # (batch, nnz_out[0])
    vals.block_until_ready()

    rows, cols = gfn.get_coo(0)       # JAX backend exposes the pattern only;
                                      # the values ARE the call output above.
    print(f"\nBatched {batch} instances through jax.vmap + FFI")
    print(f"  values: {tuple(vals.shape)} {vals.dtype} on {vals.device}")
    print(f"  Jg nnz pattern: {rows.shape[0]} entries")

    # Spot-check instance 0 against the CPU CasADi oracle.
    Jg_cpu = np.array(f(np.asarray(W[0])))
    dense = np.zeros(Jg_cpu.shape)
    dense[np.asarray(rows), np.asarray(cols)] = np.asarray(vals[0])
    print(f"  max |GPU - CasADi| on instance 0: {np.abs(dense - Jg_cpu).max():.2e}")

    gfn.close()                       # free the uploaded plan (plan_id=1)


if __name__ == "__main__":
    main()
