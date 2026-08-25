"""
Opti -> cusadi: end-to-end example.

Shows the full cusadi interface starting from a CasADi `Opti` problem:

  1. FORMULATE   build an OCP symbolically with the Opti stack.
  2. EXTRACT     pull a fixed-structure Function out of it (here the sparse
                 constraint Jacobian dg/dz) and save it as a .casadi file.
  3. BATCH       load that Function and evaluate it for thousands of parameter
                 instances at once on the GPU via CusadiFunction2.

Key boundary: cusadi batches a *fixed-structure* Function, never the solver.
You cannot batch `opti.to_function()` of the whole NLP solve -- that embeds an
nlpsol/IPOPT call as one instruction, which the level-parallel interpreter has
no opcode for. So you batch the building blocks (dynamics, constraint
Jacobians, Gauss-Newton J^T J Hessians) that the solver would call.

Run (single GPU; this repo's convention is device 1):
    CUDA_VISIBLE_DEVICES=1 python experiments/examples/opti_to_cusadi.py

The C++ file alongside this one (opti_to_cusadi.cpp) reproduces steps 1-2 --
the CasADi modeling side has a full C++ API. Step 3 (GPU batching) is
Python-only today; the .casadi file is the language-agnostic handoff.
"""

import os

import casadi as ca
import numpy as np
import torch

from cusadi import CUSADI_FUNCTION_DIR
from cusadi.pytorch.lp_unified import CusadiFunction2

N = 10        # horizon
DT = 0.1      # timestep
FN_NAME = "opti_con_jac"


def build_function() -> ca.Function:
    """Steps 1-2: formulate a double-integrator OCP with Opti and extract the
    sparse constraint Jacobian dg/dz as a batchable Function f(z, p) -> Jg."""
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

    # Opti exposes the flattened pieces of the NLP it just built:
    #   opti.x  decision vector,  opti.p  parameters,  opti.g  constraints.
    z = opti.x
    p = opti.p
    Jg = ca.jacobian(opti.g, z)      # sparse, fixed structure across instances

    f = ca.Function(FN_NAME, [z, p], [Jg], ["z", "p"], ["Jg"])

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

    # --- Step 3: batch on the GPU ----------------------------------------
    # Reload from disk to prove the .casadi file is the only thing the GPU
    # path needs (this is exactly what the C++ producer would hand off).
    f = ca.Function.load(path)
    gfn = CusadiFunction2(f)          # DAG analysis + LP kernel compile (~1.5s)
    print(gfn.dag_info)

    batch = 4096
    # One CUDA tensor per input, shape (batch, nnz_in[i]), float64, contiguous.
    z_in = torch.zeros(batch, f.nnz_in(0), device="cuda", dtype=torch.double)
    p_in = torch.randn(batch, f.nnz_in(1), device="cuda", dtype=torch.double)

    gfn.evaluate([z_in, p_in])        # mode="auto" -> lp or tiled
    rows, cols, vals = gfn.get_coo(0) # rows/cols: shared (nnz,) pattern;
                                      # vals: (batch, nnz) per-instance values
    torch.cuda.synchronize()

    print(f"\nBatched {batch} instances on '{gfn.dag_info.default_mode}' kernel "
          f"in {gfn.eval_time * 1e6:.1f} us")
    print(f"  Jg nnz pattern: {rows.shape[0]} entries")
    print(f"  values tensor:  {tuple(vals.shape)} on {vals.device}")

    # Spot-check instance 0 against the CPU CasADi oracle.
    Jg_cpu = np.array(f(z_in[0].cpu().numpy(), p_in[0].cpu().numpy()))
    gpu_dense = torch.sparse_coo_tensor(
        torch.vstack((rows, cols)), vals[0], Jg_cpu.shape
    ).to_dense().cpu().numpy()
    err = np.abs(gpu_dense - Jg_cpu).max()
    print(f"  max |GPU - CasADi| on instance 0: {err:.2e}")


if __name__ == "__main__":
    main()
