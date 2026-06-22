"""
Validate the level-parallel constrained-dynamics kernel against the CasADi CPU
oracle. Uses random but physically-plausible (q, v, tau) and samples num_instances
batch slots, comparing ddq_dq / ddq_dv / ddq_dtau output-by-output.

Usage:
    python exploration/constrained_dynamics/validate_constrained.py --robot a1
"""

from __future__ import annotations

import argparse
import os
import sys

import casadi as ca
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cusadi import CUSADI_FUNCTION_DIR  # noqa: E402
from cusadi.pytorch.lp_function import CusadiLevelParallelFunction  # noqa: E402
from cusadi.level_parallel.dag import extract_level_plan  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["a1", "h1"], required=True)
    ap.add_argument("--num_instances", type=int, default=8)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fn_name = f"{args.robot}_constrained_aba_jac"
    fn_path = os.path.join(CUSADI_FUNCTION_DIR, f"{fn_name}.casadi")
    f = ca.Function.load(fn_path)
    print(f"[validate] fn {fn_name}: n_instr={f.n_instructions():,} sz_w={f.sz_w()}")
    for i in range(f.n_in()):
        print(f"  in[{i}]={f.name_in(i)} shape={f.sparsity_in(i).shape}")
    for i in range(f.n_out()):
        print(f"  out[{i}]={f.name_out(i)} shape={f.sparsity_out(i).shape}")

    plan = extract_level_plan(f)
    smem_kb = plan.peak_live_slots * 8 / 1024.0
    print(f"[validate] DAG: ops={plan.total_compute_ops:,} levels={plan.n_compute_levels} "
          f"peak_live={plan.peak_live_slots} smem={smem_kb:.1f} KB")
    if smem_kb > 99:
        print(f"[validate] WARNING: peak_live_slots={plan.peak_live_slots} exceeds "
              f"L40S 99 KB per-block shared memory. Kernel launch will likely fail.",
              file=sys.stderr)

    # Random inputs — q needs to satisfy unit-quaternion constraint for the floating base.
    # We use CasADi's random input helper: q ~ sensible, v/tau ~ small normal.
    rng = np.random.default_rng(args.seed)
    N = args.num_instances

    # Build inputs as (N, nnz) torch tensors for the LP call, and as numpy
    # columns for CPU CasADi eval.
    q_all = np.zeros((N, f.nnz_in(0)))
    v_all = np.zeros((N, f.nnz_in(1)))
    tau_all = np.zeros((N, f.nnz_in(2)))
    for k in range(N):
        # Random unit-quaternion floating base: x,y,z free; qw,qx,qy,qz normalized.
        q = rng.normal(0.0, 0.3, size=f.nnz_in(0))
        qw, qx, qy, qz = rng.normal(size=4)
        nrm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        # Floating-base convention in pin: [x,y,z, qx,qy,qz,qw, joint_pos...]
        q[3:7] = np.array([qx, qy, qz, qw]) / nrm
        q_all[k] = q
        v_all[k] = rng.normal(0.0, 0.2, size=f.nnz_in(1))
        tau_all[k] = rng.normal(0.0, 1.0, size=f.nnz_in(2))

    # CPU reference
    cpu_outs = [[None] * N for _ in range(f.n_out())]
    for k in range(N):
        outs = f(q_all[k], v_all[k], tau_all[k])
        for j, o in enumerate(outs):
            cpu_outs[j][k] = np.asarray(o)

    # GPU level-parallel
    lp = CusadiLevelParallelFunction(f, N, lib_name="level_parallel_kernel")
    torch_inputs = [
        torch.tensor(q_all, dtype=torch.double, device="cuda").contiguous(),
        torch.tensor(v_all, dtype=torch.double, device="cuda").contiguous(),
        torch.tensor(tau_all, dtype=torch.double, device="cuda").contiguous(),
    ]
    lp.evaluate(torch_inputs)
    gpu_outs = [lp.outputs_sparse[j].detach().cpu().numpy() for j in range(f.n_out())]

    all_pass = True
    for j in range(f.n_out()):
        name = f.name_out(j)
        nnz = f.nnz_out(j)
        max_abs = 0.0
        max_rel = 0.0
        mismatches = 0
        for k in range(N):
            # CasADi stores dense outputs column-major; LP's outputs_sparse follows
            # the same convention. Default .flatten() is row-major, so flatten with
            # order="F" (Fortran / column-major) to match LP's storage order.
            cpu_flat = np.asarray(cpu_outs[j][k]).flatten(order="F")[:nnz]
            gpu_flat = gpu_outs[j][k].flatten()[:nnz]
            diff = np.abs(cpu_flat - gpu_flat)
            rel = diff / np.maximum(np.abs(cpu_flat), 1e-12)
            m = int(np.sum(diff > args.atol))
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(max_rel, float(rel.max()))
            mismatches += m
        ok = (max_abs <= args.atol)
        status = "PASS" if ok else "FAIL"
        print(f"  out[{j}] {name:<8s} shape={f.sparsity_out(j).shape} nnz={nnz}  "
              f"max|err|={max_abs:.2e}  max_rel={max_rel:.2e}  mismatches={mismatches}  {status}")
        all_pass = all_pass and ok

    print(f"\n[validate] {fn_name} overall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
