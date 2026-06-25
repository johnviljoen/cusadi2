"""Stage compilation benchmark — MPC-style horizon sweep.

Compares two ways of evaluating a horizon-N MPC residual on the GPU:

  unrolled-LP : build f_full(x_0..x_N, u_0..u_{N-1}, p) once via casadi.MX,
                pass the unrolled function to the standard level-parallel
                kernel. Shared memory and per-block work scale with N.

  stage-LP    : build f_stage(x_k, x_kp1, u_k, p) once, hand it to the
                stage-compiled kernel together with N. The DAG metadata is
                replicated across the (batch x N) launch grid; per-block
                shared memory stays at the per-stage budget.

Reports compile-extract time, peak shared memory, and runtime at fixed batch
across a horizon sweep. The headline observation: stage-LP shared memory is
flat in N while unrolled-LP grows linearly.
"""
from __future__ import annotations

import time
from typing import List, Optional

from experiments.validation._helpers import (benchmark_kernel_only, build_lp_fn,
                                  build_stage_lp_fn, print_header)

# Default sweep — small first, escalate to where unrolled-LP either tiles or
# exceeds smem entirely. Override via run_section_stage(n_stages=...).
N_STAGES_DEFAULT = [10, 25, 50, 100, 250]
NUM_INSTANCES_DEFAULT = 256


# ---------------------------------------------------------------------------
# A small but realistic stage program
# ---------------------------------------------------------------------------

def build_unicycle_stage():
    """3-state, 2-input unicycle dynamics + quadratic running cost gradient.

    Stage program f_stage(x_k, x_kp1, u_k, p) -> (defect, grad_z) where:
      x = (px, py, theta) ∈ R^3
      u = (v, omega)      ∈ R^2
      p = (Q diag (3), R diag (2), x_ref (3))  ∈ R^8     [shared across stages]
      defect = x_kp1 - (x_k + dt * f(x_k, u_k))           ∈ R^3
      grad_z = ∂/∂(x_k,u_k) [ ½(x_k-x_ref)ᵀ Q (x_k-x_ref) + ½ uᵀ R u ] ∈ R^5

    Small enough to fit comfortably in smem at any horizon, large enough to
    have non-trivial DAG depth.
    """
    import casadi as ca
    x = ca.SX.sym("x", 3)
    x_next = ca.SX.sym("x_next", 3)
    u = ca.SX.sym("u", 2)
    p = ca.SX.sym("p", 8)
    Q_diag = p[0:3]
    R_diag = p[3:5]
    x_ref = p[5:8]

    dt = 0.05
    theta = x[2]
    v, omega = u[0], u[1]
    f_dyn = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
    defect = x_next - (x + dt * f_dyn)

    z = ca.vertcat(x, u)
    e = x - x_ref
    cost = 0.5 * ca.dot(e, Q_diag * e) + 0.5 * ca.dot(u, R_diag * u)
    grad_z = ca.gradient(cost, z)

    f_stage = ca.Function(
        "unicycle_stage",
        [x, x_next, u, p],
        [defect, grad_z],
        ["x_k", "x_kp1", "u_k", "p"],
        ["defect", "grad_z"],
    )
    return f_stage


def build_unicycle_full(f_stage, n_stages: int):
    """Build the unrolled horizon-N residual by N times stamping f_stage,
    using a single CasADi function with all stage inputs concatenated."""
    import casadi as ca
    nx, nu, npar = 3, 2, 8

    X = ca.SX.sym("X", nx, n_stages + 1)
    U = ca.SX.sym("U", nu, n_stages)
    p = ca.SX.sym("p", npar)

    defects = []
    grads = []
    for k in range(n_stages):
        d_k, g_k = f_stage.call([X[:, k], X[:, k + 1], U[:, k], p])
        defects.append(d_k)
        grads.append(g_k)

    defect = ca.vertcat(*defects)        # 3*N
    grad_z = ca.vertcat(*grads)          # 5*N

    f_full = ca.Function(
        f"unicycle_full_N{n_stages}",
        [ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), p],
        [defect, grad_z],
        ["X", "U", "p"],
        ["defect", "grad_z"],
    )
    return f_full


# ---------------------------------------------------------------------------
# Buffer prep
# ---------------------------------------------------------------------------

def _make_stage_inputs(num_instances: int, n_stages: int):
    """(X, U, p) for the stage-LP path: shapes (B, N+1, 3), (B, N, 2), (B, 1, 8)."""
    import torch
    g = torch.Generator(device="cuda").manual_seed(42)
    X = torch.randn(num_instances, n_stages + 1, 3, device="cuda",
                    dtype=torch.double, generator=g).contiguous()
    U = torch.randn(num_instances, n_stages, 2, device="cuda",
                    dtype=torch.double, generator=g).contiguous()
    p = torch.randn(num_instances, 1, 8, device="cuda",
                    dtype=torch.double, generator=g).contiguous()
    return X, U, p


def _make_full_inputs(num_instances: int, X, U, p):
    """Flatten (X, U, p) into the unrolled function's input shapes
    (B, 3*(N+1)), (B, 2*N), (B, 8)."""
    X_flat = X.reshape(num_instances, -1).contiguous()
    U_flat = U.reshape(num_instances, -1).contiguous()
    p_flat = p.reshape(num_instances, -1).contiguous()
    return [X_flat, U_flat, p_flat]


def _stage_specs():
    from cusadi.level_parallel.stage import StageMode, StageSpec
    in_specs = [
        StageSpec(StageMode.COUPLED, offset=0, span=1),  # x_k
        StageSpec(StageMode.COUPLED, offset=1, span=1),  # x_kp1
        StageSpec(StageMode.STAGE),                      # u_k
        StageSpec(StageMode.SHARED),                     # p
    ]
    out_specs = [StageSpec(StageMode.STAGE), StageSpec(StageMode.STAGE)]
    return in_specs, out_specs


# ---------------------------------------------------------------------------
# Numerical equivalence check (run once at the smallest N)
# ---------------------------------------------------------------------------

def _verify_equivalence(f_stage, num_instances: int, n_stages: int,
                         atol: float = 1e-10) -> None:
    """Compare stage-LP output against unrolled-LP output element-wise."""
    X, U, p = _make_stage_inputs(num_instances, n_stages)
    in_specs, out_specs = _stage_specs()
    stage_lp, _ = build_stage_lp_fn(f_stage, num_instances, n_stages,
                                     in_specs, out_specs)
    stage_lp.evaluate([X, X, U, p])
    stage_defect = stage_lp.outputs_sparse[0].clone()  # (B, N, 3)
    stage_grad   = stage_lp.outputs_sparse[1].clone()  # (B, N, 5)

    f_full = build_unicycle_full(f_stage, n_stages)
    full_lp, _, _ = build_lp_fn(f_full, num_instances)
    full_lp.evaluate(_make_full_inputs(num_instances, X, U, p))
    # Unrolled output shapes: (B, 3*N) and (B, 5*N) flat — reshape to compare.
    full_defect = full_lp.outputs_sparse[0].view(num_instances, n_stages, 3)
    full_grad   = full_lp.outputs_sparse[1].view(num_instances, n_stages, 5)

    err_d = (stage_defect - full_defect).abs().max().item()
    err_g = (stage_grad   - full_grad  ).abs().max().item()
    print(f"  numerical check (N={n_stages}): "
          f"max|defect_err|={err_d:.2e}  max|grad_err|={err_g:.2e}")
    if err_d > atol or err_g > atol:
        raise RuntimeError(
            f"Stage-LP vs unrolled-LP mismatch: "
            f"max|defect|={err_d:.2e}, max|grad|={err_g:.2e}, atol={atol}")


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

def run_section_stage(
    n_stages_list: Optional[List[int]] = None,
    num_instances: int = NUM_INSTANCES_DEFAULT,
    n_warmup: int = 10,
    n_timed: int = 100,
    skip_unrolled_above: int = 250,
    verify: bool = True,
) -> dict:
    """Sweep over horizon length, comparing stage-LP vs unrolled-LP.

    Args:
        n_stages_list: horizons to evaluate. Defaults to N_STAGES_DEFAULT.
        num_instances: batch size.
        skip_unrolled_above: horizon above which to skip the unrolled baseline
            (its level-plan extraction time grows with N).
        verify: run numerical equivalence check at the smallest horizon.
    """
    if n_stages_list is None:
        n_stages_list = N_STAGES_DEFAULT
    print_header("STAGE COMPILATION : horizon sweep on unicycle MPC residual")
    print(f"  num_instances = {num_instances}")
    print(f"  f_stage:")
    f_stage = build_unicycle_stage()
    print(f"    n_in={f_stage.n_in()}, n_out={f_stage.n_out()}, "
          f"n_instructions={f_stage.n_instructions():,}")

    if verify:
        _verify_equivalence(f_stage, num_instances=8,
                            n_stages=min(n_stages_list))

    print()
    print(f"  {'N':>5}  {'stage_smemKB':>12}  {'stage_us':>10}  "
          f"{'full_extract_s':>14}  {'full_smemKB':>12}  {'full_us':>10}  "
          f"{'speedup':>8}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*8}")

    in_specs, out_specs = _stage_specs()
    out = {}
    for n in n_stages_list:
        # ---- stage-LP ----
        X, U, p = _make_stage_inputs(num_instances, n)
        t0 = time.time()
        stage_lp, stage_plan = build_stage_lp_fn(
            f_stage, num_instances, n, in_specs, out_specs)
        stage_extract_s = time.time() - t0
        stage_smem_kb = stage_plan.peak_live_slots * 8 / 1024.0
        stage_us = benchmark_kernel_only(
            lambda: stage_lp.evaluate([X, X, U, p]),
            lambda: stage_lp.eval_time, n_warmup, n_timed)

        # ---- unrolled-LP ----
        full_extract_s = float("nan")
        full_smem_kb = float("nan")
        full_us = float("nan")
        speedup = float("nan")
        mode = None
        if n <= skip_unrolled_above:
            t0 = time.time()
            f_full = build_unicycle_full(f_stage, n)
            full_lp, full_plan, mode = build_lp_fn(f_full, num_instances)
            full_extract_s = time.time() - t0
            full_smem_kb = full_plan.peak_live_slots * 8 / 1024.0
            full_inputs = _make_full_inputs(num_instances, X, U, p)
            full_us = benchmark_kernel_only(
                lambda: full_lp.evaluate(full_inputs),
                lambda: full_lp.eval_time, n_warmup, n_timed)
            speedup = full_us / stage_us if stage_us > 0 else float("nan")
            speedup_str = f"{speedup:.2f}x"
            full_extract_str = f"{full_extract_s:14.3f}"
            full_smem_str = f"{full_smem_kb:12.1f}"
            full_us_str = f"{full_us:10.1f}"
            # Drop the unrolled artifacts before the next iteration —
            # they grow with N.
            del full_lp, f_full
        else:
            speedup_str = "skipped"
            full_extract_str = "        skipped"
            full_smem_str = "     skipped"
            full_us_str = "   skipped"

        print(f"  {n:>5}  {stage_smem_kb:12.1f}  {stage_us:10.1f}  "
              f"{full_extract_str}  {full_smem_str}  {full_us_str}  "
              f"{speedup_str:>8}")

        out[str(n)] = {
            "stage_extract_s": stage_extract_s,
            "stage_smem_kb": stage_smem_kb,
            "stage_us": stage_us,
            "full_extract_s": full_extract_s,
            "full_smem_kb": full_smem_kb,
            "full_us": full_us,
            "full_mode": mode,
            "speedup": speedup,
        }

        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass

    return out


if __name__ == "__main__":
    run_section_stage()
