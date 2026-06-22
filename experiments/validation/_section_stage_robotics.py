"""Stage compilation benchmark on heavy (robotics-class) stage programs.

Where the unicycle benchmark probes the regime in which per-block fixed
costs dominate (so unrolled-LP wins by fusing across stages), this file
probes the opposite regime: each stage already saturates an SM and nearly
fills shared memory, so fusing N copies is what hurts you. Stage-LP keeps
each block at the per-stage budget; unrolled-LP hits the tiled kernel
fast and pays spill/reload on every stage boundary.

f_stage is just the robot's analytical ABA Jacobian — three inputs
(q, v, tau), three outputs (∂a/∂q, ∂a/∂v, ∂a/∂tau). All inputs use
StageMode.STAGE (no inter-stage coupling). The benchmark sweeps N and
reports stage-LP and unrolled-LP runtime side by side.
"""
from __future__ import annotations

import time
from typing import List, Optional

from experiments.validation._helpers import (benchmark_kernel_only, build_lp_fn,
                                  build_stage_lp_fn, load_casadi_fn,
                                  print_header)


def build_unrolled_aba_jac(f_stage, n_stages: int):
    """Stamp f_stage N times into a single CasADi function."""
    import casadi as ca
    nq, nv = f_stage.nnz_in(0), f_stage.nnz_in(1)

    Q = ca.SX.sym("Q", nq, n_stages)
    V = ca.SX.sym("V", nv, n_stages)
    T = ca.SX.sym("T", nv, n_stages)

    outs = [[], [], []]
    for k in range(n_stages):
        ddq_dq, ddq_dv, ddq_dtau = f_stage.call([Q[:, k], V[:, k], T[:, k]])
        outs[0].append(ddq_dq)
        outs[1].append(ddq_dv)
        outs[2].append(ddq_dtau)

    out0 = ca.vertcat(*outs[0])
    out1 = ca.vertcat(*outs[1])
    out2 = ca.vertcat(*outs[2])

    return ca.Function(
        f"{f_stage.name()}_unrolled_N{n_stages}",
        [ca.reshape(Q, -1, 1), ca.reshape(V, -1, 1), ca.reshape(T, -1, 1)],
        [out0, out1, out2],
        ["Q", "V", "T"],
        ["d_q", "d_v", "d_t"],
    )


def _make_stage_inputs(num_instances: int, n_stages: int, nq: int, nv: int):
    import torch
    g = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(num_instances, n_stages, nq, device="cuda",
                    dtype=torch.double, generator=g).contiguous()
    v = torch.randn(num_instances, n_stages, nv, device="cuda",
                    dtype=torch.double, generator=g).contiguous()
    t = torch.randn(num_instances, n_stages, nv, device="cuda",
                    dtype=torch.double, generator=g).contiguous()
    return q, v, t


def _make_full_inputs(num_instances: int, q, v, t):
    return [q.reshape(num_instances, -1).contiguous(),
            v.reshape(num_instances, -1).contiguous(),
            t.reshape(num_instances, -1).contiguous()]


def _stage_specs():
    from cusadi.level_parallel.stage import StageMode, StageSpec
    in_specs = [StageSpec(StageMode.STAGE)] * 3
    out_specs = [StageSpec(StageMode.STAGE)] * 3
    return in_specs, out_specs


def _verify(f_stage, num_instances: int, n_stages: int,
            atol: float = 1e-9) -> None:
    """Stage-LP vs unrolled-LP elementwise check."""
    nq, nv = f_stage.nnz_in(0), f_stage.nnz_in(1)
    q, v, t = _make_stage_inputs(num_instances, n_stages, nq, nv)
    in_specs, out_specs = _stage_specs()

    stage_lp, _ = build_stage_lp_fn(f_stage, num_instances, n_stages,
                                     in_specs, out_specs)
    stage_lp.evaluate([q, v, t])
    s_outs = [stage_lp.outputs_sparse[i].clone() for i in range(3)]

    f_full = build_unrolled_aba_jac(f_stage, n_stages)
    full_lp, _, _ = build_lp_fn(f_full, num_instances)
    full_lp.evaluate(_make_full_inputs(num_instances, q, v, t))
    f_outs = [
        full_lp.outputs_sparse[i].view(num_instances, n_stages, -1)
        for i in range(3)
    ]

    errs = [(s_outs[i] - f_outs[i]).abs().max().item() for i in range(3)]
    print(f"  numerical check (N={n_stages}): "
          f"max|err|=[{errs[0]:.2e}, {errs[1]:.2e}, {errs[2]:.2e}]")
    if max(errs) > atol:
        raise RuntimeError(f"mismatch: {errs}")


def run_section_stage_robotics(
    fn_name: str = "a1_aba_jac_analytical",
    n_stages_list: Optional[List[int]] = None,
    num_instances: int = 256,
    n_warmup: int = 5,
    n_timed: int = 30,
    skip_unrolled_above: int = 10,
    skip_unrolled_extract_seconds: float = 120.0,
    verify: bool = True,
) -> dict:
    if n_stages_list is None:
        n_stages_list = [1, 2, 3, 5, 10]

    print_header(f"STAGE COMPILATION (robotics) : {fn_name}")
    f_stage = load_casadi_fn(fn_name)
    nq, nv = f_stage.nnz_in(0), f_stage.nnz_in(1)
    print(f"  f_stage: nq={nq}, nv={nv}, n_in={f_stage.n_in()}, "
          f"n_out={f_stage.n_out()}, n_instructions={f_stage.n_instructions():,}")
    print(f"  num_instances = {num_instances}")

    if verify:
        _verify(f_stage, num_instances=4, n_stages=min(n_stages_list))

    print()
    print(f"  {'N':>4}  {'stage_smemKB':>12}  {'stage_us':>10}  "
          f"{'full_extract_s':>14}  {'full_smemKB':>12}  {'full_mode':>10}  "
          f"{'full_us':>10}  {'speedup':>8}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*12}  "
          f"{'-'*10}  {'-'*10}  {'-'*8}")

    in_specs, out_specs = _stage_specs()
    out = {}
    skip_remaining_unrolled = False
    for n in n_stages_list:
        # ---- stage-LP ----
        q, v, t = _make_stage_inputs(num_instances, n, nq, nv)
        t0 = time.time()
        stage_lp, stage_plan = build_stage_lp_fn(
            f_stage, num_instances, n, in_specs, out_specs)
        stage_extract_s = time.time() - t0
        stage_smem_kb = stage_plan.peak_live_slots * 8 / 1024.0
        stage_us = benchmark_kernel_only(
            lambda: stage_lp.evaluate([q, v, t]),
            lambda: stage_lp.eval_time, n_warmup, n_timed)
        del stage_lp

        # ---- unrolled-LP ----
        full_extract_s = float("nan")
        full_smem_kb = float("nan")
        full_us = float("nan")
        speedup = float("nan")
        mode = None

        if not skip_remaining_unrolled and n <= skip_unrolled_above:
            t0 = time.time()
            try:
                f_full = build_unrolled_aba_jac(f_stage, n)
                full_lp, full_plan, mode = build_lp_fn(f_full, num_instances)
                full_extract_s = time.time() - t0
                if full_extract_s > skip_unrolled_extract_seconds:
                    skip_remaining_unrolled = True
                full_smem_kb = full_plan.peak_live_slots * 8 / 1024.0
                full_inputs = _make_full_inputs(num_instances, q, v, t)
                full_us = benchmark_kernel_only(
                    lambda: full_lp.evaluate(full_inputs),
                    lambda: full_lp.eval_time, n_warmup, n_timed)
                speedup = full_us / stage_us if stage_us > 0 else float("nan")
                speedup_str = f"{speedup:.2f}x"
                full_extract_str = f"{full_extract_s:14.3f}"
                full_smem_str = f"{full_smem_kb:12.1f}"
                mode_str = f"{mode:>10}"
                full_us_str = f"{full_us:10.1f}"
                del full_lp, f_full
            except Exception as e:
                full_extract_s = time.time() - t0
                speedup_str = f"FAIL"
                full_extract_str = f"{full_extract_s:14.3f}"
                full_smem_str = "     ERR"
                mode_str = "       ERR"
                full_us_str = "   ERR"
                print(f"    unrolled build/run failed at N={n}: "
                      f"{type(e).__name__}: {str(e)[:80]}")
                skip_remaining_unrolled = True
        else:
            speedup_str = "skipped"
            full_extract_str = "        skipped"
            full_smem_str = "     skipped"
            mode_str = "   skipped"
            full_us_str = "   skipped"

        print(f"  {n:>4}  {stage_smem_kb:12.1f}  {stage_us:10.1f}  "
              f"{full_extract_str}  {full_smem_str}  {mode_str}  "
              f"{full_us_str}  {speedup_str:>8}")

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
    import sys
    fn = sys.argv[1] if len(sys.argv) > 1 else "a1_aba_jac_analytical"
    run_section_stage_robotics(fn_name=fn)
