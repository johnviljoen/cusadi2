"""Hybrid stage compilation benchmark — chunk_size sweep on heavy stages.

For each (fn_name, n_stages) combination, benchmarks the chunked stage
kernel at every chunk_size dividing n_stages, plus the auto-tuned choice.
Reports peak shared memory and runtime so the chunk_size optimum is
visible empirically — and so the auto-tuner's pick can be validated.

Inputs (q, v, tau) use StageMode.STAGE — independent stages, no coupling.
This isolates the chunk-fusion vs. chunk-parallelism trade-off cleanly.
"""
from __future__ import annotations

import time
from typing import List, Optional

from validation._helpers import (benchmark_kernel_only,
                                  build_hybrid_stage_lp_fn,
                                  load_casadi_fn,
                                  print_header)
from validation._section_stage_robotics import (_make_stage_inputs,
                                                 _stage_specs)
from src.level_parallel.stage import (_get_device_smem_limit_bytes,
                                       _build_chunk_function)
from src.level_parallel.dag import extract_level_plan


def run_section_stage_hybrid(
    fn_name: str = "a1_aba_jac_analytical",
    n_stages: int = 12,
    num_instances: int = 256,
    n_warmup: int = 5,
    n_timed: int = 30,
    chunk_sizes: Optional[List[int]] = None,
    verify: bool = True,
) -> dict:
    if chunk_sizes is None:
        chunk_sizes = [d for d in range(1, n_stages + 1) if n_stages % d == 0]

    print_header(f"HYBRID STAGE COMPILATION : {fn_name}, N={n_stages}")
    f_stage = load_casadi_fn(fn_name)
    nq, nv = f_stage.nnz_in(0), f_stage.nnz_in(1)
    print(f"  f_stage: nq={nq}, nv={nv}, n_instructions={f_stage.n_instructions():,}")
    print(f"  num_instances={num_instances}, n_stages={n_stages}")
    print(f"  chunk_sizes to sweep: {chunk_sizes}")

    in_specs, out_specs = _stage_specs()
    q, v, t = _make_stage_inputs(num_instances, n_stages, nq, nv)

    # Filter out chunk sizes that would overflow device shared memory. We
    # check against the chunked plan's actual peak rather than the c × per_stage
    # upper bound, so the sweep includes any c that fits regardless of fusion
    # benefits.
    smem_budget = _get_device_smem_limit_bytes() - 4096
    max_slots = smem_budget // 8
    feasible = []
    for c in chunk_sizes:
        try:
            f_chunk = _build_chunk_function(f_stage, c, in_specs)
            cp = extract_level_plan(f_chunk)
            if cp.peak_live_slots <= max_slots:
                feasible.append(c)
            else:
                print(f"  skipping chunk_size={c}: would need "
                      f"{cp.peak_live_slots*8/1024:.0f}KB > {smem_budget/1024:.0f}KB budget")
        except Exception as e:
            print(f"  skipping chunk_size={c}: {type(e).__name__}: {e}")

    if not feasible:
        print("  no feasible chunk sizes — aborting")
        return {}

    chunk_sizes = feasible
    print(f"  feasible chunk_sizes: {chunk_sizes}")

    # Reference: pure stage-LP (chunk=1) — already verified bit-exact in
    # _section_stage_robotics; we re-use it as the chunk=1 reference.
    ref_outs: Optional[List] = None
    print()
    print(f"  {'c':>4}  {'n_chunks':>9}  {'extract_s':>10}  {'smemKB':>8}  "
          f"{'us':>10}  {'us/stage':>10}  {'err':>10}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

    out = {}
    for c in chunk_sizes:
        t0 = time.time()
        hyb, plan, csz = build_hybrid_stage_lp_fn(
            f_stage, num_instances, n_stages,
            in_specs, out_specs, chunk_size=c)
        extract_s = time.time() - t0
        smem_kb = plan.peak_live_slots * 8 / 1024.0
        us = benchmark_kernel_only(
            lambda: hyb.evaluate([q, v, t]),
            lambda: hyb.eval_time, n_warmup, n_timed)

        # Numerical equivalence vs c=1
        if ref_outs is None:
            ref_outs = [hyb.outputs_sparse[i].clone() for i in range(3)]
            err = 0.0
        elif verify:
            err = max((hyb.outputs_sparse[i] - ref_outs[i]).abs().max().item()
                      for i in range(3))
        else:
            err = float("nan")

        per_stage = us / n_stages
        print(f"  {c:>4}  {n_stages // c:>9}  {extract_s:>10.3f}  "
              f"{smem_kb:>8.1f}  {us:>10.1f}  {per_stage:>10.2f}  "
              f"{err:>10.2e}")

        out[str(c)] = {
            "n_chunks": n_stages // c,
            "extract_s": extract_s,
            "smem_kb": smem_kb,
            "us": us,
            "us_per_stage": per_stage,
            "err": err,
        }

        del hyb
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass

    # Auto-tuned pick
    print()
    t0 = time.time()
    hyb_auto, plan_auto, c_auto = build_hybrid_stage_lp_fn(
        f_stage, num_instances, n_stages,
        in_specs, out_specs, chunk_size="auto")
    auto_extract_s = time.time() - t0
    auto_us = benchmark_kernel_only(
        lambda: hyb_auto.evaluate([q, v, t]),
        lambda: hyb_auto.eval_time, n_warmup, n_timed)
    print(f"  auto-tuned chunk_size = {c_auto}  "
          f"(extract={auto_extract_s:.3f}s, smem={plan_auto.peak_live_slots*8/1024:.1f}KB, "
          f"runtime={auto_us:.1f}µs)")

    out["auto"] = {
        "chunk_size": c_auto,
        "extract_s": auto_extract_s,
        "smem_kb": plan_auto.peak_live_slots * 8 / 1024.0,
        "us": auto_us,
    }
    return out


if __name__ == "__main__":
    import sys
    fn = sys.argv[1] if len(sys.argv) > 1 else "a1_aba_jac_analytical"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    run_section_stage_hybrid(fn_name=fn, n_stages=n)
