"""LP-vs-no-LP ablation: how much does within-block parallelism matter?

The 2×2 ablation: {staging on/off} × {LP on/off}, at fixed total work
(B × n_stages thread groups). For one representative function (A1 ABA
Jacobian, 41k instructions, all STAGE inputs, n_stages=24 — a typical MPC
horizon), sweep batch size and time four configurations:

    1. stage + LP:     chunk kernel, block_size = pick_block_size(plan).
                       grid = (B, n_chunks); threads cooperate within
                       each block on the chunk DAG. Our system.

    2. unrolled + LP:  LP kernel on the unrolled horizon function
                       (24 stamps of f_stage). grid = (B,); one block
                       per batch instance, LP threads cooperate on
                       the unrolled DAG. Auto-tiles if peak smem
                       exceeds device budget.
                       (= "original cusadi codegen pipeline + LP";
                        the prior level-parallel baseline before
                        stage compilation.)

    3. chunk @ bs=1:   chunk kernel forced to block_size=1 — within-
                       kernel ablation of LP. Pays the same per-block
                       smem allocation as stage+LP, with only one
                       thread doing the work. Not deployable.

    4. original seq:   original cusadi sequential kernel on f_stage at
                       batch B × n_stages — i.e., stage axis flattened
                       into batch. One thread per (instance, stage),
                       AOT-compiled inline arithmetic, no shared memory.
                       Prior state-of-the-art deployable baseline.

Plot intent: total runtime (µs) vs batch size B at fixed n_stages, log-log.
Tracks how much each axis (staging, LP) contributes at different batch sizes.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m validation._section_lp_ablation
"""
from __future__ import annotations

import os
from typing import List, Optional

from validation._helpers import (benchmark_kernel_only,
                                  build_lp_fn,
                                  load_casadi_fn,
                                  print_header,
                                  repo_path,
                                  SequentialFn)
from validation._section_stage_robotics import build_unrolled_aba_jac
from src.level_parallel.stage import (StageMode, StageSpec,
                                       CusadiHybridStageLevelParallelFunction,
                                       smart_pick_chunk_size)
from src.level_parallel.dag import extract_level_plan


FN_NAME = "a1_aba_jac_analytical"
N_STAGES = 24
BATCHES = [1, 8, 32, 128, 256, 1024, 4096]
SEQ_BUILD_DIR = repo_path("build")


def _make_inputs(f_stage, num_instances, n_stages, *, with_stage_axis: bool):
    """Random inputs for either stage-axis wrappers (B, N, nnz) or original
    sequential cusadi (B*N, nnz) — same total work either way."""
    import torch
    n_in = f_stage.n_in()
    g = torch.Generator(device="cuda").manual_seed(42)
    if with_stage_axis:
        return [torch.randn(num_instances, n_stages, f_stage.nnz_in(i),
                            device="cuda", dtype=torch.double, generator=g).contiguous()
                for i in range(n_in)]
    else:
        return [torch.randn(num_instances * n_stages, f_stage.nnz_in(i),
                            device="cuda", dtype=torch.double, generator=g).contiguous()
                for i in range(n_in)]


def _build_stage_lp(f_stage, B, n_stages, per_stage_plan,
                     *, block_size_override: Optional[int]) -> CusadiHybridStageLevelParallelFunction:
    in_specs = [StageSpec(StageMode.STAGE)] * f_stage.n_in()
    out_specs = [StageSpec(StageMode.STAGE)] * f_stage.n_out()
    decision = smart_pick_chunk_size(
        f_stage, n_stages, in_specs, per_stage_plan=per_stage_plan)
    chunk_size = decision.chunk_size
    return CusadiHybridStageLevelParallelFunction(
        f_stage, B, n_stages,
        input_specs=in_specs, output_specs=out_specs,
        chunk_size=chunk_size,
        lib_name="level_parallel_chunk_kernel",
        per_stage_plan=per_stage_plan,
        block_size_override=block_size_override,
    )


def run_section_lp_ablation(
    fn_name: str = FN_NAME,
    n_stages: int = N_STAGES,
    batches: Optional[List[int]] = None,
    n_warmup: int = 5,
    n_timed: int = 30,
    skip_sequential: bool = False,
    skip_unrolled_lp: bool = False,
) -> dict:
    import time
    if batches is None:
        batches = BATCHES
    print_header(f"LP-vs-no-LP ABLATION (2x2): {fn_name} (N={n_stages})")
    f_stage = load_casadi_fn(fn_name)
    per_stage_plan = extract_level_plan(f_stage)
    print(f"  per_stage: {f_stage.n_instructions():,} instr, "
          f"{per_stage_plan.n_compute_levels} levels, "
          f"peak_live={per_stage_plan.peak_live_slots} slots "
          f"({per_stage_plan.peak_live_slots*8/1024:.1f} KB)")
    print(f"  median per-level width = {per_stage_plan.median_width}, "
          f"max = {per_stage_plan.max_width}")

    # Pre-build the unrolled function once (independent of B).
    f_unrolled = None
    unrolled_plan = None
    unrolled_mode = None
    if not skip_unrolled_lp:
        print(f"  building unrolled function (stamping f_stage × {n_stages})...",
              flush=True, end=" ")
        t0 = time.time()
        f_unrolled = build_unrolled_aba_jac(f_stage, n_stages)
        unrolled_plan = extract_level_plan(f_unrolled)
        print(f"done in {time.time()-t0:.1f}s")
        print(f"    unrolled: {f_unrolled.n_instructions():,} instr, "
              f"{unrolled_plan.n_compute_levels} levels, "
              f"peak_live={unrolled_plan.peak_live_slots} slots "
              f"({unrolled_plan.peak_live_slots*8/1024:.1f} KB)")
        if unrolled_plan.peak_live_slots * 8 > 12000 * 8:
            print(f"    (will use TILED LP — peak smem > 96 KB device limit)")

    print()
    print(f"  {'B':>6}  {'stage+LP us':>12}  {'unrolled+LP us':>15}  "
          f"{'chunk@bs=1 us':>14}  {'orig seq us':>13}")
    print('-'*78)

    out: dict = {}
    seq_avail = (not skip_sequential and
                 os.path.exists(os.path.join(SEQ_BUILD_DIR, f"lib{fn_name}.so")))
    if not seq_avail:
        print(f"  (sequential .so missing for {fn_name}; "
              f"skipping that column. Build via run_paper_comparison.py "
              f"if you need it.)")

    for B in batches:
        # --- 1. stage + LP (default block_size) ---
        inputs_stage = _make_inputs(f_stage, B, n_stages, with_stage_axis=True)
        lp_full = _build_stage_lp(f_stage, B, n_stages, per_stage_plan,
                                    block_size_override=None)
        us_full = benchmark_kernel_only(
            lambda: lp_full.evaluate(inputs_stage),
            lambda: lp_full.eval_time, n_warmup, n_timed)
        del lp_full

        # --- 2. unrolled + LP (LP kernel on stamped function, no stage axis) ---
        if f_unrolled is not None:
            # Reshape stage-axis inputs (B, N, nnz) into the unrolled
            # function's flat (B, N*nnz) signature.
            inputs_unr = [t.reshape(B, -1).contiguous() for t in inputs_stage]
            lp_unr, _, _ = build_lp_fn(f_unrolled, B, max_smem_slots=12000)
            us_unr = benchmark_kernel_only(
                lambda: lp_unr.evaluate(inputs_unr),
                lambda: lp_unr.eval_time, n_warmup, n_timed)
            del lp_unr
            us_unr_str = f"{us_unr:15.1f}"
        else:
            us_unr = float("nan")
            us_unr_str = "          skip "

        # --- 3. chunk @ bs=1 (within-kernel ablation) ---
        lp_seq = _build_stage_lp(f_stage, B, n_stages, per_stage_plan,
                                   block_size_override=1)
        us_seq = benchmark_kernel_only(
            lambda: lp_seq.evaluate(inputs_stage),
            lambda: lp_seq.eval_time, n_warmup, n_timed)
        del lp_seq

        # --- 4. original sequential cusadi (1 thread per (instance, stage)) ---
        if seq_avail:
            inputs_flat = _make_inputs(f_stage, B, n_stages,
                                        with_stage_axis=False)
            seq_fn = SequentialFn(f_stage, B * n_stages, SEQ_BUILD_DIR)
            seq_fn.set_inputs(inputs_flat)
            us_orig = benchmark_kernel_only(
                seq_fn.evaluate,
                lambda: seq_fn.eval_time, n_warmup, n_timed)
            del seq_fn
            us_orig_str = f"{us_orig:13.1f}"
        else:
            us_orig = float("nan")
            us_orig_str = "         skip"

        print(f"  {B:>6}  {us_full:>12.1f}  {us_unr_str}  "
              f"{us_seq:>14.1f}  {us_orig_str}")

        out[str(B)] = {
            "stage_LP_us":      us_full,
            "unrolled_LP_us":   us_unr,
            "stage_only_us":    us_seq,
            "sequential_us":    us_orig,
        }

        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass

    return out


if __name__ == "__main__":
    out = run_section_lp_ablation()
    # Optional: write a quick ASCII plot to stdout
    try:
        import math
        print()
        print("ASCII log-log overview (us, lower is better):")
        print(f"  {'B':>6}  " + "stage+LP".rjust(11) + "  " + "stage-only".rjust(11)
              + "  " + "sequential".rjust(11))
        for B, row in out.items():
            def fmt(x):
                return f"{x:.1f}" if x and not math.isnan(x) else "—"
            print(f"  {B:>6}  {fmt(row['stage_LP_us']):>11}  "
                  f"{fmt(row['stage_only_us']):>11}  "
                  f"{fmt(row['sequential_us']):>11}")
    except Exception:
        pass
