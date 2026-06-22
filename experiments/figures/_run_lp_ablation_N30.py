"""Re-run the LP ablation at N=30 and save the results JSON for the
defense plot. One-shot helper; not part of the validation suite.

At N=30, the unrolled-horizon function's tiled-LP kernel exceeds the
device's per-block smem budget (99 KB on L40S) — running it raises
cudaErrorInvalidValue from the launch. We catch that, mark
unrolled+LP as unbuildable, and persist the remaining 3 curves. The
absence is itself a defense-talk data point.
"""
from __future__ import annotations

import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from experiments.validation._helpers import (benchmark_kernel_only, load_casadi_fn,
                                  repo_path, SequentialFn)
from cusadi.level_parallel.codegen import pick_block_size
from cusadi.level_parallel.dag import extract_level_plan
from cusadi.level_parallel.stage import (StageMode, StageSpec,
                                       CusadiHybridStageLevelParallelFunction,
                                       smart_pick_chunk_size)

FN_NAME  = "a1_aba_jac_analytical"
N_STAGES = 30
BATCHES  = [1, 8, 32, 128, 256, 1024, 4096]
SEQ_BUILD_DIR = repo_path("build")
OUT_JSON = os.path.join(THIS_DIR, f"lp_ablation_N{N_STAGES}.json")


def _make_inputs(f_stage, B, n_stages, *, with_stage_axis):
    import torch
    g = torch.Generator(device="cuda").manual_seed(42)
    if with_stage_axis:
        return [torch.randn(B, n_stages, f_stage.nnz_in(i),
                            device="cuda", dtype=torch.double, generator=g).contiguous()
                for i in range(f_stage.n_in())]
    return [torch.randn(B * n_stages, f_stage.nnz_in(i),
                        device="cuda", dtype=torch.double, generator=g).contiguous()
            for i in range(f_stage.n_in())]


def _build_stage_lp(f_stage, B, n_stages, per_stage_plan, *, block_size_override):
    in_specs  = [StageSpec(StageMode.STAGE)] * f_stage.n_in()
    out_specs = [StageSpec(StageMode.STAGE)] * f_stage.n_out()
    decision = smart_pick_chunk_size(
        f_stage, n_stages, in_specs, per_stage_plan=per_stage_plan)
    return CusadiHybridStageLevelParallelFunction(
        f_stage, B, n_stages,
        input_specs=in_specs, output_specs=out_specs,
        chunk_size=decision.chunk_size,
        lib_name="level_parallel_chunk_kernel",
        per_stage_plan=per_stage_plan,
        block_size_override=block_size_override,
    ), decision


def main():
    f_stage = load_casadi_fn(FN_NAME)
    per_stage_plan = extract_level_plan(f_stage)
    in_specs = [StageSpec(StageMode.STAGE)] * f_stage.n_in()
    decision = smart_pick_chunk_size(
        f_stage, N_STAGES, in_specs, per_stage_plan=per_stage_plan)

    print(f"=== LP ablation N={N_STAGES} on {FN_NAME} ===")
    print(f"per_stage: {f_stage.n_instructions():,} instr, "
          f"{per_stage_plan.n_compute_levels} levels, "
          f"peak_live={per_stage_plan.peak_live_slots} slots "
          f"({per_stage_plan.peak_live_slots*8/1024:.1f} KB)")
    block_size = pick_block_size(per_stage_plan)
    print(f"chunk_size={decision.chunk_size}, block_size={block_size}")

    seq_avail = os.path.exists(
        os.path.join(SEQ_BUILD_DIR, f"lib{FN_NAME}.so"))
    if not seq_avail:
        print(f"WARN sequential .so missing for {FN_NAME}")

    # At N=30 the unrolled-horizon function's tiled-LP kernel hits
    # cudaErrorInvalidValue (GPUassert at level_parallel_tiled_kernel.cu:254)
    # because per-chunk smem need + spill metadata exceeds the L40S
    # 99 KB per-block limit. The assert calls exit(), so we cannot
    # try/except — we predict and skip. Recorded as a missing 2×2 corner.
    unrolled_buildable = False
    f_unr = None
    unr_info = {
        "skipped": True,
        "reason": ("unrolled+LP at N=30 raises cudaErrorInvalidValue "
                   "from level_parallel_tiled_kernel.cu:254 — per-chunk "
                   "smem requirement exceeds L40S 99 KB per-block limit. "
                   "Verified on device: at N=12 this path works (peak "
                   "412 KB → tiled OK); at N≥24 it does not."),
    }
    print(f"\nSkipping unrolled-LP at N={N_STAGES} (known unbuildable: "
          f"tiled smem > device limit).")

    out = {}
    print(f"\n{'B':>6}  {'stage+LP':>11}  {'unrolled+LP':>13}  "
          f"{'chunk@bs=1':>12}  {'orig seq':>11}")
    print('-' * 72)
    for B in BATCHES:
        row = {}
        # 1. stage+LP (default block_size)
        inputs_stage = _make_inputs(f_stage, B, N_STAGES, with_stage_axis=True)
        lp_full, _ = _build_stage_lp(f_stage, B, N_STAGES, per_stage_plan,
                                      block_size_override=None)
        row["stage_LP"] = benchmark_kernel_only(
            lambda: lp_full.evaluate(inputs_stage),
            lambda: lp_full.eval_time, 5, 30)
        del lp_full

        # 2. unrolled+LP — predicted-unbuildable at N=30, see comment above.
        row["unrolled_LP"] = None

        # 3. chunk@bs=1
        lp_seq, _ = _build_stage_lp(f_stage, B, N_STAGES, per_stage_plan,
                                     block_size_override=1)
        row["chunk_bs1"] = benchmark_kernel_only(
            lambda: lp_seq.evaluate(inputs_stage),
            lambda: lp_seq.eval_time, 5, 30)
        del lp_seq

        # 4. original cusadi @ B*N
        if seq_avail:
            inputs_flat = _make_inputs(f_stage, B, N_STAGES, with_stage_axis=False)
            seq_fn = SequentialFn(f_stage, B * N_STAGES, SEQ_BUILD_DIR)
            seq_fn.set_inputs(inputs_flat)
            row["original"] = benchmark_kernel_only(
                seq_fn.evaluate, lambda: seq_fn.eval_time, 5, 30)
            del seq_fn
        else:
            row["original"] = None

        u_str = f"{row['unrolled_LP']:13.1f}" if row["unrolled_LP"] is not None else "         skip"
        o_str = f"{row['original']:11.1f}"   if row["original"]    is not None else "       skip"
        print(f"  {B:>6}  {row['stage_LP']:>11.1f}  {u_str}  "
              f"{row['chunk_bs1']:>12.1f}  {o_str}")
        out[str(B)] = row

        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass

    Bs = sorted(int(b) for b in out.keys())
    data = {
        "fn":            FN_NAME,
        "n_stages":      N_STAGES,
        "median_width":  per_stage_plan.median_width,
        "max_width":     per_stage_plan.max_width,
        "n_levels":      per_stage_plan.n_compute_levels,
        "peak_live":     per_stage_plan.peak_live_slots,
        "block_size":    block_size,
        "chunk_size":    decision.chunk_size,
        "B":             Bs,
        "stage_LP":      [out[str(b)]["stage_LP"]      for b in Bs],
        "unrolled_LP":   [out[str(b)]["unrolled_LP"]   for b in Bs],
        "chunk_bs1":     [out[str(b)]["chunk_bs1"]     for b in Bs],
        "original":      [out[str(b)]["original"]      for b in Bs],
        "unrolled_buildable": unrolled_buildable,
        "unrolled_info": unr_info,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
