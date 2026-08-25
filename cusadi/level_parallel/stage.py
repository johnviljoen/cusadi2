"""Backend-agnostic stage planning, chunk-plan construction, and autotuning.

Extracted from the torch stage evaluator (cusadi/pytorch/lp_stage.py) so the
level-parallel stage logic — spec types, chunk-DAG construction, the
smem-budget autotuner, and the CPU reference simulator — carries no torch
dependency and can be shared by any backend (torch, jax, ...). The torch
evaluator classes live in cusadi/pytorch/lp_stage.py and import these symbols.
"""

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence

import casadi
import numpy as np

from cusadi.level_parallel.dag import (LevelPlan, Level, SSANode, SSAConst,
                                       SSAInput, SSAOutput, extract_level_plan)


class StageMode(IntEnum):
    STAGE = 0
    SHARED = 1
    COUPLED = 2


@dataclass
class StageSpec:
    """Per-arg declaration for stage-axis indexing.

    mode    : STAGE / SHARED / COUPLED
    offset  : added to stage_index when mode == COUPLED
    span    : extra leading-axis elements past n_stages (e.g. +1 for x_{k+1});
              ignored when mode != COUPLED.
    """
    mode: StageMode
    offset: int = 0
    span: int = 0

    def stage_count(self, n_stages: int) -> int:
        if self.mode == StageMode.SHARED:
            return 1
        if self.mode == StageMode.COUPLED:
            return n_stages + self.span
        return n_stages


def _validate_specs(fn_casadi: casadi.Function, n_stages: int,
                    input_specs: Sequence[StageSpec],
                    output_specs: Sequence[StageSpec]) -> None:
    n_in, n_out = fn_casadi.n_in(), fn_casadi.n_out()
    if len(input_specs) != n_in:
        raise ValueError(
            f"input_specs has {len(input_specs)} entries; "
            f"f_stage has {n_in} inputs.")
    if len(output_specs) != n_out:
        raise ValueError(
            f"output_specs has {len(output_specs)} entries; "
            f"f_stage has {n_out} outputs.")
    if n_stages <= 0:
        raise ValueError(f"n_stages must be positive, got {n_stages}")
    for i, s in enumerate(input_specs):
        if s.mode == StageMode.COUPLED:
            min_idx = s.offset
            max_idx = (n_stages - 1) + s.offset
            if min_idx < 0 or max_idx >= s.stage_count(n_stages):
                raise ValueError(
                    f"input {i}: COUPLED offset={s.offset} span={s.span} "
                    f"reads outside [0, {s.stage_count(n_stages)}) "
                    f"for n_stages={n_stages}")
        elif s.mode == StageMode.STAGE and s.offset != 0:
            raise ValueError(
                f"input {i}: offset is only valid for COUPLED mode")
    for i, s in enumerate(output_specs):
        if s.mode != StageMode.STAGE:
            raise ValueError(
                f"output {i}: only STAGE mode is supported on outputs "
                f"(SHARED would race across stages, COUPLED has no meaning).")


def _build_chunk_plan_analytic(
    per_stage_plan: LevelPlan,
    chunk_size: int,
    input_specs: Sequence[StageSpec],
) -> LevelPlan:
    """Build the chunked LevelPlan analytically by replicating per_stage_plan
    chunk_size times — no CasADi stamping required.

    Construction:
      * SSA slot space is replicated c times. Replica k uses slots
        [k * S, (k+1) * S) where S = per_stage_plan.n_ssa_slots.
      * Constants, input mappings, output mappings are duplicated per replica
        with shifted slots.
      * Compute nodes from each per-stage level d are merged into the same
        chunk-level level d (so all c replicas at depth d execute as one
        wide level — exactly what level-parallel execution wants).
      * Element-index encoding for input/output mappings depends on stage
        mode: STAGE / COUPLED → element_idx = inner_stage * nnz + e;
        SHARED → element_idx = e (broadcast across stages).

    The resulting plan is byte-identical in *runtime semantics* to what
    `extract_level_plan(_build_chunk_function(f_stage, c, input_specs))`
    produces, modulo (a) constant deduplication that CasADi might apply
    when stamped values are detected as equal — irrelevant for typical
    MPC stage programs — and (b) ordering of operands within levels.

    Construction time: O(c × per_stage size). Microseconds for the
    workloads we care about, vs. tens to hundreds of seconds for the
    CasADi-stamping path at large c.
    """
    if chunk_size == 1:
        return per_stage_plan

    S = per_stage_plan.n_ssa_slots
    nnz_per_stage_in = list(per_stage_plan.nnz_in)
    nnz_per_stage_out = list(per_stage_plan.nnz_out)

    constants: List[SSAConst] = []
    for k in range(chunk_size):
        for c in per_stage_plan.constants:
            constants.append(SSAConst(ssa_slot=c.ssa_slot + k * S, value=c.value))

    input_mappings: List[SSAInput] = []
    for k in range(chunk_size):
        for m in per_stage_plan.input_mappings:
            spec = input_specs[m.input_arg]
            if spec.mode == StageMode.SHARED:
                new_elem = m.element_idx
            else:
                new_elem = k * nnz_per_stage_in[m.input_arg] + m.element_idx
            input_mappings.append(SSAInput(
                ssa_slot=m.ssa_slot + k * S,
                input_arg=m.input_arg,
                element_idx=new_elem,
            ))

    output_mappings: List[SSAOutput] = []
    for k in range(chunk_size):
        for m in per_stage_plan.output_mappings:
            output_mappings.append(SSAOutput(
                ssa_slot=m.ssa_slot + k * S,
                output_arg=m.output_arg,
                element_idx=k * nnz_per_stage_out[m.output_arg] + m.element_idx,
            ))

    compute_levels: List[Level] = []
    for level in per_stage_plan.compute_levels:
        chunk_nodes: List[SSANode] = []
        for k in range(chunk_size):
            for node in level.nodes:
                chunk_nodes.append(SSANode(
                    instr_idx=node.instr_idx,
                    op_id=node.op_id,
                    ssa_out=node.ssa_out + k * S,
                    ssa_inputs=tuple(s + k * S for s in node.ssa_inputs),
                    level=level.depth,
                    const_value=node.const_value,
                ))
        compute_levels.append(Level(depth=level.depth, nodes=chunk_nodes))

    # Per-arg total nnz in the chunk function's signature.
    nnz_chunk_in = []
    for i in range(per_stage_plan.n_inputs):
        spec = input_specs[i]
        nnz_chunk_in.append(
            nnz_per_stage_in[i] if spec.mode == StageMode.SHARED
            else chunk_size * nnz_per_stage_in[i]
        )
    nnz_chunk_out = [chunk_size * n for n in nnz_per_stage_out]

    return LevelPlan(
        function_name=f"{per_stage_plan.function_name}_chunk{chunk_size}",
        n_ssa_slots=chunk_size * S,
        peak_live_slots=chunk_size * per_stage_plan.peak_live_slots,
        n_inputs=per_stage_plan.n_inputs,
        n_outputs=per_stage_plan.n_outputs,
        nnz_in=nnz_chunk_in,
        nnz_out=nnz_chunk_out,
        constants=constants,
        input_mappings=input_mappings,
        output_mappings=output_mappings,
        compute_levels=compute_levels,
    )


def _build_chunk_function(
    f_stage: casadi.Function,
    chunk_size: int,
    input_specs: Sequence[StageSpec],
) -> casadi.Function:
    """Stamp f_stage chunk_size times into a single CasADi function.

    For every input arg of f_stage, the chunk arg has leading dimension:
        STAGE   -> chunk_size
        SHARED  -> 1
        COUPLED -> chunk_size           (yes, just chunk_size — see below)

    The chunked function's k-th internal stage always reads chunk arg
    index k (or 0 for SHARED). The COUPLED-arg `offset`/`span` are NOT
    baked into f_chunk: they describe how the kernel slices the user's
    stage-axis buffer per chunk (`chunk_offset[arg]` shifts each chunk's
    starting buffer position; `total_stages[arg]` carries the buffer's
    full extent of n_stages+span). f_chunk just sees a clean
    (chunk_size,)-extent input per arg, and the kernel maps that onto
    the correct slice of the user buffer.

    All outputs are STAGE; chunk arg has leading dimension chunk_size.
    """
    if chunk_size == 1:
        return f_stage

    n_in = f_stage.n_in()
    n_out = f_stage.n_out()
    assert len(input_specs) == n_in

    chunk_inputs = []
    chunk_in_names = []
    for i, spec in enumerate(input_specs):
        nnz = f_stage.nnz_in(i)
        extent = 1 if spec.mode == StageMode.SHARED else chunk_size
        sym = casadi.SX.sym(f"in{i}", nnz, extent)
        chunk_inputs.append(sym)
        chunk_in_names.append(f"in{i}")

    stage_outs = [[] for _ in range(n_out)]
    for k in range(chunk_size):
        per_stage_args = []
        for i, spec in enumerate(input_specs):
            sym = chunk_inputs[i]
            idx = 0 if spec.mode == StageMode.SHARED else k
            per_stage_args.append(sym[:, idx])
        outs_k = f_stage.call(per_stage_args)
        for j in range(n_out):
            stage_outs[j].append(outs_k[j])

    chunk_outputs = [casadi.horzcat(*outs) for outs in stage_outs]
    chunk_out_names = [f"out{j}" for j in range(n_out)]

    flat_inputs = [casadi.reshape(sym, -1, 1) for sym in chunk_inputs]
    flat_outputs = [casadi.reshape(o, -1, 1) for o in chunk_outputs]

    return casadi.Function(
        f"{f_stage.name()}_chunk{chunk_size}",
        flat_inputs, flat_outputs,
        chunk_in_names, chunk_out_names,
    )


def _get_device_smem_limit_bytes() -> int:
    """Query the current CUDA device's max dynamic shared memory per block.

    Mirrors cusadi.pytorch.lp_unified._get_device_smem_limit but kept local
    so this module doesn't pull in the unified Wrapper machinery.
    """
    import ctypes as _ct
    libcudart = _ct.CDLL("libcudart.so")
    val = _ct.c_int(0)
    # cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
    libcudart.cudaDeviceGetAttribute(_ct.byref(val), 97, 0)
    return val.value


def auto_tune_chunk_size(
    f_stage: casadi.Function,
    n_stages: int,
    input_specs: Sequence[StageSpec],
    smem_budget_bytes: Optional[int] = None,
    smem_margin_bytes: int = 4096,
    verify: bool = True,
    per_stage_plan: Optional[LevelPlan] = None,
) -> int:
    """Pick the largest chunk_size that:
       1. divides n_stages evenly,
       2. produces a chunked plan whose peak shared memory fits in the budget.

    Strategy: use c × per_stage_peak as a conservative upper bound to choose
    a candidate cheaply, then (if verify=True) extract the actual chunked
    plan and halve the candidate until it fits. The conservative bound
    almost always succeeds without needing a halving pass.

    `per_stage_plan` is an optional cached plan for f_stage — pass it to avoid
    a second extract_level_plan call from a caller that already has one.
    """
    if smem_budget_bytes is None:
        smem_budget_bytes = _get_device_smem_limit_bytes() - smem_margin_bytes
    max_slots = smem_budget_bytes // 8

    if per_stage_plan is None:
        per_stage_plan = extract_level_plan(f_stage)
    per_stage_peak = max(per_stage_plan.peak_live_slots, 1)

    if per_stage_peak > max_slots:
        # Even one stage doesn't fit in smem; tiling territory, not chunkable.
        return 1

    # Largest c with c * per_stage_peak ≤ max_slots, capped at n_stages.
    naive_max_c = max(1, max_slots // per_stage_peak)
    naive_max_c = min(naive_max_c, n_stages)

    # Largest divisor of n_stages ≤ naive_max_c.
    divisors = [d for d in range(1, n_stages + 1) if n_stages % d == 0]
    candidates = [d for d in divisors if d <= naive_max_c]
    if not candidates:
        return 1

    if not verify:
        return max(candidates)

    # Walk candidates from largest to smallest; verify by building the
    # chunked plan analytically (microseconds, no CasADi stamping).
    # Stops at the first c that genuinely fits.
    for c in sorted(candidates, reverse=True):
        if c == 1:
            return 1
        chunk_plan = _build_chunk_plan_analytic(per_stage_plan, c, input_specs)
        if chunk_plan.peak_live_slots <= max_slots:
            return c
    return 1


# ---------------------------------------------------------------------------
# Smart auto-dispatch
# ---------------------------------------------------------------------------

@dataclass
class DispatchDecision:
    """Why a particular (chunk_size, mode) was picked. Surfaced for transparency:
    so you can put the decision in your defense slides without re-deriving it.

    `mode` is one of:
      "chunk"       — chunked stage kernel, chunk_size as picked
      "tiled-stage" — tiled stage kernel; chunk_size==1 implicitly because
                      one stage already overflows shared memory
    """
    chunk_size: int
    reason: str
    per_stage_peak_slots: int
    per_stage_smem_kb: float
    device_smem_kb: float
    has_coupled: bool
    n_stages: int
    auto_max_c: int
    mode: str = "chunk"


# Heuristic threshold: per-stage programs below this size are small enough
# that block-launch overhead dominates, and fusion (large c) wins even
# without operand reuse from COUPLED inputs. Empirically calibrated on
# unicycle (~0.3 KB, fuses 60×) and A1 jac (34 KB, c=1 optimal).
_SMALL_STAGE_SMEM_KB = 4.0


def smart_pick_chunk_size(
    f_stage: casadi.Function,
    n_stages: int,
    input_specs: Sequence[StageSpec],
    smem_budget_bytes: Optional[int] = None,
    smem_margin_bytes: int = 4096,
    small_stage_threshold_kb: float = _SMALL_STAGE_SMEM_KB,
    per_stage_plan: Optional[LevelPlan] = None,
) -> DispatchDecision:
    """Pick chunk_size based on (problem, hardware) characteristics.

    Decision rules, in order:
      1. If `f_stage` itself exceeds the device shared-memory budget,
         raise — this is the regime that needs tiled-LP inside each
         chunk, which isn't wired through this dispatch path yet.
      2. If any input is COUPLED, fusion across stages saves redundant
         global loads (CasADi CSE collapses x_kp1 of stage k with x_k
         of stage k+1). Maximise c.
      3. If `f_stage` is small (per-stage smem < threshold), fusion
         amortises per-block fixed costs (constant load, init, smem
         setup). Maximise c.
      4. Otherwise — heavy independent stages — pick c=1. Each block
         has enough work to saturate an SM on its own; fusing two
         stages just halves block count without improving I/O.

    Returns a DispatchDecision (carries the picked c plus the
    measurements that motivated it) so callers can log / surface why.
    """
    if smem_budget_bytes is None:
        smem_budget_bytes = _get_device_smem_limit_bytes() - smem_margin_bytes
    device_smem_kb = (smem_budget_bytes + smem_margin_bytes) / 1024.0

    if per_stage_plan is None:
        per_stage_plan = extract_level_plan(f_stage)
    per_stage_peak = max(per_stage_plan.peak_live_slots, 1)
    per_stage_smem_kb = per_stage_peak * 8 / 1024.0

    has_coupled = any(s.mode == StageMode.COUPLED for s in input_specs)

    if per_stage_peak * 8 > smem_budget_bytes:
        # Per-stage exceeds device shared memory. Dispatch to the tiled-stage
        # kernel, which spills/reloads within each stage. chunk_size is
        # implicitly 1 (fusing more stages into one block would only worsen
        # the smem pressure).
        return DispatchDecision(
            chunk_size=1,
            reason=(f"f_stage peak smem ({per_stage_smem_kb:.1f} KB) exceeds "
                    f"device budget ({smem_budget_bytes/1024:.0f} KB) → "
                    f"tiled-stage kernel: each (instance, stage) block tiles "
                    f"the per-stage DAG with spill/reload."),
            per_stage_peak_slots=per_stage_peak,
            per_stage_smem_kb=per_stage_smem_kb,
            device_smem_kb=device_smem_kb,
            has_coupled=has_coupled,
            n_stages=n_stages,
            auto_max_c=1,
            mode="tiled-stage",
        )

    auto_max_c = auto_tune_chunk_size(
        f_stage, n_stages, input_specs,
        smem_budget_bytes=smem_budget_bytes, verify=False,
        per_stage_plan=per_stage_plan,
    )

    if has_coupled:
        c = auto_max_c
        reason = (f"COUPLED inputs detected → maximize chunk_size for operand "
                  f"reuse. auto_max={auto_max_c} (largest divisor of N={n_stages} "
                  f"whose chunked plan fits in {device_smem_kb:.0f} KB).")
    elif per_stage_smem_kb < small_stage_threshold_kb:
        c = auto_max_c
        reason = (f"Small per-stage program ({per_stage_smem_kb:.2f} KB < "
                  f"{small_stage_threshold_kb:.1f} KB threshold) → fusion "
                  f"amortises per-block overhead. auto_max={auto_max_c}.")
    else:
        c = 1
        reason = (f"Heavy independent stages ({per_stage_smem_kb:.1f} KB, all "
                  f"STAGE/SHARED inputs) → c=1 maximises SM occupancy. "
                  f"Fusion would halve block count without saving any global "
                  f"loads.")

    return DispatchDecision(
        chunk_size=c,
        reason=reason,
        per_stage_peak_slots=per_stage_peak,
        per_stage_smem_kb=per_stage_smem_kb,
        device_smem_kb=device_smem_kb,
        has_coupled=has_coupled,
        n_stages=n_stages,
        auto_max_c=auto_max_c,
        mode="chunk",
    )


def evaluate_stage_reference(
    f_stage: casadi.Function,
    n_stages: int,
    input_specs: Sequence[StageSpec],
    inputs_np: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """CPU reference: evaluate f_stage at every (instance, stage) pair using
    CasADi's own interpreter, applying the same stage-axis indexing the GPU
    kernel uses. Returns list of arrays shaped (num_instances, n_stages, nnz_out).

    Outputs are always STAGE mode (enforced by _validate_specs).

    Use this to validate numerical equivalence between the stage kernel and an
    unrolled (Function.fold-style) baseline, or against CasADi directly.
    """
    n_in, n_out = f_stage.n_in(), f_stage.n_out()
    assert len(inputs_np) == n_in
    num_instances = inputs_np[0].shape[0]

    out_arrs = [
        np.zeros((num_instances, n_stages, f_stage.nnz_out(i)), dtype=np.float64)
        for i in range(n_out)
    ]
    for inst in range(num_instances):
        for stage in range(n_stages):
            stage_inputs = []
            for i, spec in enumerate(input_specs):
                if spec.mode == StageMode.SHARED:
                    if inputs_np[i].ndim == 2:
                        stage_inputs.append(inputs_np[i][inst])
                    else:
                        stage_inputs.append(inputs_np[i][inst, 0])
                elif spec.mode == StageMode.COUPLED:
                    stage_inputs.append(inputs_np[i][inst, stage + spec.offset])
                else:  # STAGE
                    stage_inputs.append(inputs_np[i][inst, stage])
            outs = f_stage.call(stage_inputs)
            for i in range(n_out):
                out_arrs[i][inst, stage] = np.asarray(outs[i].nonzeros(),
                                                      dtype=np.float64)
    return out_arrs
