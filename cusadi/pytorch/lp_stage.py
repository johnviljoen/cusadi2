"""
Stage-compiled level-parallel CUDA wrapper.

Compiles the per-stage program once (via the existing data-driven LP kernel
template, with stage-aware indexing) and instantiates it across both batch
instances and stages — adding a multiplicative parallelism axis on top of
intra-stage level parallelism.

Per-arg stage modes:
    STAGE   — buffer shape (num_instances, n_stages, nnz)
    SHARED  — buffer shape (num_instances, nnz)              (broadcast)
    COUPLED — buffer shape (num_instances, n_stages+span, nnz),
              read at stage_index + offset (e.g. +1 for x_{k+1}).

Aliasing: it is fine for two inputs to point at the same backing tensor with
different (mode, offset) — e.g. x_k and x_{k+1} both reading the same X buffer
of shape (B, n_stages+1, n_x) at offsets 0 and +1. Outputs must not alias an
input that is read at a later stage in the same launch (RAW hazards across
the stage axis).
"""

import ctypes
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence

import casadi
import numpy as np
import torch

from cusadi import CUSADI_LP_BUILD_DIR
from cusadi.level_parallel.dag import (LevelPlan, TiledLevelPlan, Level, SSANode,
                                     SSAConst, SSAInput, SSAOutput,
                                     extract_level_plan,
                                     extract_tiled_level_plan)
from cusadi.level_parallel.codegen import pick_block_size


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


def _to_gpu_int(arr) -> torch.Tensor:
    return torch.tensor(arr, device="cuda", dtype=torch.int32).contiguous()


def _to_gpu_double(arr) -> torch.Tensor:
    return torch.tensor(arr, device="cuda", dtype=torch.double).contiguous()


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


class CusadiStageLevelParallelFunction:
    """Stage-compiled level-parallel CUDA wrapper.

    Args:
        f_stage:       CasADi Function describing one stage program.
        num_instances: Batch size (blockIdx.x extent).
        n_stages:      Number of stages (blockIdx.y extent).
        input_specs:   Per-input stage spec, length f_stage.n_in().
        output_specs:  Per-output stage spec, length f_stage.n_out().
        lib_name:      Name of the prebuilt .so (defaults to the canonical
                       stage kernel; same .so is reused for any f_stage).
    """

    def __init__(
        self,
        f_stage: casadi.Function,
        num_instances: int,
        n_stages: int,
        input_specs: Sequence[StageSpec],
        output_specs: Sequence[StageSpec],
        lib_name: str = "level_parallel_stage_kernel",
    ):
        assert torch.cuda.is_available()
        _validate_specs(f_stage, n_stages, input_specs, output_specs)

        self.fn_casadi = f_stage
        self.fn_name = f_stage.name()
        self.num_instances = num_instances
        self.n_stages = n_stages
        self.input_specs = list(input_specs)
        self.output_specs = list(output_specs)
        self._device = "cuda"

        lib_path = os.path.join(CUSADI_LP_BUILD_DIR, f"lib{lib_name}.so")
        self._fn_library = ctypes.CDLL(lib_path)
        self._fn_library.evaluate_lp_stage.restype = ctypes.c_float
        self._fn_library.configure_stage_shared_memory.restype = ctypes.c_int

        plan = extract_level_plan(f_stage)
        self._plan = plan
        self._block_size = pick_block_size(plan)
        self._shared_bytes = plan.peak_live_slots * 8

        self._fn_library.configure_stage_shared_memory(self._shared_bytes)

        self._upload_plan(plan)
        self._upload_stage_metadata()
        self._setup_io(f_stage, num_instances, n_stages)

    # -- Plan flattening (mirrors CusadiLevelParallelFunction._upload_plan) --
    def _upload_plan(self, plan: LevelPlan):
        if plan.constants:
            self._const_slots = _to_gpu_int([c.ssa_slot for c in plan.constants])
            self._const_vals = _to_gpu_double([c.value for c in plan.constants])
        else:
            self._const_slots = _to_gpu_int([0])
            self._const_vals = _to_gpu_double([0.0])
        self._n_consts = len(plan.constants)

        if plan.input_mappings:
            self._input_slots = _to_gpu_int([m.ssa_slot for m in plan.input_mappings])
            self._input_args = _to_gpu_int([m.input_arg for m in plan.input_mappings])
            self._input_elems = _to_gpu_int([m.element_idx for m in plan.input_mappings])
        else:
            self._input_slots = _to_gpu_int([0])
            self._input_args = _to_gpu_int([0])
            self._input_elems = _to_gpu_int([0])
        self._n_input_maps = len(plan.input_mappings)
        self._nnz_in_gpu = _to_gpu_int(plan.nnz_in)

        all_ops, all_outs, all_in_a, all_in_b = [], [], [], []
        level_offsets, level_widths = [], []
        offset = 0
        for level in plan.compute_levels:
            level_offsets.append(offset)
            level_widths.append(level.width)
            for node in level.nodes:
                all_ops.append(node.op_id)
                all_outs.append(node.ssa_out)
                all_in_a.append(node.ssa_inputs[0] if len(node.ssa_inputs) >= 1 else 0)
                all_in_b.append(node.ssa_inputs[1] if len(node.ssa_inputs) >= 2 else 0)
            offset += level.width

        if all_ops:
            self._node_ops = _to_gpu_int(all_ops)
            self._node_outs = _to_gpu_int(all_outs)
            self._node_in_a = _to_gpu_int(all_in_a)
            self._node_in_b = _to_gpu_int(all_in_b)
            self._level_offsets = _to_gpu_int(level_offsets)
            self._level_widths = _to_gpu_int(level_widths)
        else:
            self._node_ops = _to_gpu_int([0])
            self._node_outs = _to_gpu_int([0])
            self._node_in_a = _to_gpu_int([0])
            self._node_in_b = _to_gpu_int([0])
            self._level_offsets = _to_gpu_int([0])
            self._level_widths = _to_gpu_int([0])
        self._n_levels = len(level_offsets)

        if plan.output_mappings:
            self._out_slots = _to_gpu_int([m.ssa_slot for m in plan.output_mappings])
            self._out_args = _to_gpu_int([m.output_arg for m in plan.output_mappings])
            self._out_elems = _to_gpu_int([m.element_idx for m in plan.output_mappings])
        else:
            self._out_slots = _to_gpu_int([0])
            self._out_args = _to_gpu_int([0])
            self._out_elems = _to_gpu_int([0])
        self._n_output_maps = len(plan.output_mappings)
        self._nnz_out_gpu = _to_gpu_int(plan.nnz_out)

    def _upload_stage_metadata(self):
        n = self.n_stages
        in_modes  = [int(s.mode)               for s in self.input_specs]
        in_offs   = [s.offset                  for s in self.input_specs]
        in_counts = [s.stage_count(n)          for s in self.input_specs]
        out_modes  = [int(s.mode)              for s in self.output_specs]
        out_counts = [s.stage_count(n)         for s in self.output_specs]
        self._in_stage_mode   = _to_gpu_int(in_modes  or [0])
        self._in_stage_offset = _to_gpu_int(in_offs   or [0])
        self._in_stage_count  = _to_gpu_int(in_counts or [1])
        self._out_stage_mode  = _to_gpu_int(out_modes  or [0])
        self._out_stage_count = _to_gpu_int(out_counts or [1])

    def _setup_io(self, f_stage: casadi.Function, num_instances: int, n_stages: int):
        nin, nout = f_stage.n_in(), f_stage.n_out()

        # Output tensors: cusadi-managed, sized (B, stage_count, nnz).
        self._output_tensors = []
        for i in range(nout):
            sc = self.output_specs[i].stage_count(n_stages)
            self._output_tensors.append(torch.zeros(
                num_instances, sc, f_stage.nnz_out(i),
                device=self._device, dtype=torch.double).contiguous())

        # Input tensors: caller-supplied, but we hold zero-initialized stand-ins
        # so .data_ptr() works before the first evaluate() call (matches the
        # behaviour of CusadiLevelParallelFunction).
        self._input_tensors = []
        for i in range(nin):
            sc = self.input_specs[i].stage_count(n_stages)
            self._input_tensors.append(torch.zeros(
                num_instances, sc, f_stage.nnz_in(i),
                device=self._device, dtype=torch.double).contiguous())

        self._input_ptrs = torch.zeros(nin, device=self._device,
                                       dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(nout, device=self._device,
                                        dtype=torch.int64).contiguous()
        for i in range(nin):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(nout):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self.outputs_sparse = self._output_tensors

        self._coo_rows, self._coo_cols = [], []
        for i in range(nout):
            rows, cols = f_stage.sparsity_out(i).get_triplet()
            self._coo_rows.append(torch.tensor(rows, device=self._device, dtype=torch.int64))
            self._coo_cols.append(torch.tensor(cols, device=self._device, dtype=torch.int64))

    def _ptr(self, t: torch.Tensor):
        return ctypes.c_void_p(t.data_ptr())

    def _validate_input_shape(self, idx: int, t: torch.Tensor):
        spec = self.input_specs[idx]
        sc = spec.stage_count(self.n_stages)
        nnz = self.fn_casadi.nnz_in(idx)
        # Accept (B, sc, nnz) or — for SHARED — (B, nnz). Both flatten to the
        # same memory under the kernel's index arithmetic.
        if spec.mode == StageMode.SHARED and t.dim() == 2:
            expected = (self.num_instances, nnz)
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"input {idx} (SHARED) shape {tuple(t.shape)} != {expected}")
            return
        expected = (self.num_instances, sc, nnz)
        if tuple(t.shape) != expected:
            raise ValueError(
                f"input {idx} shape {tuple(t.shape)} != {expected} "
                f"(mode={spec.mode.name}, n_stages={self.n_stages}, span={spec.span})")

    def evaluate(self, inputs: List[torch.Tensor]):
        for i in range(self.fn_casadi.n_out()):
            self._output_tensors[i].zero_()

        for i in range(self.fn_casadi.n_in()):
            self._validate_input_shape(i, inputs[i])
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = inputs[i].data_ptr()

        fn_input  = ctypes.cast(self._input_ptrs.data_ptr(),  ctypes.POINTER(ctypes.c_int))
        fn_output = ctypes.cast(self._output_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))

        self.eval_time = self._fn_library.evaluate_lp_stage(
            fn_input,
            fn_output,
            self.num_instances,
            self.n_stages,
            self._block_size,
            self._shared_bytes,
            # Init
            self._n_consts,
            self._ptr(self._const_slots),
            self._ptr(self._const_vals),
            self._n_input_maps,
            self._ptr(self._input_slots),
            self._ptr(self._input_args),
            self._ptr(self._input_elems),
            self._ptr(self._nnz_in_gpu),
            self._ptr(self._in_stage_mode),
            self._ptr(self._in_stage_offset),
            self._ptr(self._in_stage_count),
            # Compute
            self._n_levels,
            self._ptr(self._level_offsets),
            self._ptr(self._level_widths),
            self._ptr(self._node_ops),
            self._ptr(self._node_outs),
            self._ptr(self._node_in_a),
            self._ptr(self._node_in_b),
            # Outputs
            self._n_output_maps,
            self._ptr(self._out_slots),
            self._ptr(self._out_args),
            self._ptr(self._out_elems),
            self._ptr(self._nnz_out_gpu),
            self._ptr(self._out_stage_mode),
            self._ptr(self._out_stage_count),
        )

    def get_coo(self, out_idx: int = 0):
        return (self._coo_rows[out_idx],
                self._coo_cols[out_idx],
                self.outputs_sparse[out_idx])


# ---------------------------------------------------------------------------
# Reference simulator for numerical verification (no GPU needed)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hybrid (chunked) stage compilation
# ---------------------------------------------------------------------------

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


class CusadiHybridStageLevelParallelFunction:
    """Hybrid stage-compiled level-parallel kernel.

    Each block processes one *chunk* of `chunk_size` consecutive stages of
    f_stage. The chunked-DAG fuses across the c stages (sharing intermediate
    values, amortising per-block fixed costs), and (B × n_chunks) blocks
    parallelise across the remaining stage axis. chunk_size=1 reduces to the
    pure stage kernel; chunk_size=n_stages is the unrolled-LP equivalent.

    n_stages must be divisible by chunk_size (use auto_tune_chunk_size to
    pick a divisor automatically).
    """

    def __init__(
        self,
        f_stage: casadi.Function,
        num_instances: int,
        n_stages: int,
        input_specs: Sequence[StageSpec],
        output_specs: Sequence[StageSpec],
        chunk_size: int = 1,
        lib_name: str = "level_parallel_chunk_kernel",
        analytic_chunk: bool = True,
        per_stage_plan: Optional[LevelPlan] = None,
        block_size_override: Optional[int] = None,
    ):
        """`block_size_override` forces a specific CUDA block size, bypassing
        `pick_block_size`. Setting it to 1 effectively disables level-parallel
        execution within each block: the per-level work loop walks all width
        nodes sequentially on a single thread, while keeping the same
        cross-block (B × n_chunks) parallelism. Useful for ablation studies
        that isolate the contribution of within-block parallelism from
        cross-block parallelism."""
        assert torch.cuda.is_available()
        _validate_specs(f_stage, n_stages, input_specs, output_specs)
        if n_stages % chunk_size != 0:
            raise ValueError(
                f"n_stages={n_stages} must be divisible by chunk_size={chunk_size}")

        self.f_stage = f_stage
        self.fn_casadi = f_stage  # for API parity
        self.fn_name = f_stage.name()
        self.num_instances = num_instances
        self.n_stages = n_stages
        self.chunk_size = chunk_size
        self.n_chunks = n_stages // chunk_size
        self.input_specs = list(input_specs)
        self.output_specs = list(output_specs)
        self._device = "cuda"

        lib_path = os.path.join(CUSADI_LP_BUILD_DIR, f"lib{lib_name}.so")
        self._fn_library = ctypes.CDLL(lib_path)
        self._fn_library.evaluate_lp_chunk.restype = ctypes.c_float
        self._fn_library.configure_chunk_shared_memory.restype = ctypes.c_int

        # Build the chunked plan. Default path: replicate the per-stage plan
        # analytically — milliseconds even at chunk_size=1000+. The CasADi
        # stamping path is kept (analytic_chunk=False) for cross-validation.
        # `per_stage_plan` may be passed by the smart-dispatch helper so we
        # don't extract f_stage's plan twice.
        if analytic_chunk:
            if per_stage_plan is None:
                per_stage_plan = extract_level_plan(f_stage)
            plan = _build_chunk_plan_analytic(
                per_stage_plan, chunk_size, self.input_specs)
            self.f_chunk = None
        else:
            self.f_chunk = _build_chunk_function(
                f_stage, chunk_size, self.input_specs)
            plan = extract_level_plan(self.f_chunk)
        self._plan = plan
        self._block_size = (block_size_override
                            if block_size_override is not None
                            else pick_block_size(plan))
        self._shared_bytes = plan.peak_live_slots * 8
        self._fn_library.configure_chunk_shared_memory(self._shared_bytes)

        self._upload_plan(plan)
        self._upload_chunk_metadata()
        self._setup_io(f_stage, num_instances, n_stages)

    # ----- plan flattening (mirrors stage wrapper) -----
    def _upload_plan(self, plan: LevelPlan):
        if plan.constants:
            self._const_slots = _to_gpu_int([c.ssa_slot for c in plan.constants])
            self._const_vals = _to_gpu_double([c.value for c in plan.constants])
        else:
            self._const_slots = _to_gpu_int([0])
            self._const_vals = _to_gpu_double([0.0])
        self._n_consts = len(plan.constants)

        if plan.input_mappings:
            self._input_slots = _to_gpu_int([m.ssa_slot for m in plan.input_mappings])
            self._input_args = _to_gpu_int([m.input_arg for m in plan.input_mappings])
            self._input_elems = _to_gpu_int([m.element_idx for m in plan.input_mappings])
        else:
            self._input_slots = _to_gpu_int([0])
            self._input_args = _to_gpu_int([0])
            self._input_elems = _to_gpu_int([0])
        self._n_input_maps = len(plan.input_mappings)

        all_ops, all_outs, all_in_a, all_in_b = [], [], [], []
        level_offsets, level_widths = [], []
        offset = 0
        for level in plan.compute_levels:
            level_offsets.append(offset)
            level_widths.append(level.width)
            for node in level.nodes:
                all_ops.append(node.op_id)
                all_outs.append(node.ssa_out)
                all_in_a.append(node.ssa_inputs[0] if len(node.ssa_inputs) >= 1 else 0)
                all_in_b.append(node.ssa_inputs[1] if len(node.ssa_inputs) >= 2 else 0)
            offset += level.width

        if all_ops:
            self._node_ops = _to_gpu_int(all_ops)
            self._node_outs = _to_gpu_int(all_outs)
            self._node_in_a = _to_gpu_int(all_in_a)
            self._node_in_b = _to_gpu_int(all_in_b)
            self._level_offsets = _to_gpu_int(level_offsets)
            self._level_widths = _to_gpu_int(level_widths)
        else:
            self._node_ops = _to_gpu_int([0])
            self._node_outs = _to_gpu_int([0])
            self._node_in_a = _to_gpu_int([0])
            self._node_in_b = _to_gpu_int([0])
            self._level_offsets = _to_gpu_int([0])
            self._level_widths = _to_gpu_int([0])
        self._n_levels = len(level_offsets)

        if plan.output_mappings:
            self._out_slots = _to_gpu_int([m.ssa_slot for m in plan.output_mappings])
            self._out_args = _to_gpu_int([m.output_arg for m in plan.output_mappings])
            self._out_elems = _to_gpu_int([m.element_idx for m in plan.output_mappings])
        else:
            self._out_slots = _to_gpu_int([0])
            self._out_args = _to_gpu_int([0])
            self._out_elems = _to_gpu_int([0])
        self._n_output_maps = len(plan.output_mappings)

    def _upload_chunk_metadata(self):
        # Chunk-axis indexing is in terms of the user's *stage-axis* buffer
        # layout. For each f_stage input arg we record:
        #   in_stage_mode      : STAGE / SHARED / COUPLED (untouched from user spec)
        #   in_chunk_offset    : original COUPLED offset, 0 otherwise
        #   in_total_stages    : leading-axis extent of user buffer for this arg
        #   in_nnz_per_stage   : f_stage's per-stage nnz_in (NOT f_chunk's)
        n = self.n_stages
        in_modes, in_offs, in_totals, in_nnz_ps = [], [], [], []
        for i, s in enumerate(self.input_specs):
            in_modes.append(int(s.mode))
            in_offs.append(s.offset if s.mode == StageMode.COUPLED else 0)
            in_totals.append(s.stage_count(n))
            in_nnz_ps.append(self.f_stage.nnz_in(i))
        self._in_stage_mode    = _to_gpu_int(in_modes  or [0])
        self._in_chunk_offset  = _to_gpu_int(in_offs   or [0])
        self._in_total_stages  = _to_gpu_int(in_totals or [1])
        self._in_nnz_per_stage = _to_gpu_int(in_nnz_ps or [0])

        out_modes, out_totals, out_nnz_ps = [], [], []
        for j, s in enumerate(self.output_specs):
            out_modes.append(int(s.mode))
            out_totals.append(s.stage_count(n))
            out_nnz_ps.append(self.f_stage.nnz_out(j))
        self._out_stage_mode    = _to_gpu_int(out_modes  or [0])
        self._out_total_stages  = _to_gpu_int(out_totals or [1])
        self._out_nnz_per_stage = _to_gpu_int(out_nnz_ps or [0])

    def _setup_io(self, f_stage: casadi.Function, num_instances: int, n_stages: int):
        nin, nout = f_stage.n_in(), f_stage.n_out()

        self._output_tensors = []
        for i in range(nout):
            sc = self.output_specs[i].stage_count(n_stages)
            self._output_tensors.append(torch.zeros(
                num_instances, sc, f_stage.nnz_out(i),
                device=self._device, dtype=torch.double).contiguous())

        self._input_tensors = []
        for i in range(nin):
            sc = self.input_specs[i].stage_count(n_stages)
            self._input_tensors.append(torch.zeros(
                num_instances, sc, f_stage.nnz_in(i),
                device=self._device, dtype=torch.double).contiguous())

        self._input_ptrs = torch.zeros(nin, device=self._device,
                                       dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(nout, device=self._device,
                                        dtype=torch.int64).contiguous()
        for i in range(nin):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(nout):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self.outputs_sparse = self._output_tensors

        self._coo_rows, self._coo_cols = [], []
        for i in range(nout):
            rows, cols = f_stage.sparsity_out(i).get_triplet()
            self._coo_rows.append(torch.tensor(rows, device=self._device, dtype=torch.int64))
            self._coo_cols.append(torch.tensor(cols, device=self._device, dtype=torch.int64))

    def _ptr(self, t: torch.Tensor):
        return ctypes.c_void_p(t.data_ptr())

    def _validate_input_shape(self, idx: int, t: torch.Tensor):
        spec = self.input_specs[idx]
        sc = spec.stage_count(self.n_stages)
        nnz = self.f_stage.nnz_in(idx)
        if spec.mode == StageMode.SHARED and t.dim() == 2:
            expected = (self.num_instances, nnz)
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"input {idx} (SHARED) shape {tuple(t.shape)} != {expected}")
            return
        expected = (self.num_instances, sc, nnz)
        if tuple(t.shape) != expected:
            raise ValueError(
                f"input {idx} shape {tuple(t.shape)} != {expected} "
                f"(mode={spec.mode.name}, n_stages={self.n_stages}, span={spec.span})")

    def evaluate(self, inputs: List[torch.Tensor]):
        for j in range(self.f_stage.n_out()):
            self._output_tensors[j].zero_()

        for i in range(self.f_stage.n_in()):
            self._validate_input_shape(i, inputs[i])
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = inputs[i].data_ptr()

        # Re-assert this wrapper's shared-memory requirement before each launch.
        # cudaFuncSetAttribute(MaxDynamicSharedMemorySize) is per-kernel-symbol,
        # so a sibling wrapper with a smaller plan could otherwise lower it
        # underneath us and our launch would fail with cudaErrorInvalidValue.
        self._fn_library.configure_chunk_shared_memory(self._shared_bytes)

        fn_input  = ctypes.cast(self._input_ptrs.data_ptr(),  ctypes.POINTER(ctypes.c_int))
        fn_output = ctypes.cast(self._output_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))

        self.eval_time = self._fn_library.evaluate_lp_chunk(
            fn_input,
            fn_output,
            self.num_instances,
            self.n_chunks,
            self.chunk_size,
            self._block_size,
            self._shared_bytes,
            self._n_consts,
            self._ptr(self._const_slots),
            self._ptr(self._const_vals),
            self._n_input_maps,
            self._ptr(self._input_slots),
            self._ptr(self._input_args),
            self._ptr(self._input_elems),
            self._ptr(self._in_stage_mode),
            self._ptr(self._in_chunk_offset),
            self._ptr(self._in_total_stages),
            self._ptr(self._in_nnz_per_stage),
            self._n_levels,
            self._ptr(self._level_offsets),
            self._ptr(self._level_widths),
            self._ptr(self._node_ops),
            self._ptr(self._node_outs),
            self._ptr(self._node_in_a),
            self._ptr(self._node_in_b),
            self._n_output_maps,
            self._ptr(self._out_slots),
            self._ptr(self._out_args),
            self._ptr(self._out_elems),
            self._ptr(self._out_stage_mode),
            self._ptr(self._out_total_stages),
            self._ptr(self._out_nnz_per_stage),
        )

    def get_coo(self, out_idx: int = 0):
        return (self._coo_rows[out_idx],
                self._coo_cols[out_idx],
                self.outputs_sparse[out_idx])


# ---------------------------------------------------------------------------
# Tiled stage kernel — closes the per-stage > smem regime
# ---------------------------------------------------------------------------

class CusadiTiledStageLevelParallelFunction:
    """Stage-aware tiled level-parallel CUDA wrapper.

    Used when a single f_stage's working set exceeds the device's shared
    memory budget. Each block processes one (instance, stage) pair, and
    the per-stage DAG is tiled into chunks with spill/reload at boundaries
    — the same mechanism as `CusadiTiledLevelParallelFunction`, plus
    stage-axis I/O indexing.

    chunk_size is implicitly 1 along the stage axis: fusing multiple
    already-too-big stages into one block would only worsen the smem
    problem. Use `CusadiHybridStageLevelParallelFunction` instead when
    f_stage fits and you want to fuse stages.
    """

    def __init__(
        self,
        f_stage: casadi.Function,
        num_instances: int,
        n_stages: int,
        input_specs: Sequence[StageSpec],
        output_specs: Sequence[StageSpec],
        max_smem_slots: int = 12000,
        lib_name: str = "level_parallel_stage_tiled_kernel",
        tiled_plan: Optional[TiledLevelPlan] = None,
    ):
        assert torch.cuda.is_available()
        _validate_specs(f_stage, n_stages, input_specs, output_specs)

        self.f_stage = f_stage
        self.fn_casadi = f_stage  # API parity
        self.fn_name = f_stage.name()
        self.num_instances = num_instances
        self.n_stages = n_stages
        self.chunk_size = 1
        self.input_specs = list(input_specs)
        self.output_specs = list(output_specs)
        self._device = "cuda"

        lib_path = os.path.join(CUSADI_LP_BUILD_DIR, f"lib{lib_name}.so")
        self._fn_library = ctypes.CDLL(lib_path)
        self._fn_library.evaluate_lp_stage_tiled.restype = ctypes.c_float
        self._fn_library.configure_stage_tiled_shared_memory.restype = ctypes.c_int

        if tiled_plan is None:
            tiled_plan = extract_tiled_level_plan(
                f_stage, max_smem_slots=max_smem_slots)
        self._plan = tiled_plan
        self._block_size = pick_block_size(tiled_plan)
        self._shared_bytes = tiled_plan.peak_live_slots * 8
        self._fn_library.configure_stage_tiled_shared_memory(self._shared_bytes)

        self._upload_plan(tiled_plan)
        self._upload_stage_metadata()
        self._setup_io(f_stage, num_instances, n_stages)

    def _upload_plan(self, plan: TiledLevelPlan):
        # --- Constants ---
        if plan.constants:
            self._const_slots = _to_gpu_int([c.ssa_slot for c in plan.constants])
            self._const_vals = _to_gpu_double([c.value for c in plan.constants])
        else:
            self._const_slots = _to_gpu_int([0])
            self._const_vals = _to_gpu_double([0.0])
        self._n_consts = len(plan.constants)

        # --- Input mappings ---
        if plan.input_mappings:
            self._input_slots = _to_gpu_int([m.ssa_slot for m in plan.input_mappings])
            self._input_args = _to_gpu_int([m.input_arg for m in plan.input_mappings])
            self._input_elems = _to_gpu_int([m.element_idx for m in plan.input_mappings])
        else:
            self._input_slots = _to_gpu_int([0])
            self._input_args = _to_gpu_int([0])
            self._input_elems = _to_gpu_int([0])
        self._n_input_maps = len(plan.input_mappings)

        # --- Compute nodes (flat) ---
        all_ops, all_outs, all_in_a, all_in_b = [], [], [], []
        level_offsets, level_widths = [], []
        offset = 0
        for level in plan.compute_levels:
            level_offsets.append(offset)
            level_widths.append(level.width)
            for node in level.nodes:
                all_ops.append(node.op_id)
                all_outs.append(node.ssa_out)
                all_in_a.append(node.ssa_inputs[0] if len(node.ssa_inputs) >= 1 else 0)
                all_in_b.append(node.ssa_inputs[1] if len(node.ssa_inputs) >= 2 else 0)
            offset += level.width

        if all_ops:
            self._node_ops = _to_gpu_int(all_ops)
            self._node_outs = _to_gpu_int(all_outs)
            self._node_in_a = _to_gpu_int(all_in_a)
            self._node_in_b = _to_gpu_int(all_in_b)
            self._level_offsets = _to_gpu_int(level_offsets)
            self._level_widths = _to_gpu_int(level_widths)
        else:
            self._node_ops = _to_gpu_int([0])
            self._node_outs = _to_gpu_int([0])
            self._node_in_a = _to_gpu_int([0])
            self._node_in_b = _to_gpu_int([0])
            self._level_offsets = _to_gpu_int([0])
            self._level_widths = _to_gpu_int([0])
        self._n_levels = len(level_offsets)

        # --- Output mappings ---
        if plan.output_mappings:
            self._out_slots = _to_gpu_int([m.ssa_slot for m in plan.output_mappings])
            self._out_args = _to_gpu_int([m.output_arg for m in plan.output_mappings])
            self._out_elems = _to_gpu_int([m.element_idx for m in plan.output_mappings])
        else:
            self._out_slots = _to_gpu_int([0])
            self._out_args = _to_gpu_int([0])
            self._out_elems = _to_gpu_int([0])
        self._n_output_maps = len(plan.output_mappings)

        # --- Tiling: per-tile chunk indices into compute_levels ---
        self._n_tiles = plan.n_chunks
        self._spill_stride = plan.spill_buffer_stride

        cl_idx = 0
        chunk_lev_starts, chunk_lev_ends = [], []
        for chunk in plan.chunks:
            start_cl = cl_idx
            while cl_idx < len(plan.compute_levels):
                if plan.compute_levels[cl_idx].depth >= chunk.level_end:
                    break
                cl_idx += 1
            chunk_lev_starts.append(start_cl)
            chunk_lev_ends.append(cl_idx)
        self._chunk_lev_start = _to_gpu_int(chunk_lev_starts)
        self._chunk_lev_end = _to_gpu_int(chunk_lev_ends)

        # --- Tiling: boundaries (inter-tile spill/reload) ---
        bnd_spill_smem_flat, bnd_spill_pos_flat = [], []
        bnd_reload_smem_flat, bnd_reload_pos_flat = [], []
        bnd_spill_offs, bnd_spill_cnts = [], []
        bnd_reload_offs, bnd_reload_cnts = [], []
        s_off, r_off = 0, 0
        for b in plan.boundaries:
            bnd_spill_offs.append(s_off)
            bnd_spill_cnts.append(len(b.spill_src_slots))
            bnd_spill_smem_flat.extend(b.spill_src_slots)
            bnd_spill_pos_flat.extend(b.spill_buffer_pos)
            s_off += len(b.spill_src_slots)

            bnd_reload_offs.append(r_off)
            bnd_reload_cnts.append(len(b.reload_dst_slots))
            bnd_reload_smem_flat.extend(b.reload_dst_slots)
            bnd_reload_pos_flat.extend(b.reload_buffer_pos)
            r_off += len(b.reload_dst_slots)

        self._bnd_spill_off = _to_gpu_int(bnd_spill_offs or [0])
        self._bnd_spill_cnt = _to_gpu_int(bnd_spill_cnts or [0])
        self._bnd_spill_smem = _to_gpu_int(bnd_spill_smem_flat or [0])
        self._bnd_spill_pos = _to_gpu_int(bnd_spill_pos_flat or [0])
        self._bnd_reload_off = _to_gpu_int(bnd_reload_offs or [0])
        self._bnd_reload_cnt = _to_gpu_int(bnd_reload_cnts or [0])
        self._bnd_reload_smem = _to_gpu_int(bnd_reload_smem_flat or [0])
        self._bnd_reload_pos = _to_gpu_int(bnd_reload_pos_flat or [0])

        # --- Tiling: sub-batch spills (per compute_levels index) ---
        lev_sb_offs = [0] * self._n_levels
        lev_sb_cnts = [0] * self._n_levels
        sb_smem_flat, sb_pos_flat = [], []
        sb_off = 0
        for cl_idx, (smem_slots, buf_positions) in plan.sub_batch_spills.items():
            lev_sb_offs[cl_idx] = sb_off
            lev_sb_cnts[cl_idx] = len(smem_slots)
            sb_smem_flat.extend(smem_slots)
            sb_pos_flat.extend(buf_positions)
            sb_off += len(smem_slots)

        self._lev_sb_off = _to_gpu_int(lev_sb_offs)
        self._lev_sb_cnt = _to_gpu_int(lev_sb_cnts)
        self._sb_spill_smem = _to_gpu_int(sb_smem_flat or [0])
        self._sb_spill_pos = _to_gpu_int(sb_pos_flat or [0])

    def _upload_stage_metadata(self):
        n = self.n_stages
        in_modes  = [int(s.mode) for s in self.input_specs]
        in_offs   = [s.offset if s.mode == StageMode.COUPLED else 0
                     for s in self.input_specs]
        in_totals = [s.stage_count(n) for s in self.input_specs]
        in_nnz_ps = [self.f_stage.nnz_in(i) for i in range(self.f_stage.n_in())]
        out_modes  = [int(s.mode) for s in self.output_specs]
        out_totals = [s.stage_count(n) for s in self.output_specs]
        out_nnz_ps = [self.f_stage.nnz_out(j) for j in range(self.f_stage.n_out())]
        self._in_stage_mode    = _to_gpu_int(in_modes  or [0])
        self._in_chunk_offset  = _to_gpu_int(in_offs   or [0])
        self._in_total_stages  = _to_gpu_int(in_totals or [1])
        self._in_nnz_per_stage = _to_gpu_int(in_nnz_ps or [0])
        self._out_stage_mode    = _to_gpu_int(out_modes  or [0])
        self._out_total_stages  = _to_gpu_int(out_totals or [1])
        self._out_nnz_per_stage = _to_gpu_int(out_nnz_ps or [0])

    def _setup_io(self, f_stage: casadi.Function, num_instances: int, n_stages: int):
        nin, nout = f_stage.n_in(), f_stage.n_out()

        self._output_tensors = []
        for i in range(nout):
            sc = self.output_specs[i].stage_count(n_stages)
            self._output_tensors.append(torch.zeros(
                num_instances, sc, f_stage.nnz_out(i),
                device=self._device, dtype=torch.double).contiguous())

        self._input_tensors = []
        for i in range(nin):
            sc = self.input_specs[i].stage_count(n_stages)
            self._input_tensors.append(torch.zeros(
                num_instances, sc, f_stage.nnz_in(i),
                device=self._device, dtype=torch.double).contiguous())

        self._input_ptrs = torch.zeros(nin, device=self._device,
                                       dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(nout, device=self._device,
                                        dtype=torch.int64).contiguous()
        for i in range(nin):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(nout):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self.outputs_sparse = self._output_tensors

        # Spill buffer: per (instance, stage) — shape (B * n_stages, spill_stride).
        if self._spill_stride > 0:
            self._spill_buffer = torch.zeros(
                num_instances * n_stages * self._spill_stride,
                device=self._device, dtype=torch.double).contiguous()
        else:
            self._spill_buffer = torch.zeros(1, device=self._device,
                                              dtype=torch.double).contiguous()

        self._coo_rows, self._coo_cols = [], []
        for i in range(nout):
            rows, cols = f_stage.sparsity_out(i).get_triplet()
            self._coo_rows.append(torch.tensor(rows, device=self._device, dtype=torch.int64))
            self._coo_cols.append(torch.tensor(cols, device=self._device, dtype=torch.int64))

    def _ptr(self, t: torch.Tensor):
        return ctypes.c_void_p(t.data_ptr())

    def _validate_input_shape(self, idx: int, t: torch.Tensor):
        spec = self.input_specs[idx]
        sc = spec.stage_count(self.n_stages)
        nnz = self.f_stage.nnz_in(idx)
        if spec.mode == StageMode.SHARED and t.dim() == 2:
            expected = (self.num_instances, nnz)
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"input {idx} (SHARED) shape {tuple(t.shape)} != {expected}")
            return
        expected = (self.num_instances, sc, nnz)
        if tuple(t.shape) != expected:
            raise ValueError(
                f"input {idx} shape {tuple(t.shape)} != {expected}")

    def evaluate(self, inputs: List[torch.Tensor]):
        for j in range(self.f_stage.n_out()):
            self._output_tensors[j].zero_()

        for i in range(self.f_stage.n_in()):
            self._validate_input_shape(i, inputs[i])
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = inputs[i].data_ptr()

        # Re-assert shared-mem requirement (per-kernel-symbol attribute can
        # be lowered by a sibling wrapper between calls).
        self._fn_library.configure_stage_tiled_shared_memory(self._shared_bytes)

        fn_input  = ctypes.cast(self._input_ptrs.data_ptr(),  ctypes.POINTER(ctypes.c_int))
        fn_output = ctypes.cast(self._output_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))

        self.eval_time = self._fn_library.evaluate_lp_stage_tiled(
            fn_input, fn_output, self.num_instances, self.n_stages,
            self._block_size, self._shared_bytes,
            self._n_consts, self._ptr(self._const_slots), self._ptr(self._const_vals),
            self._n_input_maps,
            self._ptr(self._input_slots), self._ptr(self._input_args),
            self._ptr(self._input_elems),
            self._ptr(self._in_stage_mode), self._ptr(self._in_chunk_offset),
            self._ptr(self._in_total_stages), self._ptr(self._in_nnz_per_stage),
            self._n_levels, self._ptr(self._level_offsets), self._ptr(self._level_widths),
            self._ptr(self._node_ops), self._ptr(self._node_outs),
            self._ptr(self._node_in_a), self._ptr(self._node_in_b),
            self._n_output_maps,
            self._ptr(self._out_slots), self._ptr(self._out_args),
            self._ptr(self._out_elems),
            self._ptr(self._out_stage_mode),
            self._ptr(self._out_total_stages), self._ptr(self._out_nnz_per_stage),
            self._ptr(self._spill_buffer), self._spill_stride, self._n_tiles,
            self._ptr(self._chunk_lev_start), self._ptr(self._chunk_lev_end),
            self._ptr(self._bnd_spill_off), self._ptr(self._bnd_spill_cnt),
            self._ptr(self._bnd_spill_smem), self._ptr(self._bnd_spill_pos),
            self._ptr(self._bnd_reload_off), self._ptr(self._bnd_reload_cnt),
            self._ptr(self._bnd_reload_smem), self._ptr(self._bnd_reload_pos),
            self._ptr(self._lev_sb_off), self._ptr(self._lev_sb_cnt),
            self._ptr(self._sb_spill_smem), self._ptr(self._sb_spill_pos),
        )

    def get_coo(self, out_idx: int = 0):
        return (self._coo_rows[out_idx],
                self._coo_cols[out_idx],
                self.outputs_sparse[out_idx])


# ---------------------------------------------------------------------------
# Reference simulator for numerical verification (no GPU needed)
# ---------------------------------------------------------------------------

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
