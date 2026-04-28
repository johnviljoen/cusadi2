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
from typing import List, Sequence

import casadi
import numpy as np
import torch

from src import CUSADI_BUILD_DIR
from src.level_parallel.dag import LevelPlan, extract_level_plan
from src.level_parallel.codegen import pick_block_size


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

        lib_path = os.path.join(CUSADI_BUILD_DIR, f"lib{lib_name}.so")
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
