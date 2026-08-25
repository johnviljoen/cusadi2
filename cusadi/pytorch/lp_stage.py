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
the stage axis)."""

import ctypes
import os
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
from cusadi.level_parallel.serialize import flatten_plan, flatten_stage_specs
from cusadi.level_parallel.stage import (StageMode, StageSpec, _validate_specs,
                                         _build_chunk_plan_analytic,
                                         _build_chunk_function,
                                         auto_tune_chunk_size)
from cusadi.pytorch.lp_function import LPPlanTensorsMixin


def _to_gpu_int(arr) -> torch.Tensor:
    return torch.tensor(arr, device="cuda", dtype=torch.int32).contiguous()


def _to_gpu_double(arr) -> torch.Tensor:
    return torch.tensor(arr, device="cuda", dtype=torch.double).contiguous()


class CusadiStageLevelParallelFunction(LPPlanTensorsMixin):
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
        self._upload_stage_metadata(f_stage)
        self._setup_io(f_stage, num_instances, n_stages)

    # -- Plan flattening: shared serialize.flatten_plan via the mixin. --
    def _upload_plan(self, plan: LevelPlan):
        self._upload_flat_plan(flatten_plan(plan, block_size=self._block_size))

    def _upload_stage_metadata(self, f_stage: casadi.Function):
        # Stage-axis modes/offsets come from the shared serializer. The kernel
        # takes per-arg instance spacings in doubles; the dense tensors this
        # class allocates make that stage_count * nnz for every argument.
        n = self.n_stages
        specs = flatten_stage_specs(self.input_specs, self.output_specs)
        in_spacing  = [specs.in_stage_count(i, n) * f_stage.nnz_in(i)
                       for i in range(len(self.input_specs))]
        out_spacing = [specs.out_stage_count(j, n) * f_stage.nnz_out(j)
                       for j in range(len(self.output_specs))]
        self._in_stage_mode   = _to_gpu_int(specs.in_mode   or [0])
        self._in_stage_offset = _to_gpu_int(specs.in_offset or [0])
        self._in_spacing      = _to_gpu_int(in_spacing or [1])
        self._out_stage_mode  = _to_gpu_int(specs.out_mode  or [0])
        self._out_spacing     = _to_gpu_int(out_spacing or [1])

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
            self._ptr(self._in_spacing),
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
            self._ptr(self._out_spacing),
        )

    def get_coo(self, out_idx: int = 0):
        return (self._coo_rows[out_idx],
                self._coo_cols[out_idx],
                self.outputs_sparse[out_idx])


class CusadiHybridStageLevelParallelFunction(LPPlanTensorsMixin):
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

    # ----- plan flattening: shared serialize.flatten_plan via the mixin -----
    # The chunk kernel carries per-stage nnz in chunk metadata, so with_nnz=False.
    def _upload_plan(self, plan: LevelPlan):
        self._upload_flat_plan(flatten_plan(plan, block_size=self._block_size),
                               with_nnz=False)

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


class CusadiTiledStageLevelParallelFunction(LPPlanTensorsMixin):
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
        # Base LevelPlan portion via the shared serializer (the chunk/tiled
        # kernel carries per-stage nnz separately, so with_nnz=False). The
        # tiling-specific arrays below are unique to TiledLevelPlan.
        self._upload_flat_plan(flatten_plan(plan, block_size=self._block_size),
                               with_nnz=False)

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
