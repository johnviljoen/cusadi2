"""
CusadiLevelParallelFunction: data-driven level-parallel CUDA wrapper.

Flattens a LevelPlan into GPU integer/double arrays, loads the fixed kernel
.so once, and passes the arrays as kernel arguments each evaluation call.
Same external interface as CusadiFunction (evaluate, outputs_sparse, get_coo).
"""

import ctypes
import os
from typing import List

import casadi
import numpy as np
import torch

from cusadi import CUSADI_LP_BUILD_DIR
from cusadi.level_parallel.dag import LevelPlan, extract_level_plan
from cusadi.level_parallel.codegen import pick_block_size
from cusadi.level_parallel.serialize import FlatPlan, flatten_plan


def _to_gpu_int(arr):
    return torch.tensor(arr, device="cuda", dtype=torch.int32).contiguous()


def _to_gpu_double(arr):
    return torch.tensor(arr, device="cuda", dtype=torch.double).contiguous()


class LPPlanTensorsMixin:
    """GPU-resident level-parallel plan tensors, uploaded from a FlatPlan.

    Single source of truth: the flattening lives in
    `cusadi.level_parallel.serialize.flatten_plan`; every torch LP wrapper
    (plain / tiled / stage / hybrid) mixes this in and uploads through
    `_upload_flat_plan` so slot/offset conventions cannot drift between
    backends. Declaring the attributes here also keeps them statically visible
    to type checkers even though they are assigned in the method below.
    """

    _const_slots: torch.Tensor
    _const_vals: torch.Tensor
    _n_consts: int
    _input_slots: torch.Tensor
    _input_args: torch.Tensor
    _input_elems: torch.Tensor
    _n_input_maps: int
    _nnz_in_gpu: torch.Tensor
    _node_ops: torch.Tensor
    _node_outs: torch.Tensor
    _node_in_a: torch.Tensor
    _node_in_b: torch.Tensor
    _level_offsets: torch.Tensor
    _level_widths: torch.Tensor
    _n_levels: int
    _out_slots: torch.Tensor
    _out_args: torch.Tensor
    _out_elems: torch.Tensor
    _n_output_maps: int
    _nnz_out_gpu: torch.Tensor

    def _upload_flat_plan(self, flat: FlatPlan, *, with_nnz: bool = True):
        """Upload a shared-serializer FlatPlan to GPU tensors.

        Empty arrays fall back to a single [0]/[0.0] element so each tensor's
        .data_ptr() stays valid. `with_nnz` uploads `_nnz_in_gpu`/`_nnz_out_gpu`
        for the plain-LP-style kernels; the stage chunk/tiled kernels carry
        per-stage nnz separately and pass with_nnz=False.
        """
        self._const_slots = _to_gpu_int(flat.const_slots or [0])
        self._const_vals = _to_gpu_double(flat.const_vals or [0.0])
        self._n_consts = flat.n_consts

        self._input_slots = _to_gpu_int(flat.input_slots or [0])
        self._input_args = _to_gpu_int(flat.input_args or [0])
        self._input_elems = _to_gpu_int(flat.input_elems or [0])
        self._n_input_maps = flat.n_input_maps

        self._node_ops = _to_gpu_int(flat.node_ops or [0])
        self._node_outs = _to_gpu_int(flat.node_outs or [0])
        self._node_in_a = _to_gpu_int(flat.node_in_a or [0])
        self._node_in_b = _to_gpu_int(flat.node_in_b or [0])
        self._level_offsets = _to_gpu_int(flat.level_offsets or [0])
        self._level_widths = _to_gpu_int(flat.level_widths or [0])
        self._n_levels = flat.n_levels

        self._out_slots = _to_gpu_int(flat.out_slots or [0])
        self._out_args = _to_gpu_int(flat.out_args or [0])
        self._out_elems = _to_gpu_int(flat.out_elems or [0])
        self._n_output_maps = flat.n_output_maps

        if with_nnz:
            self._nnz_in_gpu = _to_gpu_int(flat.nnz_in)
            self._nnz_out_gpu = _to_gpu_int(flat.nnz_out)


class CusadiLevelParallelFunction(LPPlanTensorsMixin):
    """Wraps the data-driven level-parallel kernel with CusadiFunction's API."""

    def __init__(self, fn_casadi: casadi.Function, num_instances: int,
                 lib_name: str = "level_parallel_kernel"):
        assert torch.cuda.is_available()
        self.fn_casadi = fn_casadi
        self.fn_name = fn_casadi.name()
        self.num_instances = num_instances
        self._device = "cuda"

        lib_path = os.path.join(CUSADI_LP_BUILD_DIR, f"lib{lib_name}.so")
        self._fn_library = ctypes.CDLL(lib_path)
        self._fn_library.evaluate_lp.restype = ctypes.c_float
        self._fn_library.configure_shared_memory.restype = ctypes.c_int

        plan = extract_level_plan(fn_casadi)
        self._block_size = pick_block_size(plan)
        self._shared_bytes = plan.peak_live_slots * 8

        self._fn_library.configure_shared_memory(self._shared_bytes)

        self._upload_plan(plan)
        self._setup_io(fn_casadi, num_instances)

    def _upload_plan(self, plan: LevelPlan):
        """Flatten LevelPlan into GPU arrays (via shared serialize.flatten_plan)."""
        self._upload_flat_plan(flatten_plan(plan, block_size=self._block_size))

    def _setup_io(self, fn_casadi, num_instances):
        """Allocate input/output tensors and pointer arrays."""
        n = num_instances
        self._input_tensors = [
            torch.zeros(n, fn_casadi.nnz_in(i), device=self._device, dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_in())
        ]
        self._output_tensors = [
            torch.zeros(n, fn_casadi.nnz_out(i), device=self._device, dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_out())
        ]
        self._input_ptrs = torch.zeros(fn_casadi.n_in(), device=self._device, dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(fn_casadi.n_out(), device=self._device, dtype=torch.int64).contiguous()
        for i in range(fn_casadi.n_in()):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(fn_casadi.n_out()):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self.outputs_sparse = self._output_tensors

        self._coo_rows = []
        self._coo_cols = []
        for i in range(fn_casadi.n_out()):
            rows, cols = fn_casadi.sparsity_out(i).get_triplet()
            self._coo_rows.append(torch.tensor(rows, device=self._device, dtype=torch.int64))
            self._coo_cols.append(torch.tensor(cols, device=self._device, dtype=torch.int64))

    def _ptr(self, t):
        """Get ctypes pointer from a tensor's data_ptr."""
        return ctypes.c_void_p(t.data_ptr())

    def evaluate(self, inputs: List[torch.Tensor]):
        for i in range(self.fn_casadi.n_out()):
            self._output_tensors[i].zero_()

        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = inputs[i].data_ptr()

        fn_input = ctypes.cast(self._input_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_output = ctypes.cast(self._output_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))

        self.eval_time = self._fn_library.evaluate_lp(
            fn_input,
            fn_output,
            self.num_instances,
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
        )

    def get_coo(self, out_idx: int = 0):
        return (self._coo_rows[out_idx],
                self._coo_cols[out_idx],
                self.outputs_sparse[out_idx])


class CusadiTiledLevelParallelFunction(LPPlanTensorsMixin):
    """Wraps the tiled level-parallel kernel for functions that may need
    shared-memory tiling (large work vectors exceeding hardware smem limits).

    For single-chunk plans (functions that fit in smem), the tiled kernel
    degenerates to the untiled path: zero spills, zero reloads, identical
    performance.
    """

    def __init__(self, fn_casadi: casadi.Function, num_instances: int,
                 tiled_plan, lib_name: str = "level_parallel_tiled_kernel"):
        assert torch.cuda.is_available()
        from cusadi.level_parallel.codegen import pick_block_size

        self.fn_casadi = fn_casadi
        self.fn_name = fn_casadi.name()
        self.num_instances = num_instances
        self._device = "cuda"
        self._plan = tiled_plan

        lib_path = os.path.join(CUSADI_LP_BUILD_DIR, f"lib{lib_name}.so")
        self._fn_library = ctypes.CDLL(lib_path)
        self._fn_library.evaluate_lp_tiled.restype = ctypes.c_float
        self._fn_library.configure_tiled_shared_memory.restype = ctypes.c_int

        self._block_size = pick_block_size(tiled_plan)
        self._shared_bytes = tiled_plan.peak_live_slots * 8

        self._fn_library.configure_tiled_shared_memory(self._shared_bytes)

        self._upload_plan(tiled_plan)
        self._setup_io(fn_casadi, num_instances)

    def _upload_plan(self, plan):
        """Flatten TiledLevelPlan into GPU arrays.

        The base LevelPlan portion goes through the shared serializer; the
        tiling-specific arrays (chunks / boundaries / sub-batch spills) are
        unique to TiledLevelPlan and stay inline below."""
        self._upload_flat_plan(flatten_plan(plan, block_size=self._block_size))

        # --- Tiling: chunks ---
        self._n_chunks = plan.n_chunks
        self._spill_stride = plan.spill_buffer_stride

        # Map compute_levels indices to chunk ranges.
        # chunk_lev_start[c] / chunk_lev_end[c] index into the flat level arrays.
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

        # --- Tiling: boundaries ---
        n_bnd = len(plan.boundaries)
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

        # --- Tiling: sub-batch spills ---
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

    def _setup_io(self, fn_casadi, num_instances):
        n = num_instances
        self._input_tensors = [
            torch.zeros(n, fn_casadi.nnz_in(i), device=self._device,
                        dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_in())
        ]
        self._output_tensors = [
            torch.zeros(n, fn_casadi.nnz_out(i), device=self._device,
                        dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_out())
        ]
        self._input_ptrs = torch.zeros(fn_casadi.n_in(), device=self._device,
                                       dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(fn_casadi.n_out(), device=self._device,
                                        dtype=torch.int64).contiguous()
        for i in range(fn_casadi.n_in()):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(fn_casadi.n_out()):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self.outputs_sparse = self._output_tensors

        # Spill buffer: per-instance, contiguous
        if self._spill_stride > 0:
            self._spill_buffer = torch.zeros(
                num_instances * self._spill_stride, device=self._device,
                dtype=torch.double).contiguous()
        else:
            self._spill_buffer = torch.zeros(1, device=self._device,
                                             dtype=torch.double).contiguous()

        self._coo_rows, self._coo_cols = [], []
        for i in range(fn_casadi.n_out()):
            rows, cols = fn_casadi.sparsity_out(i).get_triplet()
            self._coo_rows.append(torch.tensor(rows, device=self._device, dtype=torch.int64))
            self._coo_cols.append(torch.tensor(cols, device=self._device, dtype=torch.int64))

    def _ptr(self, t):
        return ctypes.c_void_p(t.data_ptr())

    def evaluate(self, inputs: List[torch.Tensor]):
        for i in range(self.fn_casadi.n_out()):
            self._output_tensors[i].zero_()

        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = inputs[i].data_ptr()

        fn_input = ctypes.cast(self._input_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_output = ctypes.cast(self._output_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))

        self.eval_time = self._fn_library.evaluate_lp_tiled(
            fn_input, fn_output, self.num_instances,
            self._block_size, self._shared_bytes,
            # Init
            self._n_consts, self._ptr(self._const_slots), self._ptr(self._const_vals),
            self._n_input_maps, self._ptr(self._input_slots),
            self._ptr(self._input_args), self._ptr(self._input_elems),
            self._ptr(self._nnz_in_gpu),
            # Levels
            self._n_levels, self._ptr(self._level_offsets), self._ptr(self._level_widths),
            self._ptr(self._node_ops), self._ptr(self._node_outs),
            self._ptr(self._node_in_a), self._ptr(self._node_in_b),
            # Outputs
            self._n_output_maps, self._ptr(self._out_slots),
            self._ptr(self._out_args), self._ptr(self._out_elems),
            self._ptr(self._nnz_out_gpu),
            # Tiling
            self._ptr(self._spill_buffer), self._spill_stride, self._n_chunks,
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
