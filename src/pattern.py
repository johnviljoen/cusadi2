"""
Host-side COO index pre-computation for per-pattern sparse assembly.

A PatternAssembler takes a CasADi Function (the pattern body), a count of
instances, and per-instance (row_offset, col_offset) arrays. It pre-builds
the global COO index arrays once at setup time by tiling the per-pattern
sparsity and applying offsets. At evaluation time, the kernel writes values
only; assembly is a single reshape.

This implements the "host-side pre-compute" design adopted after the C0
read-through: kernels never compute or write index arrays, and cuDSS gets
a fixed sparsity pattern that it analyzes once.

CusadiFunction2D wraps the 2D-launch kernel (grid.x=batch, grid.y*block=pattern).
PatternAssembler uses it when batch_size > 0 to evaluate across both axes.
"""

import ctypes
import os
from typing import Dict, List, Tuple

import casadi
import numpy as np
import torch
from scipy.sparse import coo_matrix

from src import CUSADI_BUILD_DIR
from src.CusadiFunction import CusadiFunction


class CusadiFunction2D:
    """Thin wrapper for 2D-launch kernels (batch × pattern instances).

    Loads lib{fn_name}_2d.so and calls evaluate(inputs, work, outputs,
    batch_size, n_instances) with the 5-arg C interface.

    Memory layout is flat: total = batch_size * n_instances.
    Instance (b, p) lives at flat index b * n_instances + p.
    """

    def __init__(self, fn_casadi: casadi.Function, batch_size: int, n_instances: int):
        lib_filepath = os.path.join(CUSADI_BUILD_DIR, f"lib{fn_casadi.name()}_2d.so")
        self.fn_casadi = fn_casadi
        self.batch_size = batch_size
        self.n_instances = n_instances
        self._total = batch_size * n_instances
        self._device = "cuda"

        self._fn_library = ctypes.CDLL(lib_filepath)
        self._fn_library.evaluate.restype = ctypes.c_float

        self._input_tensors = [
            torch.zeros(self._total, fn_casadi.nnz_in(i),
                        device=self._device, dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_in())
        ]
        self._output_tensors = [
            torch.zeros(self._total, fn_casadi.nnz_out(i),
                        device=self._device, dtype=torch.double).contiguous()
            for i in range(fn_casadi.n_out())
        ]
        self._work_tensor = torch.zeros(
            self._total, fn_casadi.sz_w(),
            device=self._device, dtype=torch.double).contiguous()

        self._input_ptrs = torch.zeros(
            fn_casadi.n_in(), device="cuda", dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(
            fn_casadi.n_out(), device="cuda", dtype=torch.int64).contiguous()
        for i in range(fn_casadi.n_in()):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(fn_casadi.n_out()):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self.outputs_sparse = self._output_tensors

    def evaluate(self, inputs: List[torch.Tensor]):
        for i in range(self.fn_casadi.n_out()):
            self._output_tensors[i].zero_()
        self._work_tensor.zero_()

        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = inputs[i].data_ptr()

        fn_input = ctypes.cast(
            self._input_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_output = ctypes.cast(
            self._output_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_work = ctypes.cast(
            self._work_tensor.data_ptr(), ctypes.POINTER(ctypes.c_double))

        self.eval_time = self._fn_library.evaluate(
            fn_input, fn_work, fn_output,
            self.batch_size, self.n_instances)


class PatternAssembler:
    """Pre-builds global COO index arrays for a single pattern type."""

    def __init__(
        self,
        casadi_fn: casadi.Function,
        n_instances: int,
        row_offsets: np.ndarray,
        col_offsets: np.ndarray,
        out_idx: int = 0,
    ):
        self.casadi_fn = casadi_fn
        self.n_instances = n_instances
        self.out_idx = out_idx

        rows, cols = casadi_fn.sparsity_out(out_idx).get_triplet()
        self.base_rows = np.asarray(rows, dtype=np.int64)
        self.base_cols = np.asarray(cols, dtype=np.int64)
        self.nnz_per_instance = len(self.base_rows)
        self.total_nnz = self.nnz_per_instance * n_instances

        row_off = np.asarray(row_offsets, dtype=np.int64)
        col_off = np.asarray(col_offsets, dtype=np.int64)
        assert row_off.shape == (n_instances,), \
            f"row_offsets shape {row_off.shape} != ({n_instances},)"
        assert col_off.shape == (n_instances,), \
            f"col_offsets shape {col_off.shape} != ({n_instances},)"

        self.global_rows = (
            np.tile(self.base_rows, n_instances)
            + np.repeat(row_off, self.nnz_per_instance)
        )
        self.global_cols = (
            np.tile(self.base_cols, n_instances)
            + np.repeat(col_off, self.nnz_per_instance)
        )

    def assemble_values_flat(self, output_tensor: torch.Tensor) -> np.ndarray:
        """Flatten an (n_instances, nnz_per_instance) output tensor to 1D."""
        return output_tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)

    def assemble_values_batched(
        self, output_tensor: torch.Tensor, batch_size: int
    ) -> np.ndarray:
        """Reshape a (batch_size*n_instances, nnz_per_instance) output tensor
        to (batch_size, total_nnz) — one row of COO values per batch instance.
        """
        flat = output_tensor.detach().cpu().numpy().astype(np.float64)
        return flat.reshape(batch_size, self.n_instances, -1).reshape(
            batch_size, self.total_nnz)

    def evaluate_and_assemble(
        self, inputs: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1D launch (C3 path): n_instances as batch, no outer batch axis.

        Returns (global_rows, global_cols, global_vals) numpy arrays.
        """
        fn = CusadiFunction(self.casadi_fn, self.n_instances)
        fn.evaluate(inputs)
        torch.cuda.synchronize()
        vals = self.assemble_values_flat(fn.outputs_sparse[self.out_idx])
        return (self.global_rows, self.global_cols, vals)

    def evaluate_and_assemble_2d(
        self, inputs: list, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2D launch (C3.5 path): batch_size × n_instances.

        Args:
            inputs: list of CUDA double tensors shaped
                    (batch_size * n_instances, nnz_in[i]).
            batch_size: number of independent problem instances.

        Returns:
            (global_rows, global_cols, batched_vals) where:
            - global_rows, global_cols: (total_nnz,) — shared across batch
            - batched_vals: (batch_size, total_nnz) — per-batch values
        """
        fn = CusadiFunction2D(self.casadi_fn, batch_size, self.n_instances)
        fn.evaluate(inputs)
        torch.cuda.synchronize()
        vals = self.assemble_values_batched(
            fn.outputs_sparse[self.out_idx], batch_size)
        return (self.global_rows, self.global_cols, vals)


class ProblemAssembler:
    """Multi-pattern COO assembly with duplicate-entry handling.

    Takes multiple PatternAssemblers, concatenates their global COO arrays,
    and provides conversion to dense/CSR with duplicate summing (as cuDSS
    does during COO→CSR conversion).
    """

    def __init__(self, patterns: Dict[str, PatternAssembler], matrix_shape: Tuple[int, int]):
        self.patterns = patterns
        self.matrix_shape = matrix_shape
        self.total_nnz = sum(p.total_nnz for p in patterns.values())

        all_rows = [p.global_rows for p in patterns.values()]
        all_cols = [p.global_cols for p in patterns.values()]
        self.global_rows = np.concatenate(all_rows)
        self.global_cols = np.concatenate(all_cols)

    def assemble_dense(self, pattern_values: Dict[str, np.ndarray]) -> np.ndarray:
        """Assemble a dense matrix from per-pattern value arrays.

        Duplicate (row, col) entries are summed, matching cuDSS COO→CSR behavior.

        Args:
            pattern_values: {pattern_name: values_1d} matching self.patterns keys.
                Each values_1d has shape (total_nnz_for_that_pattern,).

        Returns:
            Dense matrix of self.matrix_shape with duplicates summed.
        """
        all_vals = np.concatenate(
            [pattern_values[name] for name in self.patterns])
        M = coo_matrix(
            (all_vals, (self.global_rows, self.global_cols)),
            shape=self.matrix_shape,
        )
        return M.toarray()

    def assemble_dense_batched(
        self, pattern_values: Dict[str, np.ndarray], batch_size: int
    ) -> np.ndarray:
        """Assemble dense matrices for each batch instance.

        Args:
            pattern_values: {name: (batch_size, total_nnz)} per pattern.

        Returns:
            (batch_size, n_rows, n_cols) dense array with duplicates summed.
        """
        result = np.zeros(
            (batch_size, self.matrix_shape[0], self.matrix_shape[1]),
            dtype=np.float64)
        for b in range(batch_size):
            vals_b = {name: v[b] for name, v in pattern_values.items()}
            result[b] = self.assemble_dense(vals_b)
        return result
