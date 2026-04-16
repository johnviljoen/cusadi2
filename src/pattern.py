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
"""

from typing import Tuple

import casadi
import numpy as np
import torch

from src.CusadiFunction import CusadiFunction


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
        """
        Args:
            casadi_fn: the pattern body (e.g., a stage-cost Jacobian).
            n_instances: how many times this pattern appears in the problem.
            row_offsets: (n_instances,) int — global row offset per instance.
            col_offsets: (n_instances,) int — global col offset per instance.
            out_idx: which output of casadi_fn carries the sparse matrix.
        """
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

    def assemble_values(self, cusadi_fn: CusadiFunction) -> np.ndarray:
        """Extract the values from an evaluated CusadiFunction.

        Returns a 1D float64 array of shape (total_nnz,) ordered consistently
        with global_rows and global_cols: instance 0's nnz values first,
        then instance 1's, etc.
        """
        vals = cusadi_fn.outputs_sparse[self.out_idx]
        return vals.detach().cpu().numpy().astype(np.float64).reshape(-1)

    def get_global_coo(
        self, cusadi_fn: CusadiFunction
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full COO triplet: (global_rows, global_cols, global_vals)."""
        return (self.global_rows, self.global_cols, self.assemble_values(cusadi_fn))

    def evaluate_and_assemble(
        self, inputs: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a CusadiFunction, evaluate it, and return the global COO.

        Args:
            inputs: list of CUDA double tensors shaped (n_instances, nnz_in[i]).

        Returns:
            (global_rows, global_cols, global_vals) numpy arrays.
        """
        fn = CusadiFunction(self.casadi_fn, self.n_instances)
        fn.evaluate(inputs)
        torch.cuda.synchronize()
        return self.get_global_coo(fn)
