"""
CusadiLevelParallelFunction: Python wrapper for level-parallel CUDA kernels.

Same external interface as CusadiFunction (evaluate, outputs_sparse, get_coo)
so the validation harness works without modification.
"""

import ctypes
import os
from typing import List, Tuple

import casadi
import numpy as np
import torch

from src import CUSADI_BUILD_DIR


class CusadiLevelParallelFunction:
    """Wraps a level-parallel .so kernel with the same API as CusadiFunction."""

    def __init__(self, fn_casadi: casadi.Function, num_instances: int,
                 lib_suffix: str = "_lp"):
        assert torch.cuda.is_available()
        self.fn_casadi = fn_casadi
        self.fn_name = fn_casadi.name()
        self.num_instances = num_instances
        self._device = "cuda"

        lib_path = os.path.join(CUSADI_BUILD_DIR, f"lib{fn_casadi.name()}{lib_suffix}.so")
        self._fn_library = ctypes.CDLL(lib_path)
        self._fn_library.evaluate.restype = ctypes.c_float
        self._fn_library.configure_shared_memory()

        self._setup()

    def _setup(self):
        f = self.fn_casadi
        n = self.num_instances

        self._input_tensors = [
            torch.zeros(n, f.nnz_in(i), device=self._device, dtype=torch.double).contiguous()
            for i in range(f.n_in())
        ]
        self._output_tensors = [
            torch.zeros(n, f.nnz_out(i), device=self._device, dtype=torch.double).contiguous()
            for i in range(f.n_out())
        ]
        # Work tensor: level-parallel kernel uses shared memory, but the C
        # interface still accepts a work pointer for signature compatibility.
        # Allocate a minimal 1-element tensor as a dummy.
        self._work_tensor = torch.zeros(1, device=self._device, dtype=torch.double).contiguous()

        self._input_ptrs = torch.zeros(f.n_in(), device=self._device, dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(f.n_out(), device=self._device, dtype=torch.int64).contiguous()
        for i in range(f.n_in()):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(f.n_out()):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self.outputs_sparse = self._output_tensors

        self._coo_rows = []
        self._coo_cols = []
        for i in range(f.n_out()):
            rows, cols = f.sparsity_out(i).get_triplet()
            self._coo_rows.append(torch.tensor(rows, device=self._device, dtype=torch.int64))
            self._coo_cols.append(torch.tensor(cols, device=self._device, dtype=torch.int64))

    def evaluate(self, inputs: List[torch.Tensor]):
        for i in range(self.fn_casadi.n_out()):
            self._output_tensors[i].zero_()

        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = inputs[i].data_ptr()

        fn_input = ctypes.cast(self._input_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_output = ctypes.cast(self._output_ptrs.data_ptr(), ctypes.POINTER(ctypes.c_int))
        fn_work = ctypes.cast(self._work_tensor.data_ptr(), ctypes.POINTER(ctypes.c_double))

        self.eval_time = self._fn_library.evaluate(
            fn_input, fn_work, fn_output, self.num_instances)

    def get_coo(self, out_idx: int = 0):
        return (self._coo_rows[out_idx],
                self._coo_cols[out_idx],
                self.outputs_sparse[out_idx])
