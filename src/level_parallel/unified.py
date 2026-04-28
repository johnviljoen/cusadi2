"""
CusadiFunction2: unified routing across all three CusADi kernel paths.

Analyses the CasADi function's DAG, queries the device's shared memory
limit, and selects the best kernel automatically:
  - "lp":         original level-parallel (data-driven, no tiling)
  - "tiled":      tiled level-parallel (chunk boundaries + sub-batch spills)
  - "sequential": original CusADi sequential kernel (one thread per instance)

Construction does DAG analysis and compiles the LP kernels (~1.5 s total).
Sequential compilation is opt-in because it can take minutes for large functions.
"""

import os
import subprocess
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import casadi
import torch

from src import (CUSADI_ROOT_DIR, CUSADI_BUILD_DIR, CUSADI_CODEGEN_DIR)
from src.generateCUDACode import generateCUDACodeDouble, generateCMakeLists
from src.level_parallel.dag import (
    extract_level_plan, extract_tiled_level_plan, LevelPlan, TiledLevelPlan,
)
from src.level_parallel.codegen import (
    pick_block_size, write_kernel_source, write_tiled_kernel_source,
    write_stage_kernel_source,
)
from src.level_parallel.function import (
    CusadiLevelParallelFunction, CusadiTiledLevelParallelFunction,
)
from src.CusadiFunction import CusadiFunction


SMEM_MARGIN_BYTES = 4096  # reserve 4 KB for kernel stack


@dataclass
class DAGInfo:
    """Summary of the DAG analysis for a CasADi function."""
    function_name: str
    n_instructions: int
    n_in: int
    n_out: int
    nnz_in: List[int]
    nnz_out: List[int]
    compute_ops: int
    critical_path: int
    max_width: int
    median_width: int
    peak_live_slots: int
    shared_mem_bytes: int
    device_smem_limit: int
    fits_in_smem: bool
    # Tiling info (populated when tiling is needed)
    tiled_n_chunks: int
    tiled_peak_slots: int
    tiled_shared_mem_bytes: int
    tiled_spill_stride: int
    tiled_sub_batch_spills: int
    block_size: int
    default_mode: str

    def __repr__(self):
        lines = [
            f"DAGInfo({self.function_name}):",
            f"  instructions={self.n_instructions}, compute_ops={self.compute_ops}",
            f"  critical_path={self.critical_path}, max_width={self.max_width}",
            f"  peak_live={self.peak_live_slots} ({self.shared_mem_bytes/1024:.1f}KB), "
            f"device_limit={self.device_smem_limit/1024:.0f}KB, "
            f"fits={self.fits_in_smem}",
        ]
        if not self.fits_in_smem:
            lines.append(
                f"  tiled: {self.tiled_n_chunks} chunks, "
                f"peak={self.tiled_peak_slots} ({self.tiled_shared_mem_bytes/1024:.1f}KB), "
                f"spill_buf={self.tiled_spill_stride*8/1024:.0f}KB/inst, "
                f"sub_batch_spills={self.tiled_sub_batch_spills}"
            )
        lines.append(f"  block_size={self.block_size}, default_mode={self.default_mode!r}")
        return "\n".join(lines)


def _get_device_smem_limit() -> int:
    """Query the current CUDA device's max dynamic shared memory per block."""
    import ctypes as _ct
    libcudart = _ct.CDLL("libcudart.so")
    val = _ct.c_int(0)
    # cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
    libcudart.cudaDeviceGetAttribute(_ct.byref(val), 97, 0)
    return val.value


def _ensure_lp_kernel():
    """Ensure the original LP kernel .so is compiled."""
    so = os.path.join(CUSADI_BUILD_DIR, "liblevel_parallel_kernel.so")
    if os.path.exists(so):
        return
    os.makedirs(CUSADI_CODEGEN_DIR, exist_ok=True)
    cu = os.path.join(CUSADI_CODEGEN_DIR, "level_parallel_kernel.cu")
    write_kernel_source(cu)
    _cmake_build_lp_kernels()


def _ensure_tiled_kernel():
    """Ensure the tiled LP kernel .so is compiled."""
    so = os.path.join(CUSADI_BUILD_DIR, "liblevel_parallel_tiled_kernel.so")
    if os.path.exists(so):
        return
    os.makedirs(CUSADI_CODEGEN_DIR, exist_ok=True)
    cu = os.path.join(CUSADI_CODEGEN_DIR, "level_parallel_tiled_kernel.cu")
    write_tiled_kernel_source(cu)
    _cmake_build_lp_kernels()


def _ensure_stage_kernel():
    """Ensure the stage LP kernel .so is compiled."""
    so = os.path.join(CUSADI_BUILD_DIR, "liblevel_parallel_stage_kernel.so")
    if os.path.exists(so):
        return
    os.makedirs(CUSADI_CODEGEN_DIR, exist_ok=True)
    cu = os.path.join(CUSADI_CODEGEN_DIR, "level_parallel_stage_kernel.cu")
    write_stage_kernel_source(cu)
    _cmake_build_lp_kernels()


def _cmake_build_lp_kernels():
    """Write CMakeLists and build both LP kernels."""
    os.makedirs(CUSADI_BUILD_DIR, exist_ok=True)
    os.makedirs(CUSADI_CODEGEN_DIR, exist_ok=True)

    # Ensure all three .cu files exist
    lp_cu = os.path.join(CUSADI_CODEGEN_DIR, "level_parallel_kernel.cu")
    tiled_cu = os.path.join(CUSADI_CODEGEN_DIR, "level_parallel_tiled_kernel.cu")
    stage_cu = os.path.join(CUSADI_CODEGEN_DIR, "level_parallel_stage_kernel.cu")
    if not os.path.exists(lp_cu):
        write_kernel_source(lp_cu)
    if not os.path.exists(tiled_cu):
        write_tiled_kernel_source(tiled_cu)
    if not os.path.exists(stage_cu):
        write_stage_kernel_source(stage_cu)

    cmake_path = os.path.join(CUSADI_ROOT_DIR, "CMakeLists.txt")
    with open(cmake_path, "w") as f:
        f.write(textwrap.dedent("""\
            cmake_minimum_required(VERSION 3.15)
            project(CusADi LANGUAGES CXX CUDA)
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES 89)
            endif()
            set(CMAKE_CXX_STANDARD 11)
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 --use_fast_math)
            include_directories(${CMAKE_CURRENT_SOURCE_DIR})
            set(LP_SOURCE codegen/level_parallel_kernel.cu)
            add_library(level_parallel_kernel SHARED ${LP_SOURCE})
            set(LP_TILED_SOURCE codegen/level_parallel_tiled_kernel.cu)
            add_library(level_parallel_tiled_kernel SHARED ${LP_TILED_SOURCE})
            set(LP_STAGE_SOURCE codegen/level_parallel_stage_kernel.cu)
            add_library(level_parallel_stage_kernel SHARED ${LP_STAGE_SOURCE})
        """))

    result = subprocess.run(
        f"cd {CUSADI_BUILD_DIR} && cmake .. && make -j",
        shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LP kernel build failed:\n{result.stderr[-2000:]}")


class CusadiFunction2:
    """Unified CusADi GPU function with automatic kernel selection.

    Wraps three kernel paths behind one interface:
      - "lp":         original level-parallel (shared memory, no tiling)
      - "tiled":      tiled level-parallel (chunked shared memory + spill buffer)
      - "sequential": original CusADi (one thread per instance, global work vector)

    Default mode is chosen automatically based on DAG analysis:
      - If peak liveness fits in device shared memory → "lp"
      - Otherwise → "tiled"
      - "sequential" is never default (opt-in via mode= parameter)

    Args:
        fn_casadi: A loaded CasADi Function.
        compile_sequential: If True, also generate and compile the sequential
            kernel at construction time. Default False (sequential compilation
            can take minutes for large functions).
        smem_budget: Override shared memory budget in bytes. Default: device max
            minus 4 KB margin.
    """

    def __init__(
        self,
        fn_casadi: casadi.Function,
        compile_sequential: bool = False,
        smem_budget: Optional[int] = None,
    ):
        assert torch.cuda.is_available()
        self.fn_casadi = fn_casadi
        self.fn_name = fn_casadi.name()
        self._device = "cuda"

        # --- DAG analysis ---
        device_limit = _get_device_smem_limit()
        budget = smem_budget if smem_budget else device_limit - SMEM_MARGIN_BYTES
        max_smem_slots = budget // 8

        self._plan = extract_level_plan(fn_casadi)
        fits = self._plan.shared_mem_bytes <= budget

        if fits:
            self._tiled_plan = None
            self._default_mode = "lp"
        else:
            self._tiled_plan = extract_tiled_level_plan(fn_casadi, max_smem_slots)
            self._default_mode = "tiled"

        self._block_size = pick_block_size(self._plan)

        # --- Build DAGInfo ---
        tp = self._tiled_plan
        self.dag_info = DAGInfo(
            function_name=fn_casadi.name(),
            n_instructions=fn_casadi.n_instructions(),
            n_in=fn_casadi.n_in(),
            n_out=fn_casadi.n_out(),
            nnz_in=[fn_casadi.nnz_in(i) for i in range(fn_casadi.n_in())],
            nnz_out=[fn_casadi.nnz_out(i) for i in range(fn_casadi.n_out())],
            compute_ops=self._plan.total_compute_ops,
            critical_path=self._plan.critical_path,
            max_width=self._plan.max_width,
            median_width=self._plan.median_width,
            peak_live_slots=self._plan.peak_live_slots,
            shared_mem_bytes=self._plan.shared_mem_bytes,
            device_smem_limit=device_limit,
            fits_in_smem=fits,
            tiled_n_chunks=tp.n_chunks if tp else 1,
            tiled_peak_slots=tp.peak_live_slots if tp else self._plan.peak_live_slots,
            tiled_shared_mem_bytes=tp.shared_mem_bytes if tp else self._plan.shared_mem_bytes,
            tiled_spill_stride=tp.spill_buffer_stride if tp else 0,
            tiled_sub_batch_spills=len(tp.sub_batch_spills) if tp else 0,
            block_size=self._block_size,
            default_mode=self._default_mode,
        )

        # --- Compile kernels ---
        _ensure_lp_kernel()
        if self._tiled_plan is not None:
            _ensure_tiled_kernel()

        self._seq_compiled = False
        if compile_sequential:
            self._compile_sequential()

        # --- Kernel instance cache: {(mode, batch_size): wrapper} ---
        self._cache: Dict[tuple, object] = {}

        # --- COO data (shared, batch-independent) ---
        self._coo_rows = []
        self._coo_cols = []
        for i in range(fn_casadi.n_out()):
            rows, cols = fn_casadi.sparsity_out(i).get_triplet()
            self._coo_rows.append(
                torch.tensor(rows, device=self._device, dtype=torch.int64))
            self._coo_cols.append(
                torch.tensor(cols, device=self._device, dtype=torch.int64))

        # Public: set after evaluate()
        self.outputs_sparse: List[torch.Tensor] = []
        self.eval_time: float = 0.0

    def _compile_sequential(self):
        """Generate and compile the sequential CUDA kernel."""
        cu_path = os.path.join(CUSADI_CODEGEN_DIR, f"{self.fn_name}.cu")
        if not os.path.exists(cu_path):
            generateCUDACodeDouble(self.fn_casadi)

        so_path = os.path.join(CUSADI_BUILD_DIR, f"lib{self.fn_name}.so")
        if os.path.exists(so_path):
            self._seq_compiled = True
            return

        # Add sequential lib to CMakeLists alongside the LP kernels
        cmake_path = os.path.join(CUSADI_ROOT_DIR, "CMakeLists.txt")
        with open(cmake_path, "r") as f:
            content = f.read()
        if self.fn_name not in content:
            var = self.fn_name.upper() + "_SOURCE"
            content += (
                f"set({var} codegen/{self.fn_name}.cu)\n"
                f"add_library({self.fn_name} SHARED ${{{var}}})\n"
            )
            with open(cmake_path, "w") as f:
                f.write(content)

        os.makedirs(CUSADI_BUILD_DIR, exist_ok=True)
        result = subprocess.run(
            f"cd {CUSADI_BUILD_DIR} && cmake .. && make -j",
            shell=True, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(
                f"Sequential kernel build failed for {self.fn_name}:\n"
                f"{result.stderr[-2000:]}")
        self._seq_compiled = True

    def _get_kernel(self, mode: str, batch_size: int):
        """Get or create the kernel wrapper for (mode, batch_size)."""
        key = (mode, batch_size)
        if key in self._cache:
            return self._cache[key]

        if mode == "lp":
            if not self.dag_info.fits_in_smem:
                raise RuntimeError(
                    f"Cannot use mode='lp' for {self.fn_name}: "
                    f"peak shared memory {self.dag_info.shared_mem_bytes/1024:.0f}KB "
                    f"exceeds device limit {self.dag_info.device_smem_limit/1024:.0f}KB. "
                    f"Use mode='tiled' or mode='auto'."
                )
            wrapper = CusadiLevelParallelFunction(
                self.fn_casadi, batch_size,
                lib_name="level_parallel_kernel")

        elif mode == "tiled":
            plan = self._tiled_plan
            if plan is None:
                # Function fits in smem but user requested tiled anyway.
                # Build a tiled plan (will be 1 chunk, 0 spills).
                plan = extract_tiled_level_plan(
                    self.fn_casadi,
                    max_smem_slots=self._plan.peak_live_slots)
            wrapper = CusadiTiledLevelParallelFunction(
                self.fn_casadi, batch_size, tiled_plan=plan,
                lib_name="level_parallel_tiled_kernel")

        elif mode == "sequential":
            if not self._seq_compiled:
                raise RuntimeError(
                    f"Sequential kernel not compiled for {self.fn_name}. "
                    f"Construct with compile_sequential=True."
                )
            wrapper = CusadiFunction(self.fn_casadi, batch_size)

        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Use 'auto', 'lp', 'tiled', or 'sequential'.")

        self._cache[key] = wrapper
        return wrapper

    def evaluate(
        self,
        inputs: List[torch.Tensor],
        mode: str = "auto",
    ):
        """Evaluate the function on batched GPU inputs.

        Args:
            inputs: List of input tensors, each shape (batch_size, nnz_in[i]).
            mode: Kernel selection — "auto", "lp", "tiled", or "sequential".
        """
        batch_size = inputs[0].shape[0]

        if mode == "auto":
            mode = self._default_mode

        wrapper = self._get_kernel(mode, batch_size)
        wrapper.evaluate(inputs)

        self.eval_time = wrapper.eval_time
        self.outputs_sparse = wrapper.outputs_sparse
        if hasattr(wrapper, '_output_tensors'):
            self.outputs_sparse = wrapper._output_tensors

    def get_coo(self, out_idx: int = 0):
        """Return (rows, cols, values) for the given output."""
        return (self._coo_rows[out_idx],
                self._coo_cols[out_idx],
                self.outputs_sparse[out_idx])
