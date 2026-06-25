import os, sys

CUSADI_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CUSADI_CLASS_DIR = os.path.join(CUSADI_ROOT_DIR, "cusadi")
CUSADI_FUNCTION_DIR = os.path.join(CUSADI_ROOT_DIR, "build", "casadi_functions")
CUSADI_BENCHMARK_DIR = os.path.join(CUSADI_ROOT_DIR, "experiments", "benchmarks", "cpu_baselines")
CUSADI_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build")
# build/ holds only subdirectories — one per build tree:
CUSADI_LP_BUILD_DIR = os.path.join(CUSADI_BUILD_DIR, "lp")           # our liblevel_parallel_*.so
CUSADI_PYTORCH_BUILD_DIR = os.path.join(CUSADI_BUILD_DIR, "pytorch")  # original-cusadi per-function lib<fn>.so
CUSADI_CODEGEN_DIR = os.path.join(CUSADI_ROOT_DIR, "build", "codegen")
CUSADI_SEQ_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build", "sequential")
CUSADI_JAX_FFI_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build", "jax_ffi")
# Materialization target for the hand-authored, function-independent level-parallel
# kernel templates (source of truth: cusadi/level_parallel/codegen.py). Lives inside
# the package, kept separate from CUSADI_CODEGEN_DIR (codegen/), which holds CusADi's
# machine-generated per-function .cu.
CUSADI_LP_KERNEL_DIR = os.path.join(CUSADI_CLASS_DIR, "lp_kernels")
CUSADI_DATA_DIR = os.path.join(CUSADI_ROOT_DIR, "build", "data")
CUSADI_FIGURES_DIR = os.path.join(CUSADI_ROOT_DIR, "experiments", "figures")

sys.path.insert(0, CUSADI_BUILD_DIR)
# The JAX-FFI extension is now installed in-package as cusadi.jax_ffi._cusadi_jax_ffi
# (built by scikit-build-core), so no build/jax_ffi sys.path injection is needed.

# PyTorch-backed entry points. Guarded so the package is still importable when
# torch (or other PyTorch-only deps) is missing — needed for the JAX-FFI-only
# install path on branch feature/jax-ffi-backend.
try:
    from .pytorch.cusadi_function import CusadiFunction
    from .CusadiOperations import OP_CUDA_DICT, OP_PYTORCH_DICT, OP_CUDA_DICT_ORIG
    from .generateCUDACode import generateCUDACodeDouble, generateCUDACodeFloat, generateCMakeLists, generateCUDACodeV3, generateCUDACode2D
    from .pytorch.generate_pytorch_code import generatePytorchCode
except ImportError as _e:
    import warnings as _warnings
    _warnings.warn(f"cusadi PyTorch-path symbols unavailable: {_e}", ImportWarning)