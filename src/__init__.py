import os, sys

CUSADI_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CUSADI_CLASS_DIR = os.path.join(CUSADI_ROOT_DIR, "src")
CUSADI_FUNCTION_DIR = os.path.join(CUSADI_ROOT_DIR, "src", "casadi_functions")
CUSADI_BENCHMARK_DIR = os.path.join(CUSADI_ROOT_DIR, "src", "benchmark_functions")
CUSADI_SCRIPT_DIR = os.path.join(CUSADI_ROOT_DIR, "scripts")
CUSADI_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build")
CUSADI_CODEGEN_DIR = os.path.join(CUSADI_ROOT_DIR, "codegen")
CUSADI_DATA_DIR = os.path.join(CUSADI_ROOT_DIR, "data")
CUSADI_FIGURES_DIR = os.path.join(CUSADI_ROOT_DIR, "figures")

sys.path.insert(0, CUSADI_BUILD_DIR)

# PyTorch-backed entry points. Guarded so the package is still importable when
# torch (or other PyTorch-only deps) is missing — needed for the JAX-FFI-only
# install path on branch feature/jax-ffi-backend.
try:
    from .CusadiFunction import CusadiFunction
    from .CusadiOperations import OP_CUDA_DICT, OP_PYTORCH_DICT, OP_CUDA_DICT_ORIG
    from .generateCUDACode import generateCUDACodeDouble, generateCUDACodeFloat, generateCMakeLists, generateCUDACodeV3, generateCUDACode2D
    from .generatePytorchCode import generatePytorchCode
    from .benchmark_functions import *
except ImportError as _e:
    import warnings as _warnings
    _warnings.warn(f"cusadi PyTorch-path symbols unavailable: {_e}", ImportWarning)