"""cusadi2: level- and stage-parallel GPU evaluation of batched CasADi functions.

Build time (Python):
    from cusadi2 import compile
    plan = compile(fn, cache_dir=..., max_live="device", block_size=None)   # cached plan file
    write_plan(fn, "fn.plan", n_stages=...)   # or write one directly, uncached
    #   max_live=device_max_live(dtype): window the levels to fit shared memory
    write_unrolled_source(fn, "fn_unrolled.cu")   # optional: original-cusadi-style
                                                  # unrolled kernel, for comparison

Run time (C++/CUDA): include cusadi2.h (`get_include()`) and link the generic kernel
(`get_library("double")`, or `find_package(cusadi2 CONFIG HINTS <cmake_dir()>)` and the target
cusadi2::cusadi2_f64), load the plan, evaluate. No Python and no CUDA compilation at deployment.

Not part of this stable interface: `cusadi2.contact` / `cusadi2_contact.h` (a MuJoCo step with a
per-model contact kernel the user compiles).
"""
import os

from cusadi2.cache import CompiledPlan, compile, resolve_max_live
from cusadi2.compose import ComposedFunction, StageLayout, row_numbers, rows_active, sparsity_coo
from cusadi2 import staging  # noqa: F401
from cusadi2.derivatives import gradient, hessian, jacobian, vector_hessian, weighted_hessian
from cusadi2.dag import (LevelPlan, SerialProgram, extract_level_plan, extract_serial_program,
                         verify_plan, verify_serial_program)
from cusadi2.kernel_template import write_kernel_source
from cusadi2.order import OUTPUT_ORDERS
from cusadi2.plan import cse_function, device_max_live, plan_bytes, read_plan, write_plan, write_reference
from cusadi2.unrolled import write_unrolled_source

INCLUDE_DIR = os.path.dirname(os.path.realpath(__file__))   # holds cusadi2.h (source checkout)
HEADER = os.path.join(INCLUDE_DIR, "cusadi2.h")


def _native_paths():
    import importlib
    try:
        return importlib.import_module("cusadi2._native")
    except ImportError as e:
        raise RuntimeError("cusadi2: the native library was not built (no GPU toolchain at install time)") from e


def get_include() -> str:
    """Directory holding cusadi2.h, next to the built library."""
    return _native_paths().INCLUDE_DIR


def get_library(dtype: str = "double") -> str:
    """Path of libcusadi2_f64.so ("double") or libcusadi2_f32.so ("float")."""
    sfx = {"double": "f64", "float": "f32"}[dtype]
    path = os.path.join(_native_paths().LIB_DIR, f"libcusadi2_{sfx}.so")
    if not os.path.exists(path):
        raise RuntimeError(f"cusadi2: {path} missing")
    return path


def cmake_dir() -> str:
    """Directory of cusadi2Config.cmake (find_package(cusadi2 CONFIG HINTS ...))."""
    return _native_paths().CMAKE_DIR


__all__ = ["OUTPUT_ORDERS", "CompiledPlan", "compile", "resolve_max_live", "ComposedFunction", "StageLayout", "rows_active", "row_numbers", "sparsity_coo",
           "LevelPlan", "SerialProgram", "extract_level_plan", "extract_serial_program",
           "verify_plan", "verify_serial_program", "write_kernel_source",
           "cse_function", "gradient", "jacobian", "hessian", "weighted_hessian", "vector_hessian", "device_max_live", "plan_bytes", "read_plan", "write_plan", "write_reference", "write_unrolled_source",
           "get_include", "get_library", "cmake_dir", "INCLUDE_DIR", "HEADER"]
