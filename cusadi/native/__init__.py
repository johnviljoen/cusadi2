"""cusadi native backend — host-free C++/CUDA runtime for the level-parallel and
stage kernels. No torch, no jax, no Python at deployment.

Offline (build-time):
    from cusadi.native import generate_native_plan
    generate_native_plan(fn, "myfn_plan.h")                    # 1-axis (batch)
    generate_native_plan(fn, "myfn_plan.h", input_specs, ...)  # 2-axis (stage)

Runtime (pure C++/CUDA): include cusadi/native/include/cusadi_native.h plus the
generated header, link a kernel from cusadi/lp_kernels/. See cusadi/native/example.
"""
import os

from cusadi.native.generate import generate_native_plan, generate_plan_registry

NATIVE_INCLUDE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "include")

__all__ = ["generate_native_plan", "generate_plan_registry", "NATIVE_INCLUDE_DIR"]
