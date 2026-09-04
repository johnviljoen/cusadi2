"""cusadi2: level- and stage-parallel GPU evaluation of batched CasADi functions.

Build time (Python):
    from cusadi2 import write_kernel_source, write_plan
    write_kernel_source("kernel.cu")          # once; function-independent
    write_plan(fn, "fn.plan", n_stages=...)   # per function and horizon

Run time (C++/CUDA): include cusadi2.h (``cusadi2.INCLUDE_DIR``), compile
kernel.cu into the consumer, load the plan, evaluate. No Python at deployment.
"""
import os

from cusadi2.dag import LevelPlan, extract_level_plan, verify_plan
from cusadi2.kernel_template import write_kernel_source
from cusadi2.plan import read_plan, write_plan, write_reference

INCLUDE_DIR = os.path.dirname(os.path.realpath(__file__))   # holds cusadi2.h
HEADER = os.path.join(INCLUDE_DIR, "cusadi2.h")

__all__ = ["LevelPlan", "extract_level_plan", "verify_plan", "write_kernel_source",
           "read_plan", "write_plan", "write_reference", "INCLUDE_DIR", "HEADER"]
