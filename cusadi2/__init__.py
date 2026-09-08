"""cusadi2: level- and stage-parallel GPU evaluation of batched CasADi functions.

Build time (Python):
    from cusadi2 import write_kernel_source, write_plan
    write_kernel_source("kernel.cu")          # once; function-independent
    write_plan(fn, "fn.plan", n_stages=...)   # per function and horizon
    #   max_live=device_max_live(dtype): window the levels to fit shared memory
    write_unrolled_source(fn, "fn_unrolled.cu")   # optional: original-cusadi-style
                                                  # unrolled kernel, for comparison

Run time (C++/CUDA): include cusadi2.h (``cusadi2.INCLUDE_DIR``), compile
kernel.cu into the consumer, load the plan, evaluate. No Python at deployment.
"""
import os

from cusadi2.dag import (LevelPlan, SerialProgram, extract_level_plan, extract_serial_program,
                         verify_plan, verify_serial_program)
from cusadi2.kernel_template import write_kernel_source
from cusadi2.plan import device_max_live, read_plan, write_plan, write_reference
from cusadi2.unrolled import write_unrolled_source

INCLUDE_DIR = os.path.dirname(os.path.realpath(__file__))   # holds cusadi2.h
HEADER = os.path.join(INCLUDE_DIR, "cusadi2.h")

__all__ = ["LevelPlan", "SerialProgram", "extract_level_plan", "extract_serial_program",
           "verify_plan", "verify_serial_program", "write_kernel_source",
           "device_max_live", "read_plan", "write_plan", "write_reference", "write_unrolled_source",
           "INCLUDE_DIR", "HEADER"]
