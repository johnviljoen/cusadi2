"""cusadi2: level- and stage-parallel GPU evaluation of batched CasADi functions.

Submodules are imported eagerly so `import cusadi2 as cu` supports
`cu.kernel_template.write_kernel_source(...)` / `cu.generate.generate_native_plan(...)`.
"""

from . import cusadi_operations, dag, generate, kernel_template
from .generate import generate_native_plan, generate_reference
from .kernel_template import write_kernel_source

__all__ = ["cusadi_operations", "dag", "generate", "kernel_template",
           "generate_native_plan", "generate_reference", "write_kernel_source"]
