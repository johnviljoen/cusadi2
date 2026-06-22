"""PyTorch evaluation backend.

All torch-specific code lives here, mirroring cusadi/jax_ffi/ as the JAX backend:

  * cusadi_function.py        — upstream per-function CusadiFunction
  * generate_pytorch_code.py  — PyTorch codegen
  * pattern.py                — sparsity-pattern helper
  * lp_function.py            — level-parallel torch binding
  * lp_stage.py               — staged / tiled level-parallel torch bindings
  * lp_unified.py             — auto-dispatch over the LP + per-function paths
  * validation.py             — GPU-vs-CPU(CasADi oracle) validation harness

The backend-agnostic level-parallel engine (cusadi/level_parallel/: dag.py,
codegen.py) imports neither torch nor jax. The shared CUDA codegen
(generateCUDACode, CusadiOperations) stays at the package root, since both
backends depend on it.

Top-level symbols (CusadiFunction, generatePytorchCode) are re-exported by
cusadi/__init__.py; importing this subpackage directly is also supported, e.g.:
    from cusadi.pytorch.lp_function import CusadiLevelParallelFunction
"""
