# cusadi2

- We add another axes of parallelism to the original cusadi in the form of level-parallel parallelism
- We also added stage-parallel execution - but this seems to have been done in a way in the 2D formulation in cusadi proper - but we do it in a way that we only compile a single stage - dramatically decreasing compilation times for structured problems

# How does it work

Read files in this order:
1. cusadi_operations.py
2. kernel_template.py
3. dag.py
4. generate.py
5. cusadi2.h


# The `cusadi/` package (merged from johnviljoen/cusadi_dev)

The earlier fork of se-hwan/cusadi lives here too, history preserved (merged
2026-08-25 so there is one cusadi codebase). It is what cuipm_mk2 builds against:

- `cusadi/level_parallel/` — level-parallel DAG codegen; `codegen.py` holds the
  hand-authored CUDA kernel templates (source of truth for `cusadi/lp_kernels/*.cu`,
  which are generated and gitignored: `python -c "from cusadi.level_parallel.codegen import
  write_kernel_source, write_stage_kernel_source; write_kernel_source('cusadi/lp_kernels/level_parallel_kernel.cu');
  write_stage_kernel_source('cusadi/lp_kernels/level_parallel_stage_kernel.cu')"`).
- `cusadi/native/` — plan serialization to C headers + the header-only runtime
  `include/cusadi_native.h` (`StageProgram::evaluate_async`: a pure launch with the
  argument tables passed by value). See `cusadi/native/README.md`.
- `cusadi/pytorch/`, `cusadi/jax_ffi/` — the torch / JAX bindings; `experiments/`,
  `ALGORITHM.md`, `HANDOFF.md` — the fork's benchmarks and notes.
