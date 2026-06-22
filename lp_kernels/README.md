# `lp_kernels/` — materialized level-parallel kernels

These `.cu` files are **build artifacts, not source**. They are written here
verbatim from the hand-authored CUDA templates in
[`src/level_parallel/codegen.py`](../src/level_parallel/codegen.py)
(`KERNEL_TEMPLATE`, `TILED_KERNEL_TEMPLATE`, `STAGE_KERNEL_TEMPLATE`,
`CHUNK_KERNEL_TEMPLATE`, `STAGE_TILED_KERNEL_TEMPLATE`) by the `write_*_kernel_source`
functions, then compiled into `build/liblevel_parallel_*.so`.

- **Source of truth:** `src/level_parallel/codegen.py`. Edit the templates there,
  never the `.cu` here — they are overwritten on every build and are gitignored.
- **Why a separate directory:** `codegen/` holds CusADi's *machine-generated,
  per-function* unrolled `.cu` (one giant file per CasADi `Function`). The five
  kernels here are *function-independent interpreters* authored by hand and
  compiled once. Keeping them apart prevents the two kinds of `.cu` — authored
  vs. generated — from being confused for one another.

Safe to delete; regenerated automatically on the next build via
`src/level_parallel/unified.py`.
