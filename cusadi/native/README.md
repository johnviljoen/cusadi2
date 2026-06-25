# cusadi native backend

Host-free C++/CUDA runtime for cusadi's function-independent level-parallel
kernels — **no torch, no jax, no Python at deployment**. Drop the LP / stage
kernels into a C++ MPC controller, ROS node, or embedded target.

cusadi's `level_parallel` and `stage` kernels are already function-independent
and `extern "C"`, but until now the only plan loaders were the torch
(`pytorch/`) and JAX (`jax_ffi/`) backends. This backend serializes the plan to a
plain C header so a pure C++/CUDA program can drive the same kernels.

## Offline (build-time, needs cusadi)

```python
from cusadi.native import generate_native_plan
generate_native_plan(fn, "myfn_plan.h")                              # 1-axis (batch)
generate_native_plan(fn, "myfn_plan.h", input_specs, output_specs)  # 2-axis (batch x horizon)
```

`input_specs`/`output_specs` are lists of `cusadi.level_parallel.stage.StageSpec`
(STAGE / SHARED / COUPLED) — one per arg. The header embeds a CasADi-evaluated
reference for runtime self-check.

## Runtime (pure C++/CUDA)

```cpp
#include "cusadi_native.h"      // cusadi/native/include
#include "myfn_plan.h"          // generated
cusadi::LPProgram prog;  prog.upload(MYFN_plan);
prog.evaluate(d_in, d_out, batch);                 // device pointer arrays

cusadi::StageProgram sprog;  sprog.upload(MYFN_plan);   // StagePlanHost
sprog.evaluate(d_in, d_out, batch, n_stages);      // grid = (batch, n_stages)
```

Link one kernel from `cusadi/lp_kernels/`: `level_parallel_kernel.cu` (LP) and/or
`level_parallel_stage_kernel.cu` (stage). Inputs/outputs are caller-owned device
buffers, batch-major (`instance * nnz + element`); for the stage kernel each arg
is laid out `(instance, stage_count, nnz)` per its mode.

## Example

`example/` is a complete, self-contained program (no robot model) that emits
plans for a 1-axis and a 2-axis function and self-checks both:

```bash
cd example
python gen_example.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/native_example     # -> all checks PASSED
```

(Set `-DEXAMPLE_CUDA_ARCH=<cc>` for a non-Ada GPU.)

## Layout

- `generate.py` — offline plan→C emitter (`generate_native_plan`).
- `include/cusadi_native.h` — header-only runtime (`LPProgram`, `StageProgram`).
- `../level_parallel/serialize.py` — backend-agnostic plan flattener (shared
  source of truth; torch/jax backends should migrate to it).
