# cusadi2

Batched evaluation of CasADi functions on the GPU with one function-independent
CUDA kernel. Two parallel axes on top of the batch: **level-parallel** (the
expression DAG runs one topological level at a time, all nodes in a level in
parallel within a block) and **stage-parallel** (a per-stage function evaluated
over a horizon, one block per (instance, stage)). See `ALGORITHM.md`.

Python is used only at build time. Deployment is C++/CUDA: a header-only
runtime plus a binary plan file per function.

## Install

```bash
pip install -e .          # casadi, numpy
```

## Build time (Python)

```python
import casadi as ca
from cusadi2 import write_kernel_source, write_plan

write_kernel_source("kernel.cu")                 # once per project; dtype="float" for f32
write_plan(fn, "fn.plan")                        # batch-only (n_stages == 1)
write_plan(fn_stage, "fn_stage.plan", n_stages=30,
           in_mode=["coupled", "shared"], in_offset=[0, 0], in_span=[1, 0])
```

Stage modes per input: `"stage"` (one value per stage), `"shared"` (broadcast),
`"coupled"` (stage k reads element k+offset of a buffer of length n_stages+span).
Outputs are always `"stage"`. The horizon is baked into the plan file.

## Run time (C++/CUDA)

Compile `kernel.cu` into your program with `cusadi2.h` on the include path
(`python -c "import cusadi2; print(cusadi2.INCLUDE_DIR)"`).

```cpp
#include "cusadi2.h"
cusadi2::Plan plan = cusadi2::load_plan("fn.plan");
cusadi2::Program prog(plan);
prog.evaluate(d_in, d_out, batch, stream);     // one launch, graph-capturable
```

`d_in` / `d_out` are device-resident arrays of device buffer pointers, one per
argument; buffer `a` holds `plan.in[a].len(batch)` elements laid out
(instance, stage_count, nnz). `plan.in[a].sp` / `plan.out[a].sp` carry the CasADi
CCS sparsity (`coo_row()` / `coo_col()` for COO). Build with
`-DCUSADI2_SCALAR=float` to match a float kernel.

## Tests

```bash
pytest                    # host: plan verification + file round-trip; GPU: nvcc self-check
```

## Layout

- `cusadi2/dag.py` — CasADi instructions to level-decomposed SSA plan; host verifier
- `cusadi2/ops.py` — the op table (CUDA double/float forms, Python form)
- `cusadi2/kernel_template.py` — the kernel source
- `cusadi2/plan.py` — plan / reference file writers (format documented there)
- `cusadi2/cusadi2.h` — loader and `Program` runtime
