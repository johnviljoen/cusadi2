# cusadi2

Batched evaluation of CasADi functions on the GPU with one function-independent
CUDA kernel. Two parallel axes on top of the batch: **level-parallel** (the
expression DAG runs one topological level at a time, all nodes in a level in
parallel within a block) and **stage-parallel** (a per-stage function evaluated
over a horizon, one block per (instance, stage)). See `ALGORITHM.md`.

Python is used only at build time. Deployment is C++/CUDA: a header-only
runtime plus a binary plan file per function.

## Install

uv is the environment and build frontend:

```bash
uv sync                      # cusadi2 (casadi, numpy) + the JAX extension if the toolchain resolves
uv sync --extra jax --extra test   # + jax[cuda13] runtime, pytest
```

The only compiled part is the JAX FFI extension (`cusadi2/jax/_ffi`); its
toolchain (nvcc, cudart, XLA headers) comes entirely from pip wheels listed in
`build-system.requires`, so no system CUDA is needed. The build backend is
scikit-build-core (uv's own backend cannot compile extensions); the kernel
source is generated from the op tables at build time.

## Build time (Python)

```python
import casadi as ca
from cusadi2 import write_kernel_source, write_plan

write_kernel_source("kernel.cu")                 # once per project; dtype="float" for f32
write_plan(fn, "fn.plan")                        # batch-only (n_stages == 1)
write_plan(fn_stage, "fn_stage.plan", n_stages=30,
           in_mode=["coupled", "shared"], in_offset=[0, 0], in_span=[1, 0])
write_unrolled_source(fn, "fn_unrolled.cu")     # optional: original-cusadi-style per-function kernel
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
prog.use_serial(scratch_bytes);                // optional: one thread per (instance, stage)
```

`d_in` / `d_out` are device-resident arrays of device buffer pointers, one per
argument; buffer `a` holds `plan.in[a].len(batch)` elements laid out
(instance, stage_count, nnz). `plan.in[a].sp` / `plan.out[a].sp` carry the CasADi
CCS sparsity (`coo_row()` / `coo_col()` for COO). Build with
`-DCUSADI2_SCALAR=float` to match a float kernel. `use_serial` swaps the
level-parallel kernel for a thread-per-instance one that walks CasADi's own
instruction stream (also in the plan; work vector of `sz_w` slots in a global
scratch buffer sized by the budget; threads loop when the batch is larger),
for functions whose levels are too narrow to fill a block; `thread_major=true`
reproduces the original cusadi's strided work-vector layout. `write_unrolled_source`
emits the original cusadi's unrolled per-function kernel from the same stream
(symbols `<name>_unrolled_launch` / `_sz_w` / `_n_stages`), for comparison; it
costs a per-function nvcc compile.

## JAX front end (`cusadi2.jax`)

Plain JAX code becomes a cusadi2 plan with exact-sparsity derivatives formed
symbolically in CasADi, callable under `jit` and `vmap`:

```python
import jax, jax.numpy as jnp
from cusadi2.jax import jax_function
jax.config.update("jax_enable_x64", True)

fn  = jax_function(c, param_argnums=(1,))   # c(z, p): p is a parameter, never differentiated
y   = fn(z, p)                              # dense output, same as c(z, p); jit / vmap as usual
jac = fn.jacobian()                         # one plan: jac(z, p) -> (nnz,) values, (B, nnz) under vmap
coo = jac.coo(z, p)                         # (nnz, 2) int32 [row, col], fixed per argument signature
hes = fn.hessian()                          # true Hessian of each vector output: coo (nnz, 3) = (k, i, j)
whs = fn.hessian(weighted=True)             # Lagrangian Hessian: whs(z, p, lam=lam) -> (nnz,), coo (nnz, 2)
```

How it works: `jax.make_jaxpr` is interpreted with one CasADi SX scalar per
array element (`cusadi2/jax/interpret.py`), which keeps the exact sparsity of
vectorized code (masks, `where`, dense matmuls with sparse constants, `vmap`,
`scan`) because CasADi folds constant zeros at construction. Derivatives are
`ca.jacobian` / `ca.jacobian(ca.jacobian(.))` / `ca.hessian(lam.y)` on that
graph, each its own plan (never fused with the value). Shapes come from the
arguments at first call and every (variant, signature) is planned once.
Arguments not in `param_argnums` are the variables, flattened row-major and
concatenated in order. Boundaries: static-trip-count `scan` / `fori_loop`
unroll, `cond` becomes `if_else` (union sparsity), a dynamic `while_loop` or
a `gather` with traced indices or a `sort` of traced values raises
`cusadi2.jax.Unsupported`. Any primitive whose inputs are all constants (index
bookkeeping: `argsort`, `nonzero`, masks) is evaluated by JAX itself at trace
time, so only the symbolic path needs SX rules; `isfinite` of a symbol is
taken as true. The dtype follows the JAX arrays: float32 when `jax_enable_x64` is off or the inputs are float32, float64 otherwise (`dtype=` overrides); each dtype is its own kernel build (`_ffi_f64`, `_ffi_f32`).

Runtime: `cusadi2/jax/ffi.cu` is an XLA FFI handler (spineax pattern: plans
live in a registry keyed by id, each custom call holds a state pointing at its
entry) that launches `Program::evaluate_ptrs`, the kernel with the buffer
pointers passed by value, on XLA's stream. `vmap` folds the mapped axis into
the plan's instance axis.

### Rigid-body dynamics from a model file (`cusadi2.jax.rbd`)

```python
from cusadi2.jax import rbd
model = rbd.load("robot.xml")             # MJCF or URDF, parsed by MuJoCo
tau = model.inverse_dynamics(q, qd, qdd)  # == mj_inverse (constraints and actuation disabled)
qdd = model.forward_dynamics(q, qd, tau)  # == mj_forward (smooth dynamics)
xpos, xmat = model.forward_kinematics(q)  # body frames like MjData.xpos / xmat
q1 = model.integrate_position(q, qd, dt)  # == mj_integratePos
```

Featherstone's recursive Newton-Euler and articulated-body algorithms in body
coordinates as plain JAX, with MuJoCo's conventions (qpos / qvel layouts,
free-joint linear velocity in the world frame, `ref`, `springref`): free, ball,
slide and hinge joints, several joints per body, jointless bodies, armature,
damping, joint springs, gravity compensation (tendons and fluid forces raise).
Verified against MuJoCo to round-off (`tests/test_rbd.py`). Body-frame
recursions are the formulation of Pinocchio and give the smallest symbolic
graphs; the alternatives were measured on the G1 inverse dynamics (q, qd, qdd
in, 35 torques out; the Lagrangian Hessian is what an interior-point method
consumes):

| backend | Jacobian nnz structural / numerical (of 3710) | Lagrangian Hessian nnz (of 11236) | Lagrangian Hessian instructions |
|---|---|---|---|
| `cusadi2.jax.rbd` | 1871 / 1756 | 3466 | 0.27 M |
| jaxsim (URDF) | 1976 / 1750 | 4093 | 0.95 M |
| MuJoCo MJX `mjx.inverse` | 2545 / 1756 | 6974 | 1.82 M |
| brax generalized | 2545 / 1756 | 6974 | 2.03 M |

MuJoCo (and brax, its port) express spatial quantities at the subtree centre
of mass, which depends on every joint angle; the dependence cancels in the
torques but symbolic sparsity cannot see cancellation. Forward dynamics are
dense through the inverse mass matrix for every formulation, so a sparse
transcription imposes the inverse dynamics with the accelerations as decision
variables, contact forces entering linearly.

Examples (`uv sync --extra jax --extra examples`):

- `examples/jax/g1_dynamics.py`: the Unitree G1 through `rbd`: value, Jacobian
  and Lagrangian Hessian plans of the inverse dynamics (`--true-hessian` adds
  the full 35 x 106 x 106 tensor), value and Jacobian of the forward dynamics
  (`--fd-hessian`), each checked against MuJoCo and JAX autodiff and timed at
  batch 1024.
- `examples/jax/g1_mjx.py`: the same inverse dynamics through `mjx.inverse`,
  the comparison baseline (structural and numerical nnz are both printed).

```bash
uv run examples/jax/g1_dynamics.py            # --jax-only checks the JAX dynamics vs MuJoCo without cusadi2
uv run examples/jax/g1_mjx.py                 # needs mujoco-mjx
```

Both examples set `XLA_PYTHON_CLIENT_PREALLOCATE=false` before importing JAX:
cusadi2 plans are allocated outside XLA's pool, so the pool must not grow over
the whole device.

### Handing an NLP to cuipm (`cusadi2.jax.ipm`)

```python
from cusadi2.jax.ipm import export_cuipm, solve_ipopt
info = export_cuipm(f, c, d, x0, p0, "plans/", x_L=x_L, x_U=x_U)   # f(x, p) scalar, c(x, p) = 0, d(x, p) >= 0
ref  = solve_ipopt(f, c, d, x0, p0, x_L=x_L, x_U=x_U, options={"max_iter": 100})   # the same NLP in IPOPT
```

`export_cuipm` traces f, c and d into SX on shared symbols and writes the nine
plans cuipm loads (`f c d jac_f jac_c jac_d hess_f hess_c hess_d`, the input
signatures of `cuipm/codegen/generate.py`, `hess_c` / `hess_d` the Lagrangian
Hessians with the multipliers as inputs) plus `problem.json` with the dims,
nnz, bounds and the initial point. Patterns are exact: the `eps` input that
the hand-written codegen used to inflate the cost pattern is accepted and
ignored. On the quadcopter NMPC the exported plans reproduce cuipm's jaxipm
gold trajectory bit for bit (`tests/test_cuipm_export.py` checks values and
nnz against the hand-written CasADi NLP and against cuipm's own plans).
`solve_ipopt` is the reference solve of the identical NLP through CasADi's
`nlpsol`. `examples/jax/go2_jump_nlp.py` is a whole-body example: the Go2 jump
in the lifted inverse-dynamics transcription on `rbd`, exported, solved with
IPOPT and timed through `jax_function` on the GPU (JSON-configured, writes
`runs/go2_jump_nlp/<name>/`).

Plans whose level windows do not fit the device's shared memory fall back to
the serial (thread-per-instance) kernel automatically, with a warning;
`jax_function(..., serial=True)` forces it, `max_live=` sets the window budget.

## Tests

```bash
uv run pytest             # host: plan verification, file round-trip, jaxpr->SX interpreter;
                          # GPU: jax_function + quadcopter NMPC acceptance test
NVCC=.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc uv run pytest tests/test_gpu.py   # kernel self-check
```

## Layout

- `cusadi2/dag.py` — CasADi instructions to level-decomposed SSA plan and verbatim serial program; host verifiers
- `cusadi2/ops.py` — the op table (CUDA double/float forms, Python form)
- `cusadi2/kernel_template.py` — the kernel source
- `cusadi2/plan.py` — plan / reference file writers (format documented there)
- `cusadi2/jax/rbd.py` — rigid-body dynamics (RNEA, ABA, kinematics) in JAX from a MuJoCo-parsed MJCF / URDF
- `cusadi2/jax/ipm.py` — export of a JAX NLP to cuipm's nine plans; IPOPT reference solve
- `cusadi2/unrolled.py` — original-cusadi-style unrolled kernel emitter (baseline)
- `cusadi2/cusadi2.h` — loader and `Program` runtime
- `cusadi2/jax/` — JAX front end: `interpret.py` (jaxpr -> SX), `function.py` (`jax_function`, derivative plans), `primitive.py` (JAX primitive, vmap rule), `ffi.cu` (XLA FFI handler + plan registry)
- `examples/paper_results/` — A1/H1 kinematics, Jacobian and Hessian latency benchmark (`bench.py`, `latency.cu`; results committed), and the CusADi paper's KKT solves (`bench_orig.py`)
- `examples/menagerie_bench/` — the same benchmark once per MuJoCo Menagerie robot (`run.py`; needs the Menagerie checkout at `examples/mujoco_menagerie/`)
- `examples/humanoid_mit/` — MIT humanoid dynamics functions, and in `mpc/` a staged MPC QP solve with cuDSS in one CUDA graph
- `examples/go2_mppi/` — Unitree Go2 parkour under annealed MPPI: full-body penalty-contact rollouts as one cusadi2 function, sampling and plant in the same CUDA driver (`mppi_rollout.cu`)
- `examples/frax_bench/` — latency head-to-head against frax (JAX) on Franka Panda and Unitree G1: RNEA, CRBA and the Jacobian and Hessian of RNEA (`bench.py`, results in `results/`)
