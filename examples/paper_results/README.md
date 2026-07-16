# cusadi2 paper results — latency vs batch size

Latency-against-batch-size saturation curves for cusadi2's three axes of
parallelism, replicating the level-parallel ablation from the old cusadi
codebase but entirely within cusadi2.

```bash
CUSADI2_GPU=3 python examples/paper_results/run.py
```

## What it measures

For **A1 and H1 forward kinematics and their frame Jacobians** (free-floating
base, no contacts) it sweeps batch size `B` at a fixed horizon `N` and times one
kernel launch for four configurations that toggle the stage and level-parallel
axes on/off, plus a **MuJoCo MJX baseline** for the same workload.

**Frame set.** Both backends evaluate the *identical* link set: the MuJoCo body
frames excluding `world` (a1: 13, h1: 20), looked up by name on the Pinocchio side
(every MuJoCo body name is also a Pinocchio BODY frame). Pinocchio's full `nframes`
additionally carries joint/fixed/sensor frames MuJoCo lacks; restricting to the
common bodies is what makes the cusadi2-vs-MJX comparison "the same thing" rather
than an artefact of differing frame counts. FK returns each link's world placement
(translation + flattened rotation); the Jacobian returns each link's 6×nv
world-aligned (LOCAL_WORLD_ALIGNED) geometric Jacobian.

| config   | axes                               | plan       | grid     | block_size        |
|----------|------------------------------------|------------|----------|-------------------|
| `full`   | staged + level-parallel + batch    | per-stage  | (B, N)   | `pick_block_size` |
| `staged` | staged + batch (no LP)             | per-stage  | (B, N)   | 1                 |
| `lp`     | level-parallel + batch (no stages) | unrolled   | (B, 1)   | `pick_block_size` |
| `batch`  | batch only (≈ original cusadi)     | unrolled   | (B, 1)   | 1                 |

All four do the **same total work** (`B·N` evaluations) on identically-sized
total buffers; what differs is how that work is packed into the grid and shared
memory:

- **`full` / `staged`** use the *per-stage* plan and spread the horizon over the
  grid's stage axis → `B·N` small blocks, each evaluating one stage out of shared
  memory sized for one stage.
- **`lp` / `batch`** use the *unrolled* plan — the per-stage function stamped `N`
  times into one DAG — and run **one block per batch instance** → only `B` blocks,
  each evaluating the whole horizon out of shared memory sized for `N` stages.
  This is exactly the original cusadi approach (no stage axis ⇒ unroll the horizon
  into one function). `batch` with `block_size=1` is the original sequential kernel.

The level-parallel axis is disabled by forcing `block_size=1` (one thread runs
every level serially). The x-axis is `B`.

This is where the **stage axis** pays off for horizon `N > 1`: staging keeps blocks
small and numerous (`B·N`), so the GPU saturates at an `N×` lower batch and the
per-block shared-memory footprint stays `N×` smaller, while the unrolled baseline
has few, huge blocks. The `full` line should sit below `lp`, and `staged` below
`batch`, until everything becomes throughput-bound at large `B`.

> **Horizon and shared memory.** The unrolled plan's shared memory grows `~N×` the
> per-stage peak, and the device per-block limit is ~99 KB (L40S). `N` is kept at
> **8** — the largest horizon whose unrolled plans all fit (cusadi2 has no tiling
> scheduler yet). At `N ≥ 16` the unrolled jacobians overflow and `lp`/`batch`
> become infeasible (reported as `nan`) — which is itself the point of staging.

### MJX baseline

The same FK / frame-Jacobian workload is run in MuJoCo MJX (`jax.vmap` + `jit`,
contacts off via `geom_contype = geom_conaffinity = 0`) and overlaid on each panel,
precision-matched (fp64 figure ← MJX fp64, fp32 ← MJX fp32). MJX uses MuJoCo's
*analytical* kinematic Jacobian (`support.jac`), the fair match to Pinocchio's
`getFrameJacobian` — **not** autodiff. (The old cusadi paper's 200–800× gap was a
*different* workload: analytical ABA *dynamics* derivatives vs `jacfwd(mjx.forward)`
autodiff-through-the-solver. Kinematics is cheap for MJX, so the honest gap here is
smaller.) MJX is timed up to `B = 1024` (its per-batch JIT compile is too slow
above that), in a **separate process per dtype** (`python run.py --mjx --robot R
--dtype fpXX`) since the jax x64 toggle must precede any `jnp` op.

MJX and mujoco-warp run the **full** batch sweep (up to 16384); cases that exceed
GPU memory are recorded as `null` and simply leave a gap in the curve. Precision
split is strict: the **fp64 figure excludes mujoco-warp** (fp32-only, not
comparable to fp64 cusadi2); the fp32 figure includes it on the FK panels.

Two MJX curves per panel, **both doing `B·N` total evaluations** at plotted `B`
(matching cusadi2's stage axis), so the horizon is handled two ways:

- **MJX batched (B·N)** — `vmap` over `B·N` independent per-knot configs, one jit
  launch: the horizon evaluated fully in parallel. This is the apples-to-apples
  match to cusadi2's `full`/`lp` (same operation, same work).
- **MJX rollout (N steps)** — `vmap` over `B` instances, `lax.scan` over `N`
  sequential `mjx.step`s, computing the quantity at each knot: the realistic
  "simulate a length-N trajectory and read kinematics off it" usage, sequential in
  time (≈ N launches' dispatch). **Caveat:** the rollout *also steps the dynamics*
  (state must evolve), so it does strictly more work than cusadi2's pure kinematics
  — it is the trajectory-simulation cost, not the identical operation. Read it as
  the practical upper bound; the batched curve is the like-for-like one.

Timing is wall-clock around `block_until_ready` (median of `MJX_N_EVALS`), matching
the old cusadi MJX benchmark.

### Timing symmetry

The latency figures plot **end-to-end wall-clock on every curve**: cusadi2's
`wall_us` is a host timer around the whole `evaluate()` call (count upload +
launch dispatch + kernel + device sync), symmetric with the
`block_until_ready` / `synchronize_device` timing of the MJX and warp baselines.
The `cudaEvent` kernel-only time (`latency_us`) is used only in the throughput
figures, where the MJX side is likewise overhead-stripped (repeat-subtraction
marginal GPU time).

One asymmetry remains in the C++ sweep: its wall-clock runs in a compiled binary,
while MJX/warp pay Python interpreter dispatch. So the sweep is run a **second
time from Python**: `latency.cu` is rebuilt as a shared library
(`-DCUSADI2_BENCH_LIB`, drops `main`), loaded via ctypes, and each `evaluate()`
is timed with `time.perf_counter()` in a subprocess (`run.py --pylib`) — the
wall-clock a real Python deployment would see, including FFI dispatch. These
appear as same-color dashed ×-marked twins on the latency panels; the gap to the
solid curves is the Python-hosting cost.

### mujoco-warp baseline (FK panels only)

mujoco-warp (`mjw.kinematics`, batched via `nworld = B·N`) is also overlaid on the
**forward-kinematics panels of the fp32 figure only**. It is **fp32 only and has no
autodiff**, so it appears on neither the Jacobian panels nor the fp64 figure. Timed
wall-clock around `wp.synchronize_device` (`python run.py --warp --robot R`).

## Compile / codegen time (the other 8 panels)

`compile_<dtype>.png` — same 2×2 layout per precision — plots **compile time vs
batch** for each backend, the metric that dominates deploy/iteration cost:

- **cusadi2** — per-function codegen (`extract_level_plan` + emit header) plus the
  one-time, function-independent `nvcc` kernel compile (shared across every function
  *and* every batch). Batch-independent ⇒ a flat line.
- **MJX** — first-call JIT time (trace + XLA compile + one run). XLA re-traces per
  batch **shape**, so it rises with batch — the well-known MJX iteration tax.
- **mujoco-warp** — Warp kernels are shape-generic, compiled once (cold cache via
  `wp.clear_kernel_cache()`), so also a flat line (FK/fp32 only).

The headline: cusadi2 and warp compile once and reuse; MJX pays a fresh, growing
JIT for every batch size.

## Knobs (top of `run.py`)

- `N_STAGES = 8` — horizon (stage axis length); the largest that keeps all
  unrolled plans within the device shared-memory limit
- `BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]` — x-axis sweep
- `MJX_BATCHES` — `BATCHES` capped at 1024; `MJX_N_WARMUP`, `MJX_N_EVALS`
- `N_WARMUP`, `N_TIMED` — timing (median of `N_TIMED` launches)
- `DTYPES = ("double", "float")` — fp64 and fp32 builds

Env overrides: `CUSADI2_GPU` (device index, default `1`), `NVCC`.

## Outputs

- `build/<dtype>/` — generated `kernel.cu`, per function a per-stage and an
  unrolled `*_plan.h` header, and the compiled `latency` binary (transient).
- `results/latency_<dtype>.csv` — one row per `(func, config, B)`:
  `dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us` (cudaEvent
  kernel-only and C++-hosted end-to-end). A `nan` latency means that case was
  skipped — either its unrolled plan exceeds the device per-block shared-memory
  limit (infeasible without tiling) or its buffers exceed ~90% of free GPU memory.
- `results/latency_py_<dtype>.csv` — same schema and sweep, but each `evaluate()`
  timed from Python via ctypes (`run.py --pylib`); `wall_us` includes Python + FFI
  dispatch.
- `results/mjx_<robot>_<fpXX>.json` — MJX per batch: `results` (medians) and
  `compile` (first-call JIT), keys `{fk,jac}_{batched,rollout}`.
- `results/warp_<robot>.json` — mujoco-warp FK medians per batch + `compile_once_s`.
- `results/compile_cusadi2_<dtype>.json` — cusadi2 per-function `codegen_s` + `nvcc_s`.
- `results/latency_<dtype>.png` / `.pdf` — 8 latency panels (f64 fig has no warp).
- `results/compile_<dtype>.png` / `.pdf` — 8 compile-time panels.

## Files

- `run.py` — builds the CasADi FK / FK-Jacobian functions (restricted to the
  common body frames), emits the kernel and plan headers per dtype, compiles and
  runs `latency.cu`, runs the MJX baseline (`--mjx` subprocess), then plots.
- `latency.cu` — C++/CUDA harness: uploads the per-stage and unrolled plans and
  times the four configs across the batch sweep, with shared-memory and
  free-memory feasibility guards. Also compiles as a shared library
  (`-DCUSADI2_BENCH_LIB`) exposing `c2_setup`/`c2_eval`/`c2_teardown` for the
  Python-called timing pass.
- `plot.py` — reads the CSVs (+ MJX JSONs) and renders the figures (also runnable
  standalone).
