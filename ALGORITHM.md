# cusadi: end-to-end algorithm

This document describes the full evaluation pipeline for batched CasADi
functions on GPUs as implemented in this fork. The system covers four
operating regimes via three orthogonal contributions composed under a
single auto-dispatching entry point. The aim here is to be self-contained
enough that a reader can pick up the code with a clear mental model of
why each piece exists and how the pieces fit together.

## 1. Problem and cost axes

cusadi takes a CasADi `Function` (an expression DAG over scalar fp64
operations) and evaluates it on the GPU at large batch sizes. Three
costs determine whether a system is usable in practice:

| axis | what it measures | scales with |
|---|---|---|
| NVCC compile | one-time cost of producing a `.so` for a kernel | size of generated `.cu`; this is per-binary |
| Per-session setup | per-Python-session cost of loading a CasADi `Function` and preparing it for GPU evaluation | size of the symbolic graph; per-function-instantiation |
| Per-block shared memory | runtime smem footprint per CUDA block | depth and width of the DAG; per-evaluate |

The original cusadi paid all three at function-size and (for stage-
structured problems) horizon scale: NVCC compile of a single `.cu` per
function, sequential evaluation of the unrolled DAG. This made
medium-large CasADi functions impractical (322 s NVCC for `fn_1e5`) and
impossible at MPC horizons.

The fork's three contributions each collapse one axis:

| contribution | NVCC | per-session setup | per-block smem |
|---|---|---|---|
| **Level-parallel (LP)** | constant 1.5 s, function-independent | grows with N stages of unrolled function | grows with N |
| **Stage / hybrid (chunk)** | constant | **N-independent** (analytic chunk plan) | flat at chunk's footprint |
| **Tiled** | constant | same as LP | handles per-stage > smem via spill/reload |

The pieces compose: tiled-stage handles per-stage-too-big-for-smem
inside a stage block; chunk fuses across stages; LP runs the per-block
DAG.

## 2. How the pieces compose

Sections 3–5 describe LP, chunking, and tiling separately because each
closes a different cost axis, but at runtime they are **strictly
nested**: chunking lives at the grid level, tiling lives inside a
block, and LP is the within-block scheduler that everything else runs
on top of.

```
                ┌───────────────────────────────────────────┐
                │                  GRID                     │
                │                                           │
                │      blockIdx.x = batch instance          │
                │      blockIdx.y = stage chunk index       │   ← chunking
                │                                           │     decomposes here
                │   (one block per (instance, chunk) pair)  │
                └─────────┬─────────────────────────────────┘
                          │
                          ▼
                ┌─────────────────────────────────────────────┐
                │                  BLOCK                      │
                │                                             │
                │   shared memory  w[peak_live_slots]         │
                │   spill buffer    (only if tiled-stage)     │
                │                                             │
                │   per-block DAG = chunk_size stamped        │
                │                   copies of f_stage's DAG   │
                │                   (replicated per-stage     │
                │                    levels merge at each     │
                │                    depth into one wider     │
                │                    chunk-plan level)        │
                │                                             │
                │   ┌───────────────────────────────────────┐ │
                │   │  for tile = 0 .. n_tiles-1:           │ │   ← tiling
                │   │      [reload from global spill]       │ │     decomposes
                │   │      for level = 0 .. n_levels[tile]: │ │     here, only
                │   │          parallel for node in level: │  │     if needed
                │   │              w[out] = exec_op(...)   │  │
                │   │          __syncthreads()             │  │   ← LP
                │   │      [spill to global]                │ │     decomposes
                │   └───────────────────────────────────────┘ │     here
                └─────────────────────────────────────────────┘
```

Three knobs, three cost axes, all collapsing into a single launch:

```
  chunk_size  ─►  controls grid.y     (1 = pure stage; N = unrolled-LP)
  n_tiles     ─►  controls smem use   (1 = no tiling; >1 = spill/reload)
  level-par.  ─►  controls thread use (always: threads cooperate within
                                       each chunk-plan level)
```

The `chunk_size = 1` and `n_tiles = 1` corner is exactly the original
LP kernel. `chunk_size = N`, `n_tiles = 1` is unrolled-LP.
`chunk_size = 1`, `n_tiles > 1` is what the standalone tiled kernel
does. `chunk_size > 1`, `n_tiles > 1` would be tiled-fusion, which we
don't currently expose because if a single stage doesn't fit in smem,
fusing more stages won't help. The smart dispatcher picks one corner
or one edge of this cube based on the (DAG, device) pair (Section 6).

The level-parallel scheduling **always runs**, regardless of whether
chunking or tiling is active. What the chunk and tile knobs change is
*how big the per-block DAG is* and *whether intermediate values can
all fit in smem at once*. The within-level parallelism story is the
same:

```
  per-stage DAG (LP)          chunk-c=4 DAG (LP)
                                                                            
  level 0  ●●●                level 0  ●●●●●●●●●●●●  (4× wider)
  level 1  ●●●●●●             level 1  ●●●●●●●●●●●●●●●●●●●●●●●●
  level 2  ●●●●               level 2  ●●●●●●●●●●●●●●●●
  level 3  ●●●                level 3  ●●●●●●●●●●●●
  ...                         ...                                  
                                                                   
  threads cooperate           threads cooperate over
  over per-stage              4× more nodes per level
  width at each               (same number of levels;
  level                       linear smem; chunk_size×
                              throughput at saturation)
```

So when reading the rest of this document: every "LP" mentioned in
sections 3 (chunking) and 4 (tiling) is the SAME machinery from
Section 3 — the SSA-renamed level-parallel scheduler running over
whichever DAG the outer knob has constructed for the block.

## 3. Level-parallel execution (LP)

This is the prior contribution. The point is to evaluate a single
CasADi expression DAG in parallel within a CUDA block, and to do so
without per-function NVCC compilation.

### 3.1 Data-driven dispatch kernel

`src/level_parallel/codegen.py:KERNEL_TEMPLATE` is a fixed ~80-line
CUDA template. It contains a `__device__ exec_op(int op, double a, double b)`
switch over CasADi's `OP_*` IDs and a `__global__ evaluate_kernel(...)`
that reads per-node metadata (op id, SSA slots for operands and output)
from GPU arrays.

```cuda
for (int lev = 0; lev < n_levels; lev++) {
    for each node in level lev (parallel over threads in block):
        ra = w[node_in_a[g]];
        rb = w[node_in_b[g]];
        w[node_outs[g]] = exec_op(node_ops[g], ra, rb);
    __syncthreads();
}
```

`w[]` is the dynamic shared-memory work vector. The kernel is the same
across all CasADi functions: per-function data lives in metadata arrays
(`node_ops`, `node_outs`, `node_in_a`, `node_in_b`, `level_offsets`,
`level_widths`) uploaded to global memory.

This collapses the NVCC axis: one `.so` is built once (~1.5 s) and reused
for any input function. Compare to original cusadi, which generated a
separate `.cu` per function (322 s NVCC for `fn_1e5`).

### 3.2 SSA renaming and level assignment

`src/level_parallel/dag.py:extract_level_plan` builds the metadata.
The CasADi instruction stream uses a "work vector" with aggressive slot
reuse — instructions write back to slots that were previously read,
which would be a write-after-read hazard if executed in parallel.
SSA-renaming gives every compute output a unique slot at extraction
time, eliminating intra-level hazards. Slots are recycled at level
boundaries via a free-list once their last reader has executed.

Level assignment is naive longest-path topological. A node's level is
`max(level of dependency) + 1`, so node `n` runs at the earliest level
all its operands are available. This preserves maximum parallelism.

Concretely, what SSA renaming + level assignment buys us per CUDA
block:

```
   CasADi instruction stream             SSA-renamed level plan
   (sequential, slot reuse)              (parallel, no hazards)

   instr 0:  w[3] = w[0] + w[1]          level 0:  w[3] = w[0]+w[1]   ┐
   instr 1:  w[4] = w[2] * w[0]                    w[4] = w[2]*w[0]   │ all run
   instr 2:  w[3] = w[3] + w[4]   ◄── HAZARD       w[5] = w[2]*w[1]   │ in
   instr 3:  w[5] = w[2] * w[1]      (w[3] read    w[6] = sin(w[0])   │ parallel
   instr 4:  w[6] = sin(w[0])         then         (4 threads,        ┘
                                       written)    1 syncthreads)
                                       
   slot 3 used twice; can't            level 1:  w[7] = w[3]+w[4]
   parallelise as-is                             (depends on level 0)
```

The hazard at instr 2 (read-then-write on slot 3) is what SSA renaming
fixes: the kernel sees a fresh slot for the new value, and the level
scheduler puts independent computations into the same level for the
threads to execute concurrently.

The resulting `LevelPlan` (see dataclass in dag.py:78) carries:

- `constants`, `input_mappings`, `output_mappings` — load/store glue
- `compute_levels` — ordered list of `Level` objects, each a list of
  `SSANode` (op + operand slots + output slot)
- `nnz_in`, `nnz_out`, `peak_live_slots`, `n_ssa_slots`

Per-block shared memory is `peak_live_slots * 8` bytes.

### 3.3 Quadratic in setup, fixed

A subtle but expensive bug used to live in `_compute_liveness`: a
`max(d for ... raw_compute)` was being recomputed inside the loop over
output mappings, making total work O(|outputs| · |compute|). For H1
analytical Jacobian (~78k instructions, ~2k outputs), this took ~33 s.
Hoisting it to a single computation outside the loop dropped extraction
to ~0.9 s on the same input. The Python iteration is now linear in the
instruction count and well below the per-session setup target.

## 4. Stage compilation (chunk kernel)

Two motivating problems:

1. MPC / OCP / multi-shooting / scenario optimisation share a common
   structure: a per-stage program `f_stage(x_k, x_{k+1}, u_k, p)` is
   evaluated at every point along a horizon. If you stamp `f_stage` N
   times symbolically and pass the unrolled function to LP, the
   plan extraction time grows ~O(N · per_stage), and so does shared
   memory. The user pays for a redundant graph that has structural
   repetition by construction.
2. The unrolled DAG runs in a single block per batch instance. For
   problems where the *intra-stage* DAG already saturates an SM,
   stamping more stages into the same block doesn't add useful
   parallelism — it just inflates working set.

The fork addresses both via "chunk-axis" parallelism.

### 4.1 The chunk kernel

`src/level_parallel/codegen.py:CHUNK_KERNEL_TEMPLATE` is a strict
generalisation of the LP kernel that adds a stage axis. Each block
processes one chunk of `chunk_size` consecutive stages of `f_stage`:

```
gridDim  = (num_instances, n_chunks)
blockIdx = (inst, chunk_idx)
```

The compute body is identical to the LP kernel (level-parallel
execution of an SSA-renamed DAG). What changes is the per-arg I/O
indexing, which now slices into stage-axis user buffers based on
each input's *stage mode*:

| StageMode | layout in user buffer | offset per chunk |
|---|---|---|
| `STAGE` | `(B, n_stages, nnz)` | reads stages `[chunk_idx·c, chunk_idx·c + c)` |
| `SHARED` | `(B, nnz)` | broadcast across all chunks |
| `COUPLED` | `(B, n_stages + span, nnz)` | reads `[chunk_idx·c + offset, chunk_idx·c + offset + c)` |

`COUPLED` lets adjacent stages share state via the same user buffer:
e.g. a multi-shooting dynamics defect declares `x_k` (offset 0) and
`x_{k+1}` (offset +1, span 1) as both reading the same X tensor of
shape `(B, n_stages+1, nx)`.

The kernel reads per-input-arg metadata `in_stage_mode[arg]`,
`in_chunk_offset[arg]`, `in_total_stages[arg]`, `in_nnz_per_stage[arg]`
and computes:

```cuda
chunk_start = (mode == SHARED) ? 0 : (chunk_idx * chunk_size + offset[arg]);
base = (inst * total_stages[arg] + chunk_start) * nnz_per_stage[arg];
w[input_slots[i]] = inputs[arg][base + input_elems[i]];
```

With `chunk_size = 1` and no coupling, this collapses to the per-instance
LP kernel; with `chunk_size = n_stages` it recovers the unrolled-LP
behaviour. Intermediate values trade fusion against parallelism.

### 4.2 Analytic chunk plan replication

Naively, building the chunk DAG meant stamping `f_stage` `chunk_size`
times in CasADi (`_build_chunk_function`) and re-extracting the plan.
For A1 ABA-Jacobian at chunk_size=2 this took ~14.5 s; at unicycle
chunk_size=240 it took ~1.5 s. For unrolled-LP-equivalent at horizon
N=1000, it could take minutes.

`_build_chunk_plan_analytic` (in stage.py) replicates the per-stage
`LevelPlan` directly in pure Python integer arithmetic:

- SSA slots: each replica `k ∈ [0, c)` gets slot range
  `[k · S, (k+1) · S)` where `S = per_stage.n_ssa_slots`.
- Constants and input/output mappings: duplicated `c` times, slot indices
  shifted by `k · S`, element indices shifted by `k · nnz_per_stage` (for
  STAGE/COUPLED args) or unchanged (for SHARED).
- Compute levels: at each per-stage depth `d`, all `c` replicas merge
  into a single chunk-plan level (so they execute as one wide level —
  exactly what level-parallel scheduling wants). Each replica's compute
  node has its operand and output slots shifted by `k · S`.

Construction time becomes O(c · per_stage_size) integer ops. Validated
bit-exact against CasADi-stamped chunks across all chunk sizes on
unicycle (with COUPLED inputs) and A1 (with STAGE-only). Speedups
30–70× depending on chunk size.

The single semantic difference from CasADi-stamped: CasADi's CSE
deduplicates identical constants across replicas; analytic does not.
This results in ~5–25% larger smem footprint at large c — within the
auto-tuner's safety margin in every case we measured. For COUPLED
inputs that share user-buffer reads (e.g. `x_{k+1}` of stage k = `x_k`
of stage k+1), neither approach deduplicates the loads at the kernel
level — both rely on the L2 cache.

### 4.3 Why this composes with LP, not replaces it

At chunk_size=1, the chunk kernel does the same per-block work as the
original LP kernel: one per-stage DAG per (instance, stage) block,
with stage-aware I/O. We verified this empirically by reproducing the
paper's `a1_constrained_aba_jac` table exactly: chunk-c=1 runtime
matches original LP runtime to within 0.1% (1.00× ratio, 991 µs at
batch 1; 1963 µs at batch 256). Stage compilation is additive — the
prior LP contribution carries through unchanged at the c=1 endpoint.

## 5. Tiling

LP and chunk both assume the per-block working set fits in shared
memory (~99 KB on L40S). Some functions don't fit: `fn_1e5` peak smem
is 284 KB; `h1_constrained_aba_jac` is 148 KB.

`extract_tiled_level_plan` (in dag.py) constructs a `TiledLevelPlan`:
the per-stage DAG is split into chunks (along the *level* axis) such
that each chunk's working set fits in `max_smem_slots`. At chunk
boundaries:

- All values currently live in smem are *spilled* to per-block global
  memory (the spill buffer).
- The reset slot pool is repopulated by *selectively reloading* only
  those values that the next chunk's first level reads.

Within a chunk, levels run as in plain LP. If a single level's outputs
are too wide to fit alongside operands in smem, it sub-batches: emit
outputs in smaller groups, spilling each group's outputs to free their
slots before the next group's compute. The sub-batch spill is encoded
at codegen time via per-level `lev_sb_off`/`lev_sb_cnt` pairs.

The kernel (`evaluate_kernel_tiled` in codegen.py) handles all three
mechanisms in a single template:

```cuda
for (chunk = 0; chunk < n_chunks; chunk++) {
    // reload phase (skip chunk 0)
    for each (smem_slot, spill_pos) in boundary_reload[chunk-1]:
        w[smem_slot] = spill[spill_pos];
    __syncthreads();

    for (lev = chunk_lev_start[chunk]; lev < chunk_lev_end[chunk]; lev++) {
        // standard level-parallel compute (with sub-batching at wide levels)
        // sub-batch spill
        for each (smem, pos) in lev_sb[lev]:
            spill[pos] = w[smem];
    }

    // inter-chunk spill (skip last chunk)
    for each (smem, pos) in boundary_spill[chunk]:
        spill[pos] = w[smem];
}
```

Spill buffer is per-block global memory of size `spill_stride * 8`
bytes, indexed in the LP-tiled wrapper as
`spill_buffer + inst * spill_stride`.

Visually, tiling carves the level axis of a too-big DAG into smem-sized
slabs and walks them sequentially in time, with operand handoff via
the spill buffer:

```
   FULL PER-STAGE DAG (won't fit in smem)
   ────────────────────────────────────────────────────
   levels:  0 1 2 3 4 5 6 7 8 9 10 11 ... ... ... n
            ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●  ← peak live
                                                         exceeds smem

   TILED INTO 3 CHUNKS, executed sequentially in 1 block:
   ──────────────────────────────────────────────────────
   ┌─ tile 0 ─────┐ ┌─ tile 1 ─────┐ ┌─ tile 2 ────────┐
   │ levels 0..3  │ │ levels 4..8  │ │ levels 9..n     │
   │   ●●●●●●     │ │   ●●●●●●●●●  │ │   ●●●●●●●●●     │
   └──────────────┘ └──────────────┘ └─────────────────┘
        ⇣                ⇡   ⇣              ⇡
   spill at end       reload  spill         reload at
   of tile 0          for     for           start of
                      tile 1  tile 2        tile 2
   each ⇡⇣ = global memory round-trip via `spill[]` buffer
```

Each tile's peak-live ≤ `max_smem_slots`; what crosses the boundary is
spilled before the boundary and reloaded immediately after. Within a
tile, levels run via the normal LP scheduler.

### 5.1 Tiled-stage: tiling inside a stage block

When `f_stage` itself exceeds smem, naive stage compilation can't
launch (each block's working set would overflow). The composition is:
each `(instance, stage)` block runs the per-stage DAG with intra-stage
tiling.

`STAGE_TILED_KERNEL_TEMPLATE` (codegen.py) is the existing tiled
kernel with stage-axis I/O indexing layered on top, exactly as the
chunk kernel layered stage-axis indexing on top of plain LP. The
spill-buffer indexing is per-(instance, stage) so concurrent blocks
don't trample each other:

```cuda
spill = spill_buffer + (inst * n_stages + stage) * spill_stride;
```

`CusadiTiledStageLevelParallelFunction` (stage.py) wraps it. Smart
dispatch routes here when `per_stage_smem > device_smem`. Note that
chunk_size > 1 doesn't make sense in this regime (fusing already-
too-big stages into one block only worsens the smem problem), so
chunk_size is implicitly 1 along the stage axis here.

Validated on `h1_constrained_aba_jac` (148 KB peak, 235 tiles): the
stage-tiled output is bit-identical to the standalone tiled kernel's
output on the same inputs (max|err| = 0.00).

## 6. Smart auto-dispatch

A single Python entry point (`build_smart_stage_lp_fn` in
`validation/_helpers.py`) picks the right kernel and chunk size from
the function's properties and the device's shared-memory capacity.

Decision tree (in `smart_pick_chunk_size`):

```
extract_level_plan(f_stage)
  per_stage_peak = plan.peak_live_slots
  per_stage_smem = per_stage_peak * 8 bytes
  device_smem    = cudaDevAttrMaxSharedMemoryPerBlockOptin - margin

if per_stage_smem > device_smem:
    return DispatchDecision(mode="tiled-stage", chunk_size=1)
    # tiled-stage kernel: each block tiles the per-stage DAG.

has_coupled = any(spec.mode == COUPLED for spec in input_specs)
auto_max_c  = largest divisor of n_stages with c × per_stage_smem ≤ device_smem

if has_coupled:
    return DispatchDecision(mode="chunk", chunk_size=auto_max_c)
    # operand reuse: cache hits on shared cross-stage values.

if per_stage_smem < 4 KB:
    return DispatchDecision(mode="chunk", chunk_size=auto_max_c)
    # tiny per-stage: per-block fixed costs (constant load,
    # init, smem reset) dominate; fusion amortises them.

return DispatchDecision(mode="chunk", chunk_size=1)
# heavy independent stages: c=1 maximises SM occupancy.
```

Why these rules — empirical findings:

- **Coupled stages** with large c benefit from operand reuse via the
  L2 cache. On unicycle (67-instr stage, COUPLED), going from c=1 to
  c=N=120 takes runtime from 78 µs to 22 µs (3.5× speedup).
- **Tiny independent stages** at small c lose to per-block fixed
  costs. Fusion at large c amortises them.
- **Heavy independent stages** (A1 ABA Jac at 34 KB per-stage, all
  STAGE inputs) benefit from running at c=1, because 256 × N blocks
  saturate the GPU more thoroughly than 256 × (N/c) blocks each doing
  c stages worth of work, *and* there's no operand reuse to gain.
- **Per-stage > smem** can only run via tiled-stage; no choice.

The heuristic gets it right in 2/3 measured cases (unicycle picks
c=N optimally, H1 picks c=1 optimally) and is 5% off on A1 (picks
c=2 when c=1 was empirically optimal). Closing this gap requires a
runtime micro-benchmark; deferred as a minor refinement.

The decision-flow as a tree:

```
                            extract_level_plan(f_stage)
                            per_stage_smem = peak_live * 8
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │ per_stage > device smem?│
                          └────┬─────────────┬─────┘
                          yes  │             │  no
                               ▼             ▼
                  ┌──────────────────┐    has any input
                  │  TILED-STAGE     │    spec.mode == COUPLED?
                  │  chunk_size = 1  │       │            │
                  │  (e.g. fn_1e5,   │   yes │            │ no
                  │   h1_constrained)│       ▼            ▼
                  └──────────────────┘   CHUNK,     per_stage_smem
                                        c = max     < 4 KB?
                                       (operand     │       │
                                        reuse)   yes│       │no
                                                    ▼       ▼
                                                 CHUNK,   CHUNK,
                                                 c = max  c = 1
                                                (amortise (max SM
                                                 launch    occupancy)
                                                 overhead)
```

Examples mapped onto the tree:

```
  unicycle (67 instr, COUPLED x_k/x_kp1) ──► CHUNK c=N        ✓ rule 2
  unicycle (67 instr, all STAGE)         ──► CHUNK c=N        ✓ rule 3 (tiny)
  A1 ABA jac (41k, all STAGE, 34 KB)     ──► CHUNK c=1        ✓ rule 4
  H1 ABA jac (78k, all STAGE, 64 KB)     ──► CHUNK c=1        ✓ rule 4
  h1_constrained (170k, 148 KB)          ──► TILED-STAGE c=1  ✓ rule 1
```

## 7. Setup-time algorithm

End-to-end "construct an evaluator from a `(f_stage, n_stages,
input_specs, output_specs)` tuple" used to take 80+ seconds for
H1-class problems. The current path:

```python
build_smart_stage_lp_fn(f_stage, num_instances, n_stages, ...)
  ├── extract_level_plan(f_stage)              # ~0.9 s on H1, was ~33 s
  ├── smart_pick_chunk_size(per_stage_plan=...)  # μs
  │     └── auto_tune_chunk_size(per_stage_plan=...)  # μs
  │           └── _build_chunk_plan_analytic(per_stage_plan, c, ...) # μs
  └── CusadiHybridStageLevelParallelFunction(per_stage_plan=...)
        ├── _build_chunk_plan_analytic(per_stage_plan, c, ...)  # μs
        ├── _upload_plan(plan)                  # GPU mallocs + memcpy
        ├── _upload_chunk_metadata()            # tiny GPU memcpy
        └── _setup_io(...)                      # PyTorch tensor allocs
```

Three things make this fast, in order of impact:

1. **Algorithmic fix in `_compute_liveness`** (dag.py): the dominant
   loop was accidentally quadratic in `n_outputs × n_compute_ops`.
   Fixing the hoist gave a ~38× speedup on H1 (33 s → 0.9 s).
2. **Per-stage plan dedup**: extract once in the helper; pass through
   to picker, tuner, and wrapper via optional `per_stage_plan` kwarg.
   Avoids three redundant extractions of the same f_stage.
3. **Analytic chunk plan**: replicates `f_stage`'s plan in pure Python
   integer arithmetic instead of restamping `f_stage` symbolically in
   CasADi. Microseconds even at chunk_size = 1000+.

Empirical end-to-end:

| problem | N | first-cut | now |
|---|---|---|---|
| unicycle | 120 | 28 ms | 16 ms |
| A1 ABA jac | 12 | 10.4 s | **0.68 s** |
| A1 ABA jac | 120 | 24.3 s | **0.68 s** |
| H1 ABA jac | 12 | 74.5 s | **0.83 s** |
| H1 ABA jac | 120 | 83.9 s | **1.26 s** |

Combined with the 1.5 s NVCC compile (function-independent, paid once
per binary build), end-to-end cold start of an H1-class evaluator is
~3 s, regardless of horizon.

## 8. Numerical correctness

LP, chunk, and tiled-stage all evaluate the same traced expression
graph that CasADi built — Pinocchio (or whichever upstream library
generated the symbolic graph) is invoked at *build time* of the
`.casadi` file, not at evaluation time. Both runtime evaluators
(CasADi's interpreter, our LP kernels) walk the same fp64 instruction
list; they differ only in the order in which operations are
scheduled (CasADi: original instruction order; LP: topological-level
order). For floating-point arithmetic, different schedules of equal
algebraic expressions produce results that differ by unit roundoff.

Empirical regimes:

| function depth | abs err vs CasADi | comment |
|---|---|---|
| ≤ 108 levels (synthetic fn_1e1..fn_1e4) | **0.00** (bit-exact) | level-order matches CasADi to fp64 precision |
| 170-281 levels (a1/h1 ABA, ABA Jac) | 1e-13 to 1e-9 | order-dependent rounding, scales with depth |
| 538 levels (h1_constrained, tiled) | ~7e-7 | same depth/ordering story; tiling itself is bit-exact |

Relative error stays at fp64 unit roundoff (~10⁻¹⁵) in every regime;
absolute error scales with output magnitude. The new chunk and
tiled-stage kernels are bit-identical to their non-stage equivalents
at the corresponding chunk_size or tiling configuration; they do not
introduce new error.

### 8.1 The "1.24 catastrophic error" red herring

Random `N(0, 1)` inputs to a Pinocchio free-flyer Jacobian produce
non-unit quaternions (norm ~2), which propagate through analytical
ABA derivatives to outputs with magnitudes up to 10¹³. At those
magnitudes, fp64 unit roundoff on order-dependent schedules looks
like absolute errors of 1.24 — while relative error is still 5e-14
(fp64 working as designed).

Fix: `make_random_inputs(..., normalize_freeflyer_quat=True)` in
`validation/_helpers.py`. Opt-in flag. When enabled, normalises
`q[3:7]` (Pinocchio convention: pos[0:3], quat[3:7], joints[7:nq]).
Output magnitudes collapse to physical scale; absolute errors collapse
to ~10⁻¹³.

## 9. Implementation map

| concern | file | symbol |
|---|---|---|
| LP kernel template | `src/level_parallel/codegen.py` | `KERNEL_TEMPLATE`, `evaluate_kernel` |
| Tiled kernel template | `src/level_parallel/codegen.py` | `TILED_KERNEL_TEMPLATE`, `evaluate_kernel_tiled` |
| Chunk kernel template | `src/level_parallel/codegen.py` | `CHUNK_KERNEL_TEMPLATE`, `evaluate_chunk_kernel` |
| Stage-tiled kernel template | `src/level_parallel/codegen.py` | `STAGE_TILED_KERNEL_TEMPLATE`, `evaluate_stage_tiled_kernel` |
| Plan extraction | `src/level_parallel/dag.py` | `extract_level_plan`, `extract_tiled_level_plan` |
| Plan dataclasses | `src/level_parallel/dag.py` | `LevelPlan`, `TiledLevelPlan`, `Level`, `SSANode` |
| Analytic chunk replication | `src/level_parallel/stage.py` | `_build_chunk_plan_analytic` |
| Stage-mode declaration | `src/level_parallel/stage.py` | `StageMode`, `StageSpec` |
| Hybrid wrapper (chunk path) | `src/level_parallel/stage.py` | `CusadiHybridStageLevelParallelFunction` |
| Tiled-stage wrapper | `src/level_parallel/stage.py` | `CusadiTiledStageLevelParallelFunction` |
| Stage kernel (legacy) | `src/level_parallel/stage.py` | `CusadiStageLevelParallelFunction` |
| LP wrapper (legacy) | `src/level_parallel/function.py` | `CusadiLevelParallelFunction`, `CusadiTiledLevelParallelFunction` |
| Smart dispatch | `src/level_parallel/stage.py` | `smart_pick_chunk_size`, `DispatchDecision` |
| Auto-tune chunk size | `src/level_parallel/stage.py` | `auto_tune_chunk_size` |
| User-facing helper | `validation/_helpers.py` | `build_smart_stage_lp_fn` |
| Random-input helper | `validation/_helpers.py` | `make_random_inputs(..., normalize_freeflyer_quat=...)` |

## 10. Open issues and minor gaps

1. **n_stages must divide chunk_size.** Prime N forces c=1 even when
   fusion would help. A padding/remainder kernel would close this
   without affecting the design — minor refinement.
2. **Auto-tuner is memory-only.** Misses by ~5% on A1 N=12 (picks
   c=2 when c=1 wins by occupancy). A throughput-aware micro-benchmark
   step would close this — minor refinement.
3. **`cudaFuncSetAttribute` is per-kernel-symbol, not per-wrapper.**
   The fix is to re-assert the smem attribute on every `evaluate()`
   call. Already applied in `CusadiHybridStageLevelParallelFunction`
   and `CusadiTiledStageLevelParallelFunction`; the older
   `CusadiStageLevelParallelFunction`, `CusadiLevelParallelFunction`,
   and `CusadiTiledLevelParallelFunction` wrappers retain the latent
   bug — manifests only when sibling wrappers with different smem
   budgets coexist within one process. Worth porting if a future
   session touches their `evaluate()` paths.
4. **Plan extraction is pure Python.** ~0.9 s on H1 after the
   quadratic fix. A C++ port via the existing nanobind infrastructure
   would shave another order of magnitude — but no current workload
   needs it, and the algorithmic fix already collapsed the bottleneck.

## 11. End-to-end usage example

```python
from validation._helpers import build_smart_stage_lp_fn, make_random_inputs
from cusadi.pytorch.lp_stage import StageMode, StageSpec
import casadi as ca

# Build f_stage symbolically (or load from .casadi).
x = ca.SX.sym("x", 3); x_next = ca.SX.sym("x_next", 3)
u = ca.SX.sym("u", 2); p = ca.SX.sym("p", 8)
defect = x_next - (x + 0.05 * ca.vertcat(u[0]*ca.cos(x[2]),
                                          u[0]*ca.sin(x[2]), u[1]))
f_stage = ca.Function("unicycle_stage", [x, x_next, u, p], [defect])

# Declare stage-axis structure.
in_specs = [
    StageSpec(StageMode.COUPLED, offset=0, span=1),  # x_k
    StageSpec(StageMode.COUPLED, offset=1, span=1),  # x_{k+1}
    StageSpec(StageMode.STAGE),                       # u_k
    StageSpec(StageMode.SHARED),                      # p
]
out_specs = [StageSpec(StageMode.STAGE)]

# Smart-dispatch picks chunk_size, mode, and the right kernel.
lp, plan, decision = build_smart_stage_lp_fn(
    f_stage, num_instances=256, n_stages=120,
    input_specs=in_specs, output_specs=out_specs,
    verbose=True,
)
# Outputs: mode="chunk", chunk_size=120 (auto picks max-feasible because
# COUPLED inputs benefit from operand reuse), per-block smem ~28 KB.

# Evaluate. X buffer is shared between the two COUPLED inputs.
import torch
X = torch.randn(256, 121, 3, device="cuda", dtype=torch.double)
U = torch.randn(256, 120, 2, device="cuda", dtype=torch.double)
P = torch.randn(256, 1,   8, device="cuda", dtype=torch.double)
lp.evaluate([X, X, U, P])

# Outputs are in lp.outputs_sparse[0] of shape (256, 120, 3).
```

For floating-base robotics fixtures, generate inputs with
`make_random_inputs(f_stage, B, normalize_freeflyer_quat=True)` to
keep absolute errors at physical scale.
