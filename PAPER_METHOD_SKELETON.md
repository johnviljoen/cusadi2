# Methodology skeleton (paper)

Two parallelism contributions, level and stage, judged on three cost axes.
Everything else is a refinement of one term of the cost model and comes after.
Each section's fix creates the next section's problem. Source material:
`ALGORITHM.md`. Device-specific numbers, ablations and baseline construction
are listed at the end as moved out.

Spine, stated in Setup, filled in by section 1, refined in section 3:

```
time per instance  ≈  c · T(b) / R(b)
  c    = cost of one interpreter pass
  T(b) = Σ_l ⌈width_l / b⌉ + n_levels   passes, block of b threads
  R(b) = instances resident per SM      (R = 1 at batch 1)
```

## Setup

- **Problem.** A CasADi `Function` is a DAG of scalar operations; we evaluate
  it over a batch and, for horizon-structured problems, a horizon of stages.
- **What CasADi hands us (definitions, with the running example).** CasADi
  flattens the DAG into an *instruction stream*: a list of scalar
  instructions `w[out] = op(w[a], w[b])`, executed top to bottom. `w` is the
  *work vector*, a scratch array; a *slot* is one index into it. Besides
  compute instructions there are moves: INPUT copies an argument element into
  a slot, OUTPUT copies a slot into a result element, and constants are loaded
  into slots. The terms the rest of the section relies on:
  - a **value** is the result of one instruction; a **slot** is a storage
    location. They are not the same thing: a slot holds many values over time.
  - a value is **live** from the instruction that produces it until the last
    instruction that reads it.
  - **slot reuse**: once a value is dead, CasADi hands its slot to the next
    value that needs one. This keeps the work vector tiny, `sz_w` slots, far
    fewer than the number of instructions (2 KB against a 417 KB live set in
    one of our plans). It is the same job a compiler's register allocator does,
    done for one specific execution order: the sequential one.
- **What the GPU gives us (definitions, each tied to the term it constrains).**
  Two hierarchies, one of software and one of hardware, and a mapping
  between them.
  - **Software: grid > block > thread.** A kernel launch creates a **grid**
    of **blocks**, each block a group of `b` **threads** (32 to 1024), every
    thread running one copy of the kernel. The grid is the whole launch: we
    use one block per (instance, stage), so a batch of B over a horizon of N
    is a grid of B × N blocks. `b` is the one launch parameter. → `b`
  - **Hardware: device > SM > warp.** The device is an array of **streaming
    multiprocessors (SMs)**, the units that execute blocks. Within an SM,
    threads run in **warps**: groups of 32 threads of one block executing in
    lockstep, one instruction at a time.
  - **Mapping.** The grid's blocks are distributed over the device's SMs. A
    block runs entirely on one SM; an SM keeps several blocks *resident* at
    once, as many as fit under its shared-memory, thread and block-count
    limits, and takes the next when one finishes. **Residency** `R(b)` is
    that per-SM count. Batch 1 leaves SMs idle (latency regime); a large
    grid fills every SM to `R(b)` (throughput regime), where a plan with
    half the footprint gets twice the blocks in flight. → `R(b)`
  - **per-block shared memory** is a small, fast on-chip memory visible to
    all threads of one block and to no other block (tens of KB, against GBs
    of global memory), drawn from its SM's budget. Our work vector lives
    here, so a plan's `peak_live_slots` must fit. → the footprint
  - **divergence**: if the threads of a warp need different operations (a
    branch, or our interpreter's switch), the warp runs each distinct
    operation in turn while the other threads idle. → `c`
  - a **barrier** is a synchronisation point: every thread of the block waits
    until all have arrived. It is how one level is known to be finished
    before the next reads its results. The cost model charges it as one
    pass (a modelling choice; its sensitivity is an ablation). → the
    `+ n_levels` in `T(b)`
- **Three cost axes.**
  - **compile time**: everything before the first evaluation;
  - **latency**: one instance alone on the device (under-saturated regime,
    cost `c·T(b)`);
  - **throughput**: device saturated (cost `c·T(b)/R(b)`).
  These are distinct axes, not one curve at two batch sizes: refinements in
  section 3 trade one for the other.
- **Starting point.** The original cusadi emits one straight-line `.cu` per
  function and horizon; NVCC time grows with instruction count. Execution is
  thread-per-instance: one thread walks CasADi's instruction stream on its
  `sz_w` work slots in a global work vector. Batch is its only parallelism.
- **Enabling design: the function as data.** One fixed kernel: an `exec_op`
  switch over CasADi's scalar op ids, generated from a single op table so the
  fp64 and fp32 forms cannot drift. A function is device arrays (ops, operand
  and output slots, constants) in a plan file. Compilation is once per
  project and function-independent; setup is plan extraction, linear in
  instruction count, offline, no compiler. Ops outside the scalar set are
  rejected at plan time.
- **One launch.** `evaluate` is a single kernel launch with no host work, so
  an entire solve captures into one CUDA graph; this is why `c·T(b)` is the
  whole latency rather than kernel time plus dispatch.
- **Problem created.** Interpreting the stream thread-per-instance fixes
  compile time only: every instruction is still paid in sequence. → 1

## 1. Level parallelism

- **1.1 The goal.** Two instructions can run at the same time if neither
  needs the other's result. We want to group the stream into *levels*: sets
  of instructions with no dependencies among them, run in parallel, one level
  after another.
- **1.2 Why the stream cannot be parallelised as given.** The stream names
  slots, not values, and slot reuse makes unrelated values share a name. In
  the running example (the Dubins car step, see Figures) `x`, `y` and `u`
  all pass through `w[0]`: three independent rows of the model, chained by
  one slot. Read the stream by slot name and two problems
  appear: (i) a *hazard*, an instruction that overwrites a slot another
  instruction in the same level still has to read; (ii) *false dependencies*,
  instructions that look ordered only because they touch the same slot,
  although the values they compute are unrelated. Sequentially neither
  matters; in parallel (i) gives wrong answers and (ii) destroys the
  parallelism. (Figure: panel a; the hazard is `i4`/`i5`, where `y` would
  overwrite `x+` before it is output.)
- **1.3 Step 1, name values instead of slots (SSA renaming).** Walk the
  stream once and give every instruction's result its own fresh name, used
  exactly once as a destination: *static single assignment* (SSA) form, the
  standard compiler representation. Each operand is rewritten to the name of
  the value that slot held at that point. What remains is the true
  dependency graph: an instruction depends only on the instructions that
  produced its operands.
- **1.4 Step 2, levels.** Give every instruction the level
  `1 + max level of its operands` (longest path from the inputs): the
  earliest it can run, hence the fewest levels and the most parallelism.
  Inputs form the first level, outputs the last, constants are preloaded.

  *Why not skip step 1?* The obvious alternative keeps CasADi's slots,
  levels the stream, and wherever two instructions of a level conflict over
  a slot, delays one to a later level. It needs no renaming and keeps the
  `sz_w` footprint. It fails because of 1.2's problem (ii): a slot that holds
  hundreds of unrelated values over the stream forces them into a serial
  chain (each overwrite must wait for every reader of the previous value),
  and each delay pushes that instruction's consumers later in turn. The
  schedule's depth becomes a property of how hard slots are reused, not of
  the computation: on a humanoid Jacobian, hundreds of levels become over ten
  thousand. Renaming removes the false dependencies at the source; step 3
  then handles hazards by construction. (The experiment behind the figure is
  not in the repo; re-run before quoting it.)
- **1.5 Step 3, give values slots again.** One slot per SSA name would need
  as many slots as instructions. So we redo CasADi's slot reuse, but for the
  parallel schedule: sweep the levels in order, take a slot from a free list
  for each new value, and return a slot to the list one level after its
  value's last read. The one-level delay guarantees that **no level both
  reads and writes the same slot**. Two consequences: a single barrier
  (thread synchronisation) per level is enough, and the order of instructions
  within a level is free. The high-water mark of slots in use is
  `peak_live_slots`, the plan's memory footprint; it is larger than `sz_w`
  because a parallel schedule keeps more values live at once (1.8).
- **1.6 Execution.** One block per (instance, stage): a block is one
  evaluation of the function, with its own work vector in the block's shared
  memory. Parallelism inside the block is over the *nodes of a level*: the
  level is an array of nodes in the plan, thread `t` of `b` executes nodes
  `t, t + b, t + 2b, ...` of it, then all threads meet at the barrier and
  move to the next level. A level narrower than `b` leaves threads idle; a
  wider one takes `ceil(width / b)` passes (1.7). Note what a warp sees:
  threads `32k .. 32k + 31` execute 32 *consecutive nodes of one level of one
  instance*, so which operations share a warp is decided by the order of the
  level array, which 1.5 left free (used in 3.1). Instances never share a
  block or a warp.

  *Why not group the same operation across instances?* The SIMD-style
  alternative: a block serves one operation for many instances, so every
  warp is single-op and divergence is zero. The design space is two choices,
  where the work vector lives and where the parallelism is:

  | | work vector on-chip (per-block shared memory) | work vector in global memory |
  |---|---|---|
  | parallel within an instance | **level mode (ours)**: some divergence, bounded footprint | cross-instance op grouping: no divergence, no residency limit |
  | one thread per instance | not possible (shared memory is per block) | serial mode / original cusadi: no divergence |

  We take the top-left corner for a locality reason. An instance's values
  would be produced and consumed by different blocks at every level, and
  blocks cannot see each other's shared memory, so the work vector (or a
  scatter and gather per level) must go through global memory: two global
  reads and a global write per node, where level mode touches global memory
  only at inputs and outputs. Scalar ops are memory-bound, so this is the
  dominant cost. What it would buy is the divergence left after 3.1, a
  fraction of `c`. It also gives nothing at batch 1 (no instances to group),
  and a level spread over blocks has no barrier (`__syncthreads` is per
  block), so each level becomes a kernel launch and the one-launch property
  is lost. Its one real advantage, no residency limit, is stated honestly:
  for plans with `R` of 1 or 2 it is not obviously slower at saturation.
  (Status: argued, not measured; the repo has no interpreted run with a
  global work vector to compare against.)
- **1.7 Cost model: passes.** A block of `b` threads takes
  `T(b) = Σ_l ⌈width_l / b⌉ + n_levels` interpreter passes, each barrier
  costed as a pass. Alone on the device, latency is `c·T(b)`.
- **1.8 Cost model: footprint and residency.** The block's footprint is
  `peak_live_slots · sizeof(scalar)`, the high-water mark of 1.5 (so planning
  is per dtype). `R(b)` is the number of blocks of size `b` an SM holds at
  once: the minimum over its shared-memory, thread and block-count limits.
  At saturation the cost is `c·T(b)/R(b)`. Stated as a formula,
  device-agnostic.
- **Problem created.** Width is liveness: everything a level produces is live
  at once, so a longest-path plan's live set far exceeds `sz_w`. A horizon
  makes this acute. Unrolled symbolically, N independent stages are N copies
  of the per-stage DAG, and what that costs depends on the representation:

  | | length / depth | width / memory |
  |---|---|---|
  | CasADi's stream (sequential) | N× more instructions | same `sz_w` (slots reused stage to stage) |
  | longest-path level plan | same depth | N× wider levels, N× live set |
  | windowed level plan (3.2) | more levels | narrower, live set back near `sz_w` |

  Longest-path levelling is blind to stages: level-l nodes of all N copies
  are independent, so they all land in level l. H1 frame Hessian, horizon 8:
  23 levels per stage and 23 unrolled, footprint 52 KB against 417 KB
  (exactly 8×), which does not fit a block. The planner must then either pay
  N× footprint or window, re-serialising work that was independent all
  along (115 levels, 83 KB). The stage structure is known; the scheduler
  just cannot see it. And the plan, like the original's compiled artefact,
  is per horizon. → 2
  (Scope: stages independent through the function, as for kinematics and
  per-knot derivatives. A rollout where stage k+1 consumes stage k's output
  unrolls into a deeper DAG and is not what the stage axis addresses.)

## 2. Stage parallelism

- **2.1 The horizon as a grid axis.** For horizon-structured problems the
  function of interest is a per-stage function applied at every knot. Rather
  than unrolling, the launch grid gets a stage axis: one block per
  (instance, stage), every block running the same per-stage plan. Horizon 1
  is plain batch evaluation. Level parallelism is the within-block scheduler;
  the stage axis only changes which slice of each argument a block reads and
  writes.
- **2.2 Indexing modes.** Each argument declares how it is indexed along the
  stage axis: `stage` (element k), `shared` (element 0), `coupled`
  (element k + offset, so stage k sees x_k and x_{k+1} from one trajectory
  buffer). Outputs are `stage`. One element-index formula covers all three
  and reduces to batch-major layout at horizon 1.
- **2.3 Consequence, compile time.** Plan size and extraction cost are
  independent of the horizon; the original compiles per function and horizon.
- **2.4 Consequence, latency.** Stages are concurrent blocks, so latency does
  not grow with the horizon until the device fills; an unrolled plan instead
  widens every level inside one block.
- **2.5 Consequence, throughput.** The footprint is per stage, so `R` is set
  by the stage function, not the horizon; but the grid has N times more
  blocks, and at saturation the unrolled plan sometimes wins. State it.
- **Problem created.** The cost model's three terms are each still loose:
  the switch diverges (`c`), large live sets limit residency (`R`), and `b`
  is unchosen. → 3

## 3. Refinements, one per term of the cost model

- **3.1 `c`: op-sorted levels (reducing warp divergence).** Naming: this is
  divergence *reduction* by data layout, not specialisation. Every thread
  still runs the full `exec_op` switch on an op id read from the plan; no
  warp is bound to an op and no code is specialised (which would contradict
  the one function-independent kernel). Nor is it provably *minimal*: a
  sort gives each op one contiguous run, so at most (distinct ops - 1) warps
  per level are mixed, but a packing that aligns run boundaries to multiples
  of 32 could do better. Claim "within a few percent of the floor", not
  "minimised".
  A warp executes 32 consecutive nodes of a
  level (1.6) and serialises over the distinct ops among them. This regroups
  nodes within a block; it is still one block per (instance, stage) and the
  kernel is unchanged, only the plan's node order. Order
  within a level is free (1.5), so sort each level by op id, keeping CasADi's
  order within an op so operands stay near each other: about one op body per
  warp pass. Input and output nodes are sorted by element so a warp's global
  accesses coalesce. Moves latency and throughput equally. Rejected variant:
  padding each op's run to a multiple of 32 (true single-op warps). Only a
  warp straddling an op boundary diverges, at most one per distinct op per
  level, so the ceiling is small; padding buys it back with about 16 idle
  lanes per op per level, which on narrow levels adds passes to `T(b)`. The
  reason is lane accounting, not memory locality: the work vector is in
  uniformly fast shared memory and padding changes no order. (Status: the
  cost model predicts a wash; not measured. Cheap ablation. Also reconcile
  "1.05 to 1.2 op bodies per pass" with "within 3 to 5% of the floor" in
  `ALGORITHM.md` before quoting either; and the claim that within-op order
  keeps operands near each other has no measurement behind it.)
- **3.2 `R` at the expense of `T`: windowing.** Form levels within windows of
  W stream instructions, each window's levels starting below the previous
  window's deepest: consumers stay near producers and the live set stays near
  `sz_w`, at the cost of more, narrower levels. W is the largest power of two
  whose footprint fits `max_live`. Latency gets worse, throughput gets better:
  first explicit axis conflict. Also what makes horizon-unrolled plans
  runnable in level mode at all.
- **3.3 `argmin T(b)/R(b)`: the block-size rule.** Chosen at upload, per
  device. Three regimes in one bullet: a wide, shallow plan takes the largest
  block that keeps the SM full; a plan whose live set limits residency to one
  or two blocks takes wide blocks (idle lanes are cheaper than an idle SM); a
  small plan, bounded by the block-count limit, stays at 32 to 64. At batch 1
  the rule gives up a few microseconds where passes were traded for residency
  one block cannot use: second explicit axis conflict.
- **3.4 Serial fallback.** When levels are too narrow to fill a block, or the
  live set exceeds per-block shared memory (level order keeps three to four
  times more values live than `sz_w`), run the Setup's thread-per-instance
  stream from the same plan. The stream cannot be renamed (CasADi reuses
  constant, input and output slots), so the plan carries both programs. Safe
  to switch to silently by 4.1.

## 4. Correctness

- **4.1 Same graph, same rounding.** The kernel evaluates CasADi's nodes one
  for one with the same operands; reordering independent nodes changes no
  node's inputs. The level kernel and the serial stream are bit-identical.
- **4.2 Verifying a plan.** A sequential interpreter would hide intra-level
  conflicts, so the verifier executes every read of a level before every write
  and compares with CasADi: the executable form of 1.5's invariant. The serial
  stream must match exactly.
- **4.3 The only deviation is the math library.** `+ − × ÷` functions are
  bit-exact at any depth; transcendentals can land an ulp from the host libm
  and compound with depth. fp32 is the same kernel with a different scalar
  type; whether it is accurate enough is the function's conditioning.

## Figures

- **Section 1 unifying figure** (sketch: https://claude.ai/artifact/Vc5Y6jhxjP2phGdckunHtk).
  One running example, the Dubins car step (x+ = x + cos th, y+ = y + sin th,
  th+ = th + u), in five panels: (a) CasADi's real stream, 12 instructions on
  `sz_w = 3`, x, y and u all through `w[0]`; (b) named values in levels,
  widths 4, 3, 2, 3; (c) level-by-slot grid, written / read / held,
  `peak_live_slots = 7`; (d) one block, `T = 8` passes; (e) the grid of
  blocks onto the device's SMs. Levels run top to bottom on shared rows in
  b, c, d. All plan numbers are from `extract_level_plan`.
- Device facts behind panel e (RTX 4080, queried): 76 SMs, warp 32, shared
  memory 100 KB per SM (99 KB per block opt-in), `R(32) = 24` for the Dubins
  plan, so 1824 blocks resident. `R` by footprint and b = 32 ... 1024:
  56 B gives 24, 20, 10, 5, 2, 1 (24-block limit, then a 1280-thread limit);
  22 KB gives 4, 4, 4, 4, 2, 1; 52 KB gives 1 throughout.
- Dubins timing (fp64, kernel-only median of 3000, one run): level 5.1 us
  against interpreted serial 6.0 us at batch 1; 214 us against 13 us at
  batch 65536. Both axes and their conflict on a five-op function. Repeat
  across runs and add the compiled baseline before quoting. Harness was in
  the session scratchpad (`dubins/time.cu`, `dubins/occ.cu`), not committed.
- The same function is the natural section 2 example (N knots = N copies of
  panel d, trajectory as a `coupled` argument).

## Moved out of the methodology

| item | destination |
|---|---|
| `write_unrolled_source`, `thread_major`, split rule, ptxas constant-memory detail, `-fmad`, agreement with the original generator | evaluation: baseline construction |
| median-width heuristic comparison, barrier-weight sensitivity, the block-size sweep | evaluation: ablations |
| RTX 4080 figures (per-level cost, per-SM capacity, window and residency timings, op-sort speedups) | evaluation |
| 3e-16 figure, 240-level polynomial, fp32 MPC KKT error | evaluation |
| `erfinv` accuracy, input-sampling advice | footnote or dropped |
| plan file binary layout, process-wide shared-memory cap, `upload` / `use_serial` API caveats | README only |
| pipeline view (plan → kernel → launch) | one overview figure |

## Open (deliberately not decided here)

- Headline table and figures; whether throughput is reported as the plateau
  and whether the knee `B* ≈ Θ·t₁` is reported. Data note for later: cusadi2
  curves fit `t₁ + (B−1)/Θ` within about 10% except at the knee; the
  original-style baselines do not fit a two-parameter curve at all.
- Whether the interpreter (function as data) is listed as a third
  contribution in the intro or only as the enabling design; the skeleton
  treats it as enabling.
