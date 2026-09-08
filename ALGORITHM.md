# cusadi2: the algorithm

cusadi2 evaluates a CasADi `Function` (an expression DAG over scalar
operations) on the GPU at large batch sizes, and optionally over a horizon of
stages, using **one fixed CUDA kernel** for every function. Per-function data
lives in a plan file the kernel interprets. This document explains why each
piece exists and how they fit. File references are to `cusadi2/`.

## 1. Cost axes

| axis | what it measures | how cusadi2 pays it |
|---|---|---|
| NVCC compile | producing device code for a function | once per project, function-independent, about a second |
| setup | preparing a function for GPU evaluation | plan extraction in Python, linear in instruction count, done offline |
| per-block shared memory | working set of one evaluation | `peak_live_slots * sizeof(scalar)`, bounded by slot recycling |

The original cusadi generated one `.cu` per function and evaluated the unrolled
instruction list sequentially per thread. NVCC time grew with function size
(minutes to hours for medium functions) and every horizon change meant a new
compile. cusadi2 moves all function-specific content out of the compiled code
and into data.

## 2. Structure of one launch

```
  GRID   blockIdx.x = batch instance,  blockIdx.y = stage
         one block per (instance, stage); n_stages == 1 is plain batch evaluation

  BLOCK  shared memory  w[peak_live_slots]
         load constants into w
         for level = 0 .. n_levels-1:
             parallel for node in level:   w[out] = exec_op(op, w[a], w[b])
                                           or w[out] = input[a][b]   (OP_INPUT)
                                           or output[a][b] = w[out]  (OP_OUTPUT)
             __syncthreads()
```

Level parallelism is the within-block scheduler; the stage axis only changes
which slice of each argument buffer a block reads and writes.

## 3. Level-parallel execution

### 3.1 The data-driven kernel

`kernel_template.py` emits one kernel. It carries a `__device__ exec_op(op, a, b)`
switch over CasADi's op ids (built from `ops.py`, so the double and float forms
cannot drift) and a compute loop that reads each node's op id and slot indices
from device arrays: `node_ops`, `node_outs`, `node_in_a`, `node_in_b`, indexed
per level by `level_offsets` / `level_widths`. Every plan runs on the same
symbol, so the kernel compiles once and the dynamic shared-memory cap is a
process-wide, raise-only setting (`cusadi2_set_shared_memory`).

### 3.2 SSA renaming and level assignment

`dag.py:extract_level_plan` turns CasADi's instruction stream into that data.
CasADi's stream uses a work vector with aggressive slot reuse: an instruction
may write a slot that an earlier instruction read. Executed in parallel that is
a hazard.

```
   CasADi instruction stream             SSA-renamed level plan
   (sequential, slot reuse)              (parallel, no hazards)

   instr 0:  w[3] = w[0] + w[1]          level 0:  w[3] = w[0]+w[1]   all run in
   instr 1:  w[4] = w[2] * w[0]                    w[4] = w[2]*w[0]   parallel,
   instr 2:  w[3] = w[3] + w[4]   <- HAZARD        w[5] = w[2]*w[1]   one
   instr 3:  w[5] = w[2] * w[1]      (w[3] read    w[6] = sin(w[0])   syncthreads
   instr 4:  w[6] = sin(w[0])         then written)
                                                   level 1:  w[7] = w[3]+w[4]
```

Pass 1 assigns each compute op the level `max(level of operands) + 1`
(longest path), which preserves the maximum available parallelism, and gives
every value a fresh SSA id. Constants sit at depth 0 and are preloaded; INPUT
and OUTPUT are nodes (encoded as in the serial program: slot, argument,
element) that form level 0 and the last level of an unwindowed plan, sorted by
element so a warp's global loads and stores are coalesced. The plan file and
the kernel index the level arrays from 0, which is the numbering the figure
above uses. Pass 2 maps SSA ids to physical slots with a free-list: a slot is
released one level after its last read and reused. Within a level no slot is
both read and written, so one barrier per level is enough, and the high-water
mark of allocated slots is `peak_live_slots`, the block's shared-memory budget.

Longest-path levels maximise width, and width is liveness: every value a level
produces is live at once. A wide function can therefore need far more than
CasADi's own `sz_w` (the horizon-unrolled H1 Hessian: 417 KB against 2 KB),
which is why the largest plans did not fit a block's shared memory. The
planner's `window` option forms levels within windows of that many compute
instructions of CasADi's stream, each window's levels starting below the
previous window's deepest, so consumers stay close to their producers and the
live set stays near `sz_w` at the cost of more, narrower levels.
`max_live` (`write_plan(..., max_live=device_max_live(dtype))`) searches the
largest power-of-two window whose peak fits the budget, starting from the plain
longest-path plan, so any plan that already fits is unchanged and one that does
not gets the fewest extra levels. The kernel is indifferent to how the levels
were formed; what changes is the level count and the shared-memory footprint.
Measured on the RTX 4080 (99 KB per block, 100 KB per SM): a level costs about
0.25 to 0.4 us at batch 1, where one block runs alone and every level's index
loads and barrier are exposed, so forcing the H1 Hessian from 23 to 56 levels
adds 19% there. At large batch the footprint dominates instead: that same
window takes the plan from 52 KB (one block per SM) to 22 KB (two), and halves
the time; a window that leaves the residency unchanged (A1 Hessian, 31 KB to
29 KB) costs 15% throughout. The horizon-unrolled Hessians, previously
unrunnable in level mode, fit at 77 KB and 57 KB and run 10x faster than the
original kernel at batch 1 and 1.3x at 16384. So `max_live` is a budget, not
only a limit: a cap below the device maximum can buy occupancy.

Assigning levels naively on the original work slots and then trying to repair
conflicts by delaying nodes does not work: on a humanoid dynamics Jacobian the
critical path explodes from a few hundred levels to over ten thousand, because
the parallelism was hidden by slot reuse. Renaming is required.

### 3.3 Block size

One block evaluates one (instance, stage); its thread count `b` is the only
launch parameter, and the right value depends on the plan's shape. An instance
costs `T(b) = sum_l ceil(width_l / b) + n_levels` interpreter passes, counting
each level's barrier as one pass, and at saturation an SM works on `R(b)`
instances at once, the number of blocks of that size resident under its
shared-memory (the plan's footprint), register, block-count and thread limits.
The runtime picks, at upload, the `b` in {32, ..., 1024} minimising `T(b) /
R(b)`, the modelled SM time per instance, with `R(b)` from the CUDA occupancy
calculator (`cusadi2.h:pick_block_size`); ties go to the smaller block. The
plan file carries no choice (0) unless the caller overrides it, so the same
plan is sized for whichever device loads it.

The rule reads as three regimes. A wide, shallow plan whose footprint allows
many resident blocks takes the largest block that keeps the SM full (an unrolled
Jacobian: 512 threads, 2 blocks). A plan whose live set alone limits residency
to one or two blocks also takes wide blocks, because idle lanes cost nothing
when the alternative is an idle SM (an unrolled Hessian or the 1e5 KKT solve:
1024 threads, 1 block). A small plan is bounded by the block-count limit
instead, and a block wider than its levels only adds idle warps to every
barrier, so it stays at 32 or 64 threads. The previous heuristic (the smallest
power of two at or above the median level width) got the first and third
regimes right and the second wrong: a deep, narrow plan with a large live set,
which is what a dense elimination looks like, got 32 to 128 threads and hence 3
to 4 warps per SM.

Validated on the RTX 4080 by measuring every `b` for all 15 benchmark plans
(the six Pinocchio functions per stage and horizon-unrolled, the five KKT
solves of the CusADi paper): the rule picks the measured-fastest block at
saturation for 14 and is within 4% for the other; halving the barrier weight
changes one pick (10% slower), doubling it changes two (3% and 1%). Against the median heuristic, at
16384 to 32768 instances: 2.5x on the 1e4 KKT solve, 1.3x on 1e5, 1.5x on the
H1 Hessian, 1.2x on the unrolled H1 Jacobian, and no plan slower by more than
noise. At batch 1 six plans gain 1.2 to 1.5x and two lose 3 to 4 us (A1
kinematics unrolled, A1 Jacobian per stage), where the rule trades passes for
residency that batch 1 cannot use.

### 3.4 Serial mode

The plan file also carries CasADi's own instruction stream verbatim: op ids in
instruction order on CasADi's work slots (`sz_w` of them), with the constant,
input and output moves inline where CasADi placed them. They cannot be renamed
onto fresh slots the way the level plan renames them, because CasADi reuses
a constant's or input's slot after its last read and an output's source slot
after the OUTPUT instruction; even the ten-instruction demo function has both
hazards. `cusadi2_kernel_serial` gives each (instance, stage) one thread that
walks that stream with no barriers, keeping its `sz_w` work vector in a global
scratch buffer laid out slot-major so a warp's accesses to any slot coalesce.
`Program::use_serial(bytes)` sizes the thread count from a scratch budget
(capped at 2^20 threads); larger batches loop over items.

This is the thread-per-instance evaluation of the original cusadi, with the
same thread mapping, instruction order and slot reuse, so its results are
bit-identical to the level kernel's (section 6). Two things still differ from
the original, and each is an option that removes the difference:

* **Layout.** The original kept its work vector thread-major
  (`work[thread * sz_w + slot]`), so a warp's accesses to one slot were
  strided; `use_serial(..., thread_major=true)` reproduces that.
* **Code shape.** The original unrolled the stream into straight-line device
  code, one statement per instruction, compiled by nvcc per function.
  `unrolled.py:write_unrolled_source` emits exactly that from the same stream
  (thread-major global work vector, literal constants, nonzero counts read
  from `__constant__` memory, and the original's split rule: one kernel body
  up to 250k instructions, 200k-instruction chunks beyond), so the only
  change between the interpreter and the original is the code shape. Timed
  against the original's own generator on the benchmark functions it agrees
  within 10% at every batch size and compiles in the same time. The
  constant-memory counts are not cosmetic: folded into the addresses as
  literals, ptxas takes 2.7x longer on a 205k-instruction stream. It pays the per-function
  nvcc compile cusadi2 exists to avoid, and nvcc's default multiply-add
  contraction across statements makes its rounding differ from the
  interpreter's by an ulp (`-fmad=false` restores bit identity).

The benchmark's `staged` and `batch` configs run the unrolled kernels, so
their no-level-parallelism baseline is the original cusadi itself, and its
per-function compile time is measured rather than quoted.

Why the level program cannot stand in for the stream: level order is
breadth-first and keeps three to four times more values live than CasADi's
order (`peak_live_slots` 96 against `sz_w` 28 on a 300-level function), and
recycling slots sequentially over the level order barely helps, so a faithful
scratch footprint needs CasADi's order itself. Serial mode wins when levels
are too narrow to fill a block.

### 3.5 Verifying a plan

A sequential interpreter of the plan would hide intra-level conflicts, because
it executes nodes in a level one after another. `dag.py:verify_plan` executes
every read of a level before every write and compares with CasADi. It is the
regression guard for the conflict class. `dag.py:verify_serial_program` runs
the serial stream sequentially the same way and must match CasADi exactly.

## 4. The stage axis

For horizon-structured problems the function of interest is a per-stage
function applied at every knot. Rather than unrolling the horizon symbolically
(which multiplies plan size and shared memory by the horizon) the kernel adds a
grid axis: one block per (instance, stage), every block running the same
per-stage plan. Each argument declares how it is indexed along that axis:

| mode | leading-axis length | stage k reads |
|---|---|---|
| `stage` | n_stages | element k |
| `shared` | 1 | element 0 |
| `coupled` | n_stages + span | element k + offset |

`coupled` lets stage k see its neighbours (for example x_k and x_{k+1} from one
state trajectory buffer, span 1). Outputs are always `stage`. The element index
is `((inst * count[arg] + s) * nnz[arg] + elem)`, which for `n_stages == 1`
(and no coupled span) reduces to plain batch-major layout. The horizon, and therefore every count, is
baked into the plan file; a different horizon is a different file.

## 5. The plan file

`plan.py` writes the plan as one binary file (format version 3) and documents
the layout. It stores everything the runtime needs so the C++ side computes nothing but
the block size (§3.3; the file carries 0 unless the caller overrides it): peak
live slots, float64 constants, the
node arrays (INPUT and OUTPUT nodes included), CasADi's instruction stream and
`sz_w` for serial mode, the stage specs and counts, and the CasADi CCS
sparsity of every input and output.
`cusadi2.h:load_plan` reads it; `Program` uploads it and `evaluate` is exactly
one kernel launch with no host-side work, so it can be captured into a CUDA
graph. Upload raises the kernel's shared-memory cap to the plan's budget only
if the device can honour it; a plan whose `peak_live_slots` exceeds the
per-block opt-in limit fails at launch in level mode and still runs in
serial mode.

## 6. Numerical behaviour

The kernel evaluates the expression graph CasADi built node for node with the
same operands. The order in which independent nodes run (level order on the
GPU, instruction order on the CPU) changes no node's inputs, so it cannot
change rounding. Results differ from CasADi only where the CUDA math library
differs from the host libm: transcendental ops (`sin`, `exp`, `pow`, ...) can
land an ulp apart, and those ulps compound with depth. A function built from
`+ - * /` alone is bit-exact at any depth (a 240-level polynomial recursion
reproduces CasADi exactly in both kernels); a 300-level function with
`sin`/`cos`/`exp` differs at relative 3e-16. Absolute error scales with output
magnitude, so compare relatively, and sample test inputs inside the function's
domain (random non-unit quaternions fed to a free-flyer Jacobian produce
outputs of size 1e13 and absolute errors that look alarming while relative
error is still 1e-14).
