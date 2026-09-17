# cusadi2: the algorithm

cusadi2 evaluates a CasADi `Function` (a DAG of scalar operations) on the GPU
over a batch and, optionally, a horizon of stages, with **one fixed CUDA
kernel** for every function. Everything function-specific is data in a plan
file the kernel interprets. File references are to `cusadi2/`.

## 1. Cost axes

| axis | original cusadi | cusadi2 |
|---|---|---|
| NVCC compile | one `.cu` per function and horizon, minutes to hours | once per project, function-independent, about a second |
| setup | that compile | plan extraction in Python, linear in instruction count, offline |
| per-block shared memory | none (thread per instance, global work vector) | `peak_live_slots * sizeof(scalar)`, bounded by slot recycling and windowing |

## 2. One launch

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
which slice of each argument a block reads and writes.

## 3. Level-parallel execution

### 3.1 The kernel

`kernel_template.py` emits one kernel: a `__device__ exec_op(op, a, b)` switch
over CasADi's op ids (generated from `ops.py`, so the double and float forms
cannot drift) and a loop that reads each node's op and slot indices from the
device arrays `node_ops / node_outs / node_in_a / node_in_b`, indexed per level
by `level_offsets / level_widths`. The switch covers every scalar op an SX
stream can carry: CasADi's ids 0 to 43 and the five added later (`erfinv`,
`log1p`, `expm1`, `hypot`, `remainder`). The planner rejects a stream holding
any other op (the MX-level ops, `printme`) at extraction, since the kernel does
not range-check ids. Every plan runs on the same symbol, so the
dynamic shared-memory cap is process-wide and raise-only
(`cusadi2_set_shared_memory`).

### 3.2 SSA renaming and levels

CasADi's instruction stream reuses work slots aggressively: an instruction may
overwrite a slot an earlier one read, which in parallel is a hazard.
`dag.py:extract_level_plan` renames and schedules it:

```
   CasADi stream (sequential, slot reuse)      level plan (parallel, hazard-free)

   instr 0:  w[3] = w[0] + w[1]                level 0:  w[3] = w[0]+w[1]
   instr 1:  w[4] = w[2] * w[0]                          w[4] = w[2]*w[0]    one
   instr 2:  w[3] = w[3] + w[4]   <- HAZARD              w[5] = w[2]*w[1]    barrier
   instr 3:  w[5] = w[2] * w[1]      (w[3] read          w[6] = sin(w[0])
   instr 4:  w[6] = sin(w[0])         then written)      level 1:  w[7] = w[3]+w[4]
```

Pass 1 gives every value a fresh SSA id and every compute op the level
`max(level of operands) + 1` (longest path, maximum parallelism). Constants are
preloaded; INPUT and OUTPUT are nodes (slot, argument, element) forming the
first and last level, sorted by element so a warp's global accesses coalesce.
Pass 2 maps SSA ids to physical slots with a free list, releasing a slot one
level after its last read. No level both reads and writes a slot, so one
barrier per level suffices, and the high-water mark is `peak_live_slots`.
Levelling on the original slots and repairing conflicts by delaying nodes does
not work: on a humanoid Jacobian the critical path grows from hundreds of
levels to over ten thousand.

Order within a level is free, since no level both reads and writes a slot, and
it decides warp divergence: a warp takes 32 consecutive nodes of the level
(`tid, tid + b, ...` for any block size `b`) and serialises over the distinct
ops among them, each op body issued at full warp cost for its active lanes.
CasADi's emission order interleaves the ops of adjacent products (a Jacobian
level reads `sub x13, add x4, mul, add, sub, mul, ...`), so the Pinocchio plans
issue 2.0 to 2.3 op bodies per warp pass and the KKT solves 1.2 to 1.7. The
planner therefore sorts each level's compute nodes by op id (CasADi's order
within an op, so a warp's operands stay near each other in the work vector),
which leaves one op per warp except where an op's run ends mid-warp: 1.05 to
1.2 op bodies per pass, within 3 to 5% of the floor of one. Everything about
the resulting warp roles (the distinct ops of a level, `ceil(run / 32)` warps
each, which warp takes which op) is fixed by the plan, so the kernel, the
plan file and the block-size rule are unchanged. Measured on the RTX 4080 in
fp64, horizon 8, by permuting the benchmarked plans: 1.4x on the frame
Hessians at every batch size, 1.2 to 1.6x on forward kinematics, the frame
Jacobians within noise (55 to 58% of their nodes are output stores, already
grouped). Padding each op's run to a multiple of 32, true single-op warps,
trades the 3 to 5% boundary waste for idle lanes and models as a wash on
every plan, so the runs are left unpadded.

Width is liveness: everything a level produces is live at once, so longest-path
plans can need far more than CasADi's `sz_w` (horizon-unrolled H1 Hessian:
417 KB against 2 KB) and the largest did not fit a block. The `window` option
forms levels within windows of that many stream instructions, each window's
levels starting below the previous window's deepest, so consumers stay near
producers and the live set stays near `sz_w`, at the cost of more, narrower
levels:

```
  compute op   level = max(floor, 1 + max level of its operands)
  INPUT node   level = floor - 1
  OUTPUT node  level = deepest + 1, emitted when its window closes
  window close every W compute instructions:  floor = deepest + 1
  W = first of  { none, 2^k, 2^(k-1), ..., 64 },  k = floor(log2 n_compute),
      with peak_live_slots(W) <= max_live          (`max_live` search)
```

`max_live` (`write_plan(..., max_live=device_max_live(dtype))`) therefore
keeps the plain plan if it fits, else takes the largest power-of-two window
that does; it is a plan-time choice, since the footprint depends on the
runtime scalar type. Measured on the RTX 4080 (99 KB per block, 100 KB per SM): a level costs
0.25 to 0.4 us at batch 1, where one block runs alone; at large batch the
footprint sets residency instead, and a window that takes the H1 Hessian from
52 KB (one block per SM) to 22 KB (two) halves its time, while one that leaves
residency unchanged costs 15%. The unrolled Hessians, previously unrunnable in
level mode, fit at 77 and 57 KB and run 10x faster than the original kernel at
batch 1, 1.3x at 16384.

### 3.3 Block size

The thread count `b` of a block is the one launch parameter. An instance takes
`T(b) = sum_l ceil(width_l / b) + n_levels` interpreter passes (each barrier
costed as a pass), and at saturation an SM works on `R(b)` instances at once,
the blocks of that size resident under its shared-memory, register, block and
thread limits. At upload the runtime picks the `b` in {32, ..., 1024}
minimising `T(b) / R(b)`, with `R(b)` from the CUDA occupancy calculator
(`cusadi2.h:pick_block_size`, `cusadi2_max_active_blocks`); ties go to the
smaller block. The plan file carries 0 unless overridden, so a plan is sized
for whichever device loads it.

Three regimes follow. A wide, shallow plan with a small footprint takes the
largest block that keeps the SM full (unrolled Jacobians: 512 threads, two
blocks). A plan whose live set alone limits residency to one or two blocks also
takes wide blocks, since idle lanes cost nothing when the alternative is an
idle SM (unrolled Hessians, the 1e5 KKT solve: 1024 threads). A small plan is
bounded by the block-count limit, where a block wider than its levels only adds
idle warps to every barrier, so it stays at 32 or 64. The earlier heuristic,
the power of two at or above the median level width, got the middle regime
wrong: a deep, narrow plan with a large live set, the shape of a dense
elimination, got 32 to 128 threads and 3 to 4 warps per SM.

Measuring every `b` for all 15 benchmark plans (six Pinocchio functions per
stage and unrolled, the CusADi paper's five KKT solves) on the RTX 4080: the
rule picks the fastest block at saturation for 14 and is within 4% for the
other; halving the barrier weight changes one pick (10% slower), doubling it
two (3%, 1%). Against the median heuristic at 16384 to 32768 instances: 2.5x on
the 1e4 solve, 1.3x on 1e5, 1.5x on the H1 Hessian, 1.2x on the unrolled H1
Jacobian, nothing slower beyond noise; at batch 1 six plans gain 1.2 to 1.5x
and two lose 3 to 4 us, where passes were traded for residency a single block
cannot use.

### 3.4 Serial mode

The plan also carries CasADi's stream verbatim: op ids in instruction order on
CasADi's `sz_w` slots, constant, input and output moves inline. It cannot be
renamed, because CasADi reuses a constant's or input's slot after its last read
and an output's source after the OUTPUT. `cusadi2_kernel_serial` gives each
(instance, stage) one thread that walks the stream without barriers, its work
vector in a global scratch buffer laid out slot-major so a warp's accesses
coalesce; `Program::use_serial(bytes)` sizes the thread count from a scratch
budget (at most 2^20 threads) and larger batches loop. This is the original
cusadi's thread-per-instance evaluation with the same mapping, order and slot
reuse, so it is bit-identical to the level kernel (section 6). Two options
remove the remaining differences: `thread_major=true` reproduces the original's
strided `work[thread * sz_w + slot]` layout, and `unrolled.py:write_unrolled_source`
emits the original's code shape, straight-line device code with one statement
per instruction, literal constants, nonzero counts in `__constant__` memory and
its split rule (one body up to 250k instructions, 200k chunks beyond). Against
the original's own generator it agrees within 10% at every batch size and
compiles in the same time (the constant-memory counts matter: as address
literals they make ptxas 2.7x slower on 205k instructions). It pays the
per-function nvcc compile cusadi2 exists to avoid, and nvcc's multiply-add
contraction moves its rounding an ulp from the interpreter's (`-fmad=false`
restores identity). The benchmark's no-level-parallelism baseline is these
unrolled kernels, the original cusadi itself, with its compile time measured.

The level program cannot replace the stream here: level order keeps three to
four times more values live than CasADi's (`peak_live_slots` 96 against `sz_w`
28 on a 300-level function). Serial mode wins when levels are too narrow to
fill a block.

### 3.5 Verifying a plan

A sequential interpreter would hide intra-level conflicts. `dag.py:verify_plan`
executes every read of a level before every write and compares with CasADi;
`verify_serial_program` runs the stream sequentially and must match exactly.

## 4. The stage axis

For horizon-structured problems the function of interest is a per-stage
function applied at every knot. Rather than unrolling the horizon symbolically
(which multiplies plan size and shared memory by the horizon) the grid gets a
stage axis, every block running the same per-stage plan. Each argument
declares how it is indexed along it:

| mode | leading-axis length | stage k reads |
|---|---|---|
| `stage` | n_stages | element k |
| `shared` | 1 | element 0 |
| `coupled` | n_stages + span | element k + offset |

`coupled` lets stage k see its neighbours (x_k and x_{k+1} from one trajectory
buffer, span 1). Outputs are always `stage`. The element index is
`((inst * count[arg] + s) * nnz[arg] + elem)`, which for `n_stages == 1` is
plain batch-major layout. The horizon, and so every count, is baked into the
plan file; a different horizon is a different file.

## 5. The plan file

`plan.py` writes one binary file per function and horizon (format version 3)
and documents the layout: peak live slots, float64 constants, the node arrays,
CasADi's stream and `sz_w`, the stage specs and counts, the CCS sparsity of
every argument, and a block size (0 unless overridden). `cusadi2.h:load_plan`
reads it; `Program::upload` raises the shared-memory cap if the device can
honour it and picks the block size (section 3.3); `evaluate` is one kernel
launch with no host work, so it captures into a CUDA graph. A plan whose
`peak_live_slots` exceeds the per-block opt-in limit still runs in serial mode.

## 6. Numerical behaviour

The kernel evaluates CasADi's graph node for node with the same operands; the
order of independent nodes (level order on the GPU, instruction order on the
CPU) changes no node's inputs, so it cannot change rounding. Results differ
from CasADi only where the CUDA math library differs from the host libm:
transcendentals can land an ulp apart, and those ulps compound with depth. A
function of `+ - * /` alone is bit-exact at any depth (a 240-level polynomial
recursion reproduces CasADi exactly in both kernels); a 300-level function with
`sin`/`cos`/`exp` differs at relative 3e-16. `erfinv` is the one op where
CasADi itself is the less accurate side: its rational approximation is off by
about 1e-11 near 0.5, where the CUDA library and the host verifier are exact. Absolute error scales with output
magnitude, so compare relatively and sample inputs inside the function's
domain (non-unit quaternions fed to a free-flyer Jacobian give outputs of 1e13
and alarming absolute errors at relative 1e-14).
