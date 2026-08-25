# Warp-batched (instances-across-lanes) interpreter

Prototype of the unexplored quadrant of the design space

|                 | nodes across lanes | instances across lanes |
|-----------------|--------------------|------------------------|
| **interpreted** | cusadi2            | **this example**       |
| **compiled**    | `../unrolled_codegen` (4–8x slower) | Singe (PPoPP 2014) |

`wb_kernel` keeps cusadi2's data-driven dispatch (one function-independent
kernel, compiled once in ~1 s) but adopts the lane mapping of warp-specialized
codegen: **each warp processes one DAG node at a time across 32 batch
instances in its lanes**. Node metadata (op, operand slots) is then loaded at
one warp-uniform address (a single broadcast transaction per node instead of
one per lane) and the `exec_op` switch is warp-convergent — zero divergence
regardless of the level's op mix. The structural price is a transposed shared
work vector `[slot][lane32]`: **32x the shared memory per block**.

Both kernels consume the same uploaded plan arrays (`cusadi2::StageProgram`),
run on the same device buffers, and are validated elementwise per batch.
`run.py` gates on the device per-block shared-memory limit.

## Run

```bash
CUSADI2_GPU=3 python run.py --selftest
CUSADI2_GPU=3 python run.py                          # a1_fk,h1_fk, warps 4,8
CUSADI2_GPU=3 python run.py --funcs a1_fk --warps 8,16 --batches 16384,262144
```

## Results (RTX 4090-class, sm_89, fp64, batch-major layout, 2026-07-28)

Correctness: `max_abs_diff = 0.0` (bitwise) everywhere.

* **Feasibility**: a1_fk 1.3 KB -> 42 KB ✓, h1_fk 1.9 KB -> 62 KB ✓;
  a1_fkjac -> 112 KB ✗ and h1_fkjac -> 176 KB ✗ against the 101 KB cap — the
  paper's headline Jacobians cannot run this mapping in fp64 at all.
* **selftest** (tiny function, 19 KB wb footprint): the mechanism works — wb
  is ~2x faster at saturation (B=4096: 45 us -> 23 us, W=8).
* **a1_fk**: interpreter wins at every batch up to 65536 (B=16384: 75 us vs
  91 us at W=16); wb first edges ahead at B=262144/W=16 by ~5% (1074 -> 1018 us).
* **h1_fk**: interpreter wins everywhere (B=16384: 130 us vs 210 us at W=8);
  the 62 KB footprint allows only ONE resident block per SM.

## Reading

The two savings the mapping buys (broadcast metadata loads, convergent
dispatch) are real but small — the interpreter's metadata is L2-cached and its
divergence is bounded by the few distinct opcodes per warp. The two costs are
large: the 32x shared footprint collapses occupancy (1–2 blocks/SM vs many),
and with batch-major buffers the per-lane instance mapping turns global IO
into 32-way scatters (consecutive lanes hit addresses `nnz` apart).

Untested mitigations: element-major (transposed) IO layout would restore
coalescing; fp32 halves the footprint (fkjac would fit, barely). Neither
changes the structural conclusion: even granted zero compile cost, the
warp-specialization *mapping* does not beat the plain interpreter on the
functions the paper targets — so a full Singe-style compiled variant starts
from the same shared-memory wall, pays the superlinear nvcc scaling measured
in `../unrolled_codegen`, and its only additional saving is eliminating
already-cached broadcast loads.
