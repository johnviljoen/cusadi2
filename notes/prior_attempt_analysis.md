# Prior Attempt Analysis: feature/level-parallel-codegen

## What was built

One commit beyond master: `40e6eef Phase 1: DAG extraction and level assignment module`.

### src/dag.py — DAG extraction
- `extract_level_plan(f)`: walks CasADi instructions in sequential order, assigns
  each compute op a topological level `d = max(slot_depth[dep_slots]) + 1`, updates
  `slot_depth[output_slot] = d`.
- Data structures: `Op`, `Level`, `LevelPlan` — clean, well-factored.
- `verify_level_plan(f, plan)`: interprets the DAG level-by-level in Python,
  compares against `f.call()`. Covers all CasADi opcodes.

### src/generateLevelParallel.py (uncommitted, on c0-validation-harness working tree)
- Implements Option B (read-then-write per level) with switch-based codegen.
- Per-level: read phase loads `ra`, `rb` from shared `w[]`, `__syncthreads()`,
  write phase computes and stores. Chunked iteration for levels wider than block size.
- C interface with `cudaFuncSetAttribute` for dynamic shared memory.

## What specifically failed

### Bug 1: Intra-level write-write conflicts
The DAG extraction uses CasADi's original slot assignments. CasADi aggressively
reuses work slots — a slot is freed and reallocated as soon as its last reader is
processed in the sequential instruction stream. Two independent compute ops can
land at the same topological level yet write to the same storage slot.

Measured conflict counts (naive DAG):
- Panda: 2,099 conflict (level,slot) pairs, 4,111 extra writers
- A1: 5,886 pairs, 19,008 extra writers
- H1: 9,918 pairs, 40,501 extra writers

Option B's read-then-write pattern handles read-write conflicts (one node reads a
slot that another node at the same level writes) but NOT write-write conflicts
(two nodes at the same level both write to the same slot — last writer wins the race,
corrupting downstream consumers of the other writer's value).

### Bug 2: Verification doesn't catch the bug
`verify_level_plan` executes ops sequentially within each level, which matches
CasADi's behavior and hides intra-level conflicts. A parallel-execution simulation
is needed to expose the issue.

### Why "correcting" the level assignment doesn't work
Adding the constraint `d = max(d, old_slot_depth + 1)` to prevent write-write
conflicts causes the critical path to explode (H1: 281 → 14,195 levels) with
mean width collapsing to ~5. The parallelism was an illusion created by slot reuse.

## Lessons for v2

1. Option C (level-aware SSA with slot recycling) is required — keeps the naive
   281-level structure and resolves conflicts by renaming write destinations.
2. Verification must simulate parallel execution (all nodes at a level execute
   simultaneously on renamed slots) to catch conflicts before codegen.
3. Peak liveness analysis confirms shared memory feasibility: H1 peaks at 8,076
   live values (63 KB), fitting in L40S extended shared memory.
