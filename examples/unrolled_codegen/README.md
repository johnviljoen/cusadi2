# Resurrected fully-unrolled level-parallel codegen

Head-to-head between cusadi2's data-driven interpreter and the design it
replaced: the fully-unrolled level-parallel codegen from the old cusadi repo
(commit `2b352bd` "Phase 2: level-parallel CUDA codegen", abandoned in
`e7ea152`: *"35k-line .cu files with 61-minute compile times and maximum warp
divergence"*). This example re-emits that design from the **current** cusadi2
`LevelPlan` so the comparison is controlled — identical SSA slot assignment,
level schedule, block size and shared-memory footprint; the only difference is
dispatch:

* **interpreter** — one fixed ~80-line kernel; each thread reads its opcode and
  operand slots from node arrays and dispatches through an `exec_op` switch.
* **unrolled** — every DAG node is a hardcoded `case` in a `switch (tid)` per
  level chunk, opcode and slots burned into the instruction stream.

For the paper functions the unrolled kernel is emitted from the existing plan
headers in `../paper_results/build/double` (the exact plans behind the paper's
interpreter numbers). The bench binary runs both backends on the same device
buffers, checks outputs elementwise, and sweeps batch with kernel-only
cudaEvent timing (one block per instance, `n_stages=1` — the paper's "lp"
configuration).

One deliberate deviation from `2b352bd`, to avoid strawmanning codegen: the old
emitter used a two-phase read/sync/write pattern per level; the current slot
allocator guarantees no slot is both read and written within a level, so we
emit single-phase nodes with one `__syncthreads()` per level — the same sync
count as the interpreter.

## Run

```bash
CUSADI2_GPU=3 python run.py --selftest       # small function, fast gate
CUSADI2_GPU=3 python run.py                  # a1/h1 fk + fkjac
CUSADI2_GPU=3 python run.py --funcs h1_fkjac_unr   # N=8 horizon-unrolled plans
```

Needs a Python with `casadi` and importable `cusadi2` (e.g.
`PYTHONPATH=<repo>/cusadi2`). Artifacts: `build/<stem>_unrolled.cu`,
`results/bench_<stem>.csv` (runtime + max-abs-diff), and
`results/compile_<stem>.json` (emit time, nvcc time per TU, line counts).

## Results (RTX 4090-class, sm_89, fp64, 2026-07-28)

Correctness: `max_abs_diff = 0.0` (bitwise) for every function and batch —
both backends execute the same expression per node in the same level order.

Runtime, kernel-only median: the unrolled kernel is **4–8x slower** at every
batch size, e.g. h1_fkjac at B=16384: interpreter 773 us vs unrolled 5168 us;
h1_fk at B=1: 13.3 us vs 54 us. Two mechanisms, both visible in the design:
`switch (tid)` forces every lane of a warp down a different case — 32-way
divergence on every chunk regardless of op homogeneity, where the interpreter's
`switch (op)` serializes only over the distinct opcodes actually present — and
each node being unique code turns the instruction stream into an icache-miss
stream (the interpreter's hot loop fits in icache permanently).

Compile time: interpreter kernel 1.1 s once, function-independent. Unrolled
nvcc grows superlinearly with function size:

| function | nodes | .cu lines | nvcc |
|---|---|---|---|
| a1_fk | 360 | 640 | 1.2 s |
| h1_fkjac | 2,180 | 5,376 | 16.7 s |
| a1_fkjac_unr (N=8) | 9,080 | 20,608 | 190 s |
| h1_fkjac_unr (N=8) | 17,440 | 41,892 | **1,147 s (19.1 min)** |

(the historical 61-minute datapoint was a ~35k-line dynamics function of the
same emission scheme, on an older toolchain). On the N=8 horizon-unrolled
Jacobians the runtime gap also widens to 10–14x (h1_fkjac_unr B=4096:
interpreter 1.27 ms vs unrolled 16.1 ms) — larger levels mean every warp is
full of distinct hardcoded cases.

Caveat for the paper: this is the *naive* unrolled emission (the one actually
built and abandoned in April 2026). A smarter codegen could sort nodes by
opcode within levels into warp-coherent groups (cf. Singe, PPoPP 2014) and
avoid the tid-switch entirely; that would attack the divergence term but not
the compile-time scaling, which is the disqualifying axis for iterative CasADi
workflows.
