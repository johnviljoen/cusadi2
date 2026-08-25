# MPC-pattern timing demo: level-parallel vs sequential

Hardware: NVIDIA L40S. GPU 5. fp64.
Kernels: `h1_aba_jac_analytical`, `a1_aba_jac_analytical` — Pinocchio analytical ABA-derivative SX expressions compiled through both the sequential CusADi codegen and the data-driven level-parallel CUDA kernel.
Timing: CUDA events inside the kernel wrapper; 10 warmup, median of 100 timed iterations.

No MPC is solved end-to-end — the KKT assembly and linear solve are **not** measured. Implied MPC frequencies assume dynamics evaluation is ~70% of one MPC iteration (stated with every claim).

---

## H1 whole-body MPC pattern

Per MPC step: `K=5` SQP / iLQR iterations, each batching `h1_aba_jac_analytical` across `N_stages` stages in parallel.

Function: 78,290 instructions, 281 DAG levels, 8191 peak live slots (64.0 KB shared memory).

| N_stages | seq single-iter | lp single-iter | speedup | seq K=5 total | lp K=5 total | implied seq MPC Hz | implied lp MPC Hz | slack at 100 Hz (seq / lp) |
|---------:|----------------:|---------------:|--------:|--------------:|-------------:|-------------------:|------------------:|---------------------------:|
|       10 |      1,021 µs   |      267 µs    |  3.82×  |    5,105 µs   |    1,336 µs  |            137 Hz  |           524 Hz  |   +4.90 ms / +8.66 ms      |
|       20 |      1,246 µs   |      268 µs    |  4.65×  |    6,231 µs   |    1,341 µs  |            112 Hz  |           522 Hz  |   +3.77 ms / +8.66 ms      |
|       50 |      1,856 µs   |      267 µs    |  6.95×  |    9,282 µs   |    1,336 µs  |             75 Hz  |           524 Hz  |   +0.72 ms / +8.66 ms      |

**Largest `N_stages` whose `K=5` dynamics cost fits in a time budget:**

|  budget  | seq `N_stages` | lp `N_stages` | lp / seq |
|---------:|---------------:|--------------:|---------:|
|   1.0 ms |     none fit   |     none fit  |     —    |
|   2.0 ms |     none fit   |          50   |   lp-only|
|   5.0 ms |     none fit   |          50   |   lp-only|
|  10.0 ms |         50     |          50   |   1.00×  |

### Headline claims

- **H1 whole-body MPC, `N=20` stages, 5 iterations: dynamics evaluation takes 1.35 ms with LP vs 6.23 ms with sequential (4.65× speedup).**
- **At 100 Hz MPC (10 ms budget), LP leaves 8.66 ms for the linear solve + KKT assembly; sequential leaves 3.77 ms.**
- Between 2 ms and 5 ms per MPC step, LP serves `N_stages = 50` while sequential's smallest tested `N_stages=10` already overshoots those budgets (5.1 ms > 2 ms; 5.1 ms ~ 5 ms).

### Why LP's per-iteration cost is flat across `N_stages ∈ {10, 20, 50}`

LP launches one thread block per instance. At batch ≤ 142 (L40S SM count), every block gets its own SM and the kernel time is the per-instance critical path, not throughput. Sequential evaluates the whole batch on a single thread-per-instance grid where per-instance work is serialized, so wall-clock scales linearly with batch.

---

## A1 MPPI pattern

Per MPPI control step: `T=20` sequential timesteps, each evaluating `a1_aba_jac_analytical` across `N_samples` rollouts in parallel.

Function: 40,969 instructions, 226 DAG levels, 4402 peak live slots (34.4 KB shared memory).

| N_samples | seq per-step | lp per-step | speedup | seq T=20 rollout | lp T=20 rollout | seq MPPI Hz | lp MPPI Hz |
|----------:|-------------:|------------:|--------:|-----------------:|----------------:|------------:|-----------:|
|       128 |    2,084 µs  |     161 µs  | 12.96×  |       41.67 ms   |       3.22 ms   |       24 Hz |     311 Hz |
|       256 |    4,113 µs  |     170 µs  | 24.20×  |       82.27 ms   |       3.40 ms   |       12 Hz |     294 Hz |

### Headline claim

- **MPPI with 256 samples on A1: dynamics rollout over T=20 timesteps takes 3.4 ms with LP vs 82.3 ms with sequential — 24.2× speedup, 12 Hz → 294 Hz (rollout-only).**

### Why the MPPI speedup is larger than the H1 MPC speedup

At `N_samples=256` the GPU is well past SM-saturation: sequential runs out of SMs and serializes. LP's flat latency up to its knee at ~142 instances is still well above what sequential can do at the same per-step cost.

---

## Caveats

- "Implied MPC frequency" uses an assumed 70/30 split between dynamics evaluation and the rest of one MPC iteration. Actual split varies with solver implementation, sparsity exploitation, and whether the linear system is solved on GPU or CPU. Claims of specific Hz should be read as upper bounds on dynamics evaluation alone; plug in your own solver measurement.
- Inputs are random but physically plausible `(q, v, tau)` with `q ~ N(0, 0.5²)`, `v ~ N(0, 0.3²)`, `tau ~ N(0, 5²)`. Output correctness was previously validated against CasADi CPU oracle (see `tests/test_level_parallel_cuda.py --fn h1_aba_jac_analytical`); this demo measures timing only.
- Sequential kernel compile time (one-off per function): ~30–60 s for h1, ~20 s for a1. Level-parallel data-driven kernel compiles once in ~1 s and then is reused across all functions.
- Reported numbers include ctypes call overhead and CUDA-event overhead; they do not include `torch` tensor allocation (which happens outside the timing loop).

## Reproducibility

```bash
CUDA_VISIBLE_DEVICES=5 python demo/mpc-benchmark/run_mpc_demo.py
```

Raw JSON results are written under `demo/mpc-benchmark/results/mpc_demo_<timestamp>.json`.
