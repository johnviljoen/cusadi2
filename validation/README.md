# Paper reproduction

Master script that reproduces every numerical claim in the LP paper and
flags any discrepancy above 10% (warn) / 25% (fail) against the saved
`paper_expected_values.json`.

## Quick start

```bash
# GPU 5, cusadi2 env, L40S expected
CUDA_VISIBLE_DEVICES=5 conda run -n cusadi2 \
    python validation/reproduce_all_results.py \
        --skip-mjx --skip-compilation --quick
```

Runs in ~2 minutes and exercises every non-slow section (DAG analysis,
VI-A/VI-C/VI-E/VI-F/VI-G, inline claims). The three gated sections are:

| Flag | Section | Default runtime |
|---|---|---|
| `--skip-mjx` | Tables V, VI (MJX) | ~25 minutes (JIT cost) |
| `--skip-compilation` | Table III (seq nvcc) | ~6 minutes (fn_1e5 nvcc) |
| `--quick` | all (fewer samples) | reduces median-of-100 to median-of-10 |

## Full reproduction

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n cusadi2 \
    python validation/reproduce_all_results.py
```

End-to-end time on L40S (GPU idle): **~30 min** without MJX,
**~2 hours** with MJX enabled.

## Output

Console: formatted tables alongside paper-expected values and a
discrepancy report at the end.

JSON: `validation/results/paper_reproduction_<timestamp>.json`
containing every measured value keyed by section plus the raw
discrepancy flag list.

Intermediate MJX subprocess JSONs land in
`validation/results/mjx/<robot>_<contact|nocontact>_fp64.json`.

## What's covered

| Section | Table | What it reproduces |
|---|---|---|
| `dag` | I, II | DAG structure + peak liveness for Panda, A1, A1+4×3D, H1, H1+2×6D |
| `VI-A` | II | LP vs Seq sweep (batch 1..2048) on fn_1e1..fn_1e5 |
| `VI-B` | III | Sequential nvcc time (cold) vs LP one-off kernel compile |
| `VI-C` | IV | Unconstrained A1/H1 ABA Jacobian LP vs Seq @ batch={1, 128, 256} |
| `VI-D` | V, VI | Constrained A1/H1 LP vs MJX(no-contact)/MJX(with-contact) |
| `VI-E` | VII | H1 iLQR MPC pattern: K=5 iters × N_stages={10, 20, 50} |
| `VI-F` | VIII | A1 MPPI rollout: T=20 × N_samples={128, 256} |
| `VI-G` | IX | DAG ceiling vs achieved (fn_1e3, fn_1e4, h1_aba) |
| `inline` | — | Analytical vs naive Jacobian ratio; fn_1e5 tiling stats; MJX+LP contact overheads; SRBM MPC DAG |

## Prerequisites

All artifacts that the script depends on should already exist in the repo:

- `src/casadi_functions/*.casadi` — Pinocchio functions (panda/a1/h1 ABA
  Jac, constrained-ABA-Jac, etc.)
- `src/benchmark_functions/fn_1e*.casadi` — paper synthetic functions
- `build/liblevel_parallel_kernel.so`
- `build/liblevel_parallel_tiled_kernel.so`
- `build/lib{a1,h1,panda}_aba_jac_analytical.so` — sequential CusADi
  kernels for unconstrained Jacobians
- `build_sequential/libfn_1e*.so` — sequential kernels for paper
  synthetic functions

If any are missing, the script prints which command to run (usually
`benchmark/run_paper_comparison.py` or
`tests/test_level_parallel_cuda.py --fn <fn>`).

## Discrepancy thresholds

- **OK**: within 10% of paper value. GPU timing varies by ±5% across
  runs due to thermals, clock gating, and ambient GPU contention.
- **WARN**: 10-25% off. Might indicate drift from paper setup (different
  CUDA version, thermal state, other workloads).
- **FAIL**: >25% off. Likely a real regression or a different codepath
  being exercised.

The summary prints all WARN and FAIL items at the end. Missing values
(section didn't run, file absent) are reported separately.

## Troubleshooting

- **"CUDA not available"**: the conda env is missing a PyTorch CUDA
  build. Re-check `cusadi2` is active.
- **"Detected GPU ... does not match expected NVIDIA L40S"**: timings
  won't match paper absolute numbers but ratios should; results are
  still written to JSON.
- **MJX JIT fails mid-run**: the subprocess output is captured in the
  result JSON's `stderr_tail`. Most common cause is JAX/CUDA toolkit
  version mismatch; confirm `cusadi2` has `jax[cuda12]` installed.
- **fn_1e5 sequential compile OOMs**: this `.cu` is large (103 K
  instructions); nvcc can use >10 GB during PTX assembly. Either wait
  longer or pass `--skip-compilation`.

## Source layout

```
validation/
├── reproduce_all_results.py       # master CLI — lazy-imports sections
├── paper_expected_values.json     # every number the paper claims
├── README.md                      # this file
├── _helpers.py                    # SequentialFn, build_lp_fn, benchmark_evaluate
├── _section_dag.py                # Tables I + II
├── _section_VIA.py                # Table II (fn_1e*)
├── _section_VIB.py                # Table III (compile time)
├── _section_VIC.py                # Table IV (unconstrained ABA Jac)
├── _section_VID.py                # Tables V + VI (MJX)
├── _section_VIE.py                # Table VII (MPC pattern)
├── _section_VIF.py                # Table VIII (MPPI)
├── _section_VIG.py                # Table IX (ceiling vs achieved)
├── _section_inline.py             # Inline claims
└── results/                       # (created at first run)
    └── paper_reproduction_YYYYMMDD_HHMMSS.json
```
