## Three pipelines over batch size (N = 20, 1358 rows, fp64, RTX 4080)

Milliseconds per batched call, median (p99 in parentheses for the end-to-end columns), jit + vmap, device-resident inputs, block_until_ready, 1000 calls for B ≤ 128 else 200. The budget is 20 ms (50 Hz). `model` = nominal OSC + all QP rows; `solve` = the QP on precomputed data; `e2e` = model + solve in one jitted program.

| B | model frax | model cusadi2 | solve qpax | solve CUDA IPM | e2e frax + qpax | e2e frax + CUDA IPM | e2e cusadi2 + CUDA IPM | ours ÷ budget | frax+qpax ÷ budget | frax+IPM ÷ budget | max\|Δu\| vs frax+qpax |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.12 | 2.72 | 4.31 | 1.87 | 27.88 (28.89) | 25.73 (26.51) | 4.56 (5.16) | 0.23 | 1.39 | 1.29 | 5.9e-07 |
| 4 | 11.10 | 2.83 | 3.99 | 2.02 | 14.67 (15.42) | 12.92 (13.29) | 4.79 (5.36) | 0.24 | 0.73 | 0.65 | 1.7e-04 |
| 16 | 12.17 | 2.83 | 5.64 | 2.19 | 17.72 (18.63) | 14.18 (15.39) | 4.96 (5.50) | 0.25 | 0.89 | 0.71 | 1.7e-04 |
| 64 | 14.48 | 2.87 | 8.78 | 2.32 | 22.73 (23.33) | 16.34 (17.01) | 5.14 (5.58) | 0.26 | 1.14 | 0.82 | 2.1e-04 |
| 256 | 35.02 | 6.19 | 27.40 | 6.86 | 61.79 (63.57) | 41.58 (42.35) | 13.23 (13.53) | 0.66 | 3.09 | 2.08 | 1.3e-03 |
| 1024 | 114.71 | 23.71 | 116.25 | 23.48 | 230.02 (230.36) | 137.52 (137.75) | 46.87 (47.23) | 2.34 | 11.50 | 6.88 | 1.3e-03 |
| 4096 | 444.19 | 92.17 | 450.00 | 90.38 | 887.17 (887.43) | 523.42 (523.62) | 181.76 (181.89) | 9.09 | 44.36 | 26.17 | 1.0e-01 |

XLA compile time of the jitted end-to-end program per batch size (s), frax + CUDA IPM (the frax + qpax program was already compiled for the reference torques, so its compile time is not separately measured; it is the same order): B=1: 22; B=4: 23; B=16: 22; B=64: 22; B=256: 22; B=1024: 21; B=4096: 21. The cusadi2 + CUDA IPM program compiles in 0.2 to 0.3 s at every batch (two FFI calls and a scatter); the cusadi2 plans are batch-independent.

max|Δu| is the largest difference in the safe torque against the frax + qpax pipeline over the first batch window. It grows with batch because the batch is more likely to contain a QP that the solvers leave at the 1e-5 KKT tolerance after different iterates (both solvers converge on 99.9 % of the instances; the remaining ones hit the 30-iteration cap), the same effect as in profiling/results/cusadi2_swap.md.

## Pass test at batch 256 (fixed before the runs)

- ours (cusadi2 model + CUDA IPM) p99 ≤ 16 ms: **13.53 ms → PASS**
- frax model side alone, fastest variant, median ≥ 30 ms: **35.0 ms → frax FAILS real time**
- frax + CUDA IPM (the fairness row): 41.6 ms = 2.08× the budget; frax + qpax: 61.8 ms = 3.09× the budget

## frax implementation variants (model side, B = 256, median ms)

| N | rows | scan + vmap-jvp | unrolled + vmap-jvp | scan + jacfwd | unrolled + jacfwd | fastest |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 590 | 15.74 | 15.12 | 29.14 | 27.17 | unrolled-jvp |
| 12 | 846 | 22.81 | 21.75 | 42.65 | 39.76 | unrolled-jvp |
| 16 | 1102 | 30.01 | 28.46 | 56.12 | 52.15 | unrolled-jvp |
| 20 | 1358 | 36.80 | 35.03 | 70.78 | 64.90 | unrolled-jvp |
| 24 | 1614 | 43.74 | 41.73 | 82.03 | 77.43 | unrolled-jvp |

### fp32 footnote (frax model side only, same states, B = 256)

| precision | unrolled-jvp | scan-jvp | rows G rel. error vs fp64 | max abs Δu vs fp64 (N m) |
|---|---:|---:|---:|---:|
| float32, XLA default (TF32 matmuls) | 8.9 ms | 9.5 ms | 5e-1 | 1.0e+2 |
| float32, matmul precision highest (no TF32) | 9.0 ms | 9.7 ms | 5.5e-6 | 3.9e-1 |

Accuracy from `fp32_check.py` (256 states, the fp32 rows solved by the fp64 CUDA IPM and compared with the fp64 solution). With XLA's default TF32 matmuls the fp32 rollout is unusable (50 % error in the rows after 20 steps of sensitivities, torques off by up to 100 N m); with TF32 disabled the rows are accurate to 5e-6 and the safe torques agree to 0.4 N m worst case (median 2e-4). qpax itself does not converge in fp32 on this QP (profiling/results.md), so the only fp32 configuration that exists is the frax model in fp32 feeding the fp64 CUDA IPM. **Caveat to the claim:** that configuration is not measured end to end (a mixed-precision jitted program needs x64 enabled, under which frax's constants promote to fp64), but the sum of its separately measured parts at B = 256 is about 9.0 + 6.9 = 15.9 ms, i.e. 0.8× the budget. In fp64, the precision the published stack runs at and the only one whose default settings give correct rows, frax misses the budget by 2.1× even with our solver; in fp32 with TF32 disabled it would fit with about 20 % headroom. Disabling TF32 costs frax nothing here (9.0 ms vs 8.9 ms), so the fp32 route is a real option for frax on a GPU with full-rate fp32 and 1/64-rate fp64. A float32 cusadi2 build (cusadi2 has float plans) was not run.

## CUDA interior-point solver

`cbf_ipm.cu`: one thread block (512 threads) per instance; each thread keeps up to 3 rows of G and their slack / dual state in registers; the 7 x 7 normal equations are block reductions (warp shuffles + one shared-memory pass) and thread 0 factors them (Cholesky). Same elastic formulation, initialisation, Mehrotra predictor-corrector, line search, 1e-5 inf-norm convergence test and 30-iteration cap as qpax's explicit backend; each block stops at its own convergence. Before the solve every CBF row with h_i > Σ_j |G_ij| u_max,j is dropped: such a row cannot be violated by any torque inside the box, so it cannot be active (exact for the box-constrained problem; the elastic relaxation makes the bounds soft at a 1e5 penalty, which never binds here). At N = 20 this removes 43 % of the rows on the dataset. Validation against qpax on 512 dataset states: identical convergence fractions, safe torques within 6e-4 N m (median 1e-6; 5e-7 with screening off), objective within 2e-5. The solver is bound by the RTX 4080's 1/64-rate FP64 pipe and by reduction latency at one block per SM: removing the H accumulation (28 FMAs per row per iteration) saves 26 %, removing the serial factorisation or the divisions saves nothing measurable.

## cusadi2 plans

| plan | SX nodes | levels | max level width | peak live slots | shared memory / block |
|---|---:|---:|---:|---:|---:|
| head | 37249 | 285 | 715 | 3936 | 31 KB |
| step | 61922 | 228 | 1270 | 4572 | 36 KB |

Bare kernel time (CUDA events, no XLA): head + N step launches, µs

| N | B = 1 | 64 | 256 | 1024 | 4096 | per step at 256 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2,540 | 2,342 | 4,956 | 17,049 | 66,015 | 290 |
| 20 | 2,820 | 2,909 | 6,150 | 21,150 | 81,932 | 291 |
| 24 | 3,345 | 3,458 | 7,306 | 25,141 | 97,424 | 291 |

## Closed loop, 256 instances, 20 s at 50 Hz

- applied pipeline (cusadi2 + CUDA IPM): median 12.91 ms, p99 13.29 ms, max 14.71 ms per tick
- frax + qpax on the same states (not applied): median 56.6 ms, p99 59.4 ms
- batch-minimum barrier over the run: -0.0056 (per group: joint_limits -0.000, singularity -0.000, obstacle -0.006, self_collision -0.001, floor -0.000)
- filter changed the torque in 70.9 % of instance-ticks; the predictive rows changed it (vs the one-step filter on the same state) in 8.39 % (max 14.5 % of the batch in one tick), in 99 % of ticks; a predictive row was active in 29.69 % of instance-ticks
- IPM: max iterations in any batch 29, lowest converged fraction in a tick 100.0 %
- MuJoCo (256 × 10 substeps, 12 threads): median 9.8 ms per tick

