# Where FRAX's CBF safety filter spends its time

Measured split of one control tick between model evaluation (kinematics, dynamics, Jacobians, barriers and Lie derivatives) and the QP solve, for frax + cbfpy as shipped. Numbers only; no optimisation of either library.

## Setup

- GPU: NVIDIA GeForce RTX 4080, 610.43.02 (driver in the same field); CUDA runtime seen by JAX: 13030, cuDNN 92501; nvcc: Build cuda_13.3.r13.3/compiler.38244171_0
- CPU: AMD Ryzen 7 5800X3D 8-Core Processor — 8 physical cores, 16 logical; OS Linux-7.0.0-31-generic-x86_64-with-glibc2.43; Python 3.12.14
- Versions: jax 0.11.1, jaxlib 0.11.1, jax-cuda13-plugin 0.11.1, jax-cuda13-pjrt 0.11.1, frax 0.0.4, cbfpy 0.1.0, qpax 0.1.4, elastiqp 0.1.0, numpy 2.5.2
- frax source: git cc0a066b5b (pyproject version 0.1.0), used unmodified from `profiling/third_party/frax` (the pip wheel 0.0.4 lacks the demo API)
- fp64 everywhere (`JAX_ENABLE_X64=True`, as the demos); qpax additionally in fp32 (x64 disabled).
- CPU runs use frax's demo flags: `JAX_PLATFORMS=cpu`, `--xla_cpu_multi_thread_eigen=false`, `OPENBLAS_NUM_THREADS=1`, memory-optimised XLA CPU scheduler. GPU runs: `JAX_PLATFORMS=cuda`, default flags.
- No MuJoCo, no viewer. States: q uniform inside the joint limits, v ~ N(0, 0.5²) rad/s per joint (measured RMS ≈ 0.5); 25 % of the states have ~15 % of their joints within 0.01 rad of a limit moving outward, 25 % are the lowest-clearance states of a 10× random pool (collision rows), 50 % plain random. Desired end-effector poses are FK of independent random q, zero desired twist.
- Every piece is `jax.jit(jax.vmap(f))` on GPU; on CPU batch 1 is unbatched like the demo. Warm-up 10 calls, then `block_until_ready` after each timed call; median | p99 over 1000 calls (200 for B ≥ 1024). Compile time via `.lower().compile()`.
- QP: cbfpy's l1-relaxed (elastic) QP, penalties 1e3 (CBF rows) / 1e5 (torque bounds), `solver_tol` 1e-5. qpax: `solve_qp_elastic_primal` (explicit PDIP, max_iter 30). elastiqp: `elastiqp.jax.solve` (dual active set, `eps_abs` 1e-5), CPU-only FFI.

## Headline split: QP solve vs model evaluation, share of the end-to-end tick

`solve` = the solver alone on precomputed QP data; `model (e2e − solve)` = everything else in the tick (nominal controller, dynamics, barriers, Lie derivatives, QP assembly). Shares are of the end-to-end median. Separately timed pieces overlap and fuse differently, so `solve + nominal + qp_data` is also shown relative to e2e.

| problem | backend | dtype | batch | e2e median | solve | model (e2e − solve) | solve share | model share | (nominal + qp_data) / e2e |
|---|---|---|---|---|---|---|---|---|---|
| panda | elastiqp CPU | f64 | 1 | 142 | 34 | 108 | 24 % | 76 % | 97 % |
| panda | qpax CPU | f64 | 1 | 284 | 176 | 108 | 62 % | 38 % | 48 % |
| panda | qpax CPU | f32 | 1 | 402 | 297 | 105 | 74 % | 26 % | 33 % |
| panda | qpax GPU | f64 | 1 | 3,395 | 2,144 | 1,251 | 63 % | 37 % | 47 % |
| panda | qpax GPU | f64 | 16 | 2,645 | 1,874 | 771 | 71 % | 29 % | 33 % |
| panda | qpax GPU | f64 | 128 | 3,621 | 2,501 | 1,120 | 69 % | 31 % | 38 % |
| panda | qpax GPU | f64 | 1024 | 9,902 | 4,488 | 5,413 | 45 % | 55 % | 60 % |
| panda | qpax GPU | f64 | 4096 | 33.7 ms | 12.8 ms | 20.9 ms | 38 % | 62 % | 66 % |
| panda | qpax GPU | f32 | 1 | 3,346 | 2,618 | 728 | 78 % | 22 % | 23 % |
| panda | qpax GPU | f32 | 16 | 2,692 | 2,575 | 117 | 96 % | 4 % | 22 % |
| panda | qpax GPU | f32 | 128 | 2,997 | 2,796 | 201 | 93 % | 7 % | 24 % |
| panda | qpax GPU | f32 | 1024 | 4,088 | 3,099 | 989 | 76 % | 24 % | 35 % |
| panda | qpax GPU | f32 | 4096 | 8,746 | 5,264 | 3,483 | 60 % | 40 % | 44 % |
| g1 | elastiqp CPU | f64 | 1 | 753 | 148 | 605 | 20 % | 80 % | 87 % |
| g1 | qpax CPU | f64 | 1 | 1,689 | 999 | 690 | 59 % | 41 % | 38 % |
| g1 | qpax CPU | f32 | 1 | 1,907 | 1,284 | 622 | 67 % | 33 % | 27 % |
| g1 | qpax GPU | f64 | 1 | 7,133 | 6,147 | 987 | 86 % | 14 % | 20 % |
| g1 | qpax GPU | f64 | 16 | 14.9 ms | 13.7 ms | 1,149 | 92 % | 8 % | 12 % |
| g1 | qpax GPU | f64 | 128 | 21.3 ms | 19.4 ms | 1,947 | 91 % | 9 % | 15 % |
| g1 | qpax GPU | f64 | 1024 | 102 ms | 87.8 ms | 14.1 ms | 86 % | 14 % | 14 % |
| g1 | qpax GPU | f64 | 4096 | 424 ms | 363 ms | 61.8 ms | 85 % | 15 % | 14 % |
| g1 | qpax GPU | f32 | 1 | 4,393 | 3,749 | 644 | 85 % | 15 % | 26 % |
| g1 | qpax GPU | f32 | 16 | 10.8 ms | 9,994 | 787 | 93 % | 7 % | 13 % |
| g1 | qpax GPU | f32 | 128 | 11.9 ms | 10.9 ms | 1,039 | 91 % | 9 % | 14 % |
| g1 | qpax GPU | f32 | 1024 | 29.1 ms | 25.6 ms | 3,450 | 88 % | 12 % | 14 % |
| g1 | qpax GPU | f32 | 4096 | 149 ms | 133 ms | 15.6 ms | 89 % | 11 % | 11 % |

## QP dimensions

| problem | nv = QP variables | inequality rows (CBF + 2·nv torque bounds) | barrier rows | self-collision pairs | collision spheres | elastic vars incl. slacks |
|---|---|---|---|---|---|---|
| panda | 7 | 78 = 64 + 14 | 64 (joint_limits 14, singularity 1, obstacle 21, self_collision 7, floor 21) | 7 | 21 | 85 |
| g1 | 29 | 213 = 155 + 58 | 155 (joint_limits 58, self_collision 97) | 97 | 45 | 242 |

All barriers are relative degree 2 on torque (`h_2`), so each barrier row of G is one row of Lgh = ∂(ḣ₂ + α₂h₂)/∂z · g(z) and the RHS is Lfh + α(h); cbfpy forms them with a jvp over f and a vmapped jvp over the columns of g.

## Panda: timings per batch (median | p99, µs unless marked ms)

| backend | dtype | batch | t_nominal | t_qp_data | 2a dyn_terms | 2b barriers | t_solve | t_end_to_end | sum of pieces | solve / e2e | iters mean / max (batch-max mean / max) | conv. | active CBF | compile e2e (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| elastiqp CPU | f64 | 1 | 24 \| 31 | 113 \| 153 | 21 \| 29 | 102 \| 122 | 34 \| 45 | 142 \| 159 | 171 | 24 % | 6.8 / 68 (6.8 / 68) | 100 % | 93 % | 1.7 |
| qpax CPU | f64 | 1 | 24 \| 38 | 111 \| 119 | 22 \| 30 | 114 \| 131 | 176 \| 197 | 284 \| 325 | 311 | 62 % | 14.6 / 19 (14.6 / 19) | 100 % | 93 % | 1.8 |
| qpax CPU | f32 | 1 | 24 \| 30 | 108 \| 118 | 22 \| 29 | 97 \| 118 | 297 \| 311 | 402 \| 453 | 429 | 74 % | 28.0 / 30 (28.0 / 30) | 25 % | 66 % | 1.8 |
| qpax GPU | f64 | 1 | 252 \| 261 | 1,331 \| 1,498 | 178 \| 190 | 1,294 \| 1,450 | 2,144 \| 2,437 | 3,395 \| 3,882 | 3,727 | 63 % | 14.6 / 19 (14.6 / 19) | 100 % | 93 % | 3.8 |
| qpax GPU | f64 | 16 | 177 \| 209 | 706 \| 824 | 171 \| 196 | 663 \| 781 | 1,874 \| 2,144 | 2,645 \| 3,034 | 2,756 | 71 % | 14.6 / 19 (16.3 / 19) | 100 % | 93 % | 2.9 |
| qpax GPU | f64 | 128 | 220 \| 245 | 1,141 \| 1,292 | 212 \| 242 | 1,103 \| 1,240 | 2,501 \| 2,814 | 3,621 \| 4,190 | 3,861 | 69 % | 14.6 / 19 (17.3 / 19) | 100 % | 93 % | 3.0 |
| qpax GPU | f64 | 1024 | 455 \| 475 | 5,521 \| 6,081 | 401 \| 409 | 5,516 \| 6,113 | 4,488 \| 4,748 | 9,902 \| 10.2 ms | 10.5 ms | 45 % | 14.6 / 19 (18.2 / 19) | 100 % | 93 % | 3.1 |
| qpax GPU | f64 | 4096 | 1,191 \| 1,232 | 21.1 ms \| 21.5 ms | 1,019 \| 1,065 | 20.9 ms \| 21.2 ms | 12.8 ms \| 13.0 ms | 33.7 ms \| 33.8 ms | 35.1 ms | 38 % | 14.6 / 19 (19.0 / 19) | 100 % | 93 % | 3.3 |
| qpax GPU | f32 | 1 | 166 \| 192 | 597 \| 654 | 149 \| 170 | 546 \| 607 | 2,618 \| 2,991 | 3,346 \| 3,836 | 3,381 | 78 % | 30.0 / 30 (30.0 / 30) | 0 % | 67 % | 4.2 |
| qpax GPU | f32 | 16 | 158 \| 185 | 440 \| 531 | 143 \| 154 | 389 \| 421 | 2,575 \| 2,808 | 2,692 \| 2,999 | 3,173 | 96 % | 30.0 / 30 (30.0 / 30) | 0 % | 67 % | 3.1 |
| qpax GPU | f32 | 128 | 173 \| 187 | 542 \| 594 | 147 \| 167 | 461 \| 518 | 2,796 \| 3,121 | 2,997 \| 3,572 | 3,511 | 93 % | 30.0 / 30 (30.0 / 30) | 0 % | 67 % | 3.4 |
| qpax GPU | f32 | 1024 | 193 \| 225 | 1,218 \| 1,331 | 191 \| 200 | 1,190 \| 1,226 | 3,099 \| 3,452 | 4,088 \| 4,531 | 4,510 | 76 % | 30.0 / 30 (30.0 / 30) | 0 % | 67 % | 3.2 |
| qpax GPU | f32 | 4096 | 262 \| 277 | 3,567 \| 3,920 | 242 \| 273 | 3,497 \| 3,832 | 5,264 \| 5,313 | 8,746 \| 9,654 | 9,093 | 60 % | 30.0 / 30 (30.0 / 30) | 0 % | 67 % | 3.4 |

Extra columns for panda: raw barrier values without derivatives (`h2_only`) and per-piece compile times.

| backend | dtype | batch | h2_only | compile nominal (s) | compile qp_data (s) | compile solve (s) | compile e2e (s) | n calls |
|---|---|---|---|---|---|---|---|---|
| elastiqp CPU | f64 | 1 | 19 \| 25 | 0.32 | 1.43 | 0.01 | 1.67 | 1000 |
| qpax CPU | f64 | 1 | 19 \| 26 | 0.26 | 1.44 | 0.19 | 1.82 | 1000 |
| qpax CPU | f32 | 1 | 18 \| 22 | 0.26 | 1.50 | 0.19 | 1.83 | 1000 |
| qpax GPU | f64 | 1 | 1,018 \| 1,179 | 0.64 | 3.92 | 0.55 | 3.79 | 1000 |
| qpax GPU | f64 | 16 | 431 \| 488 | 0.85 | 3.85 | 0.53 | 2.94 | 1000 |
| qpax GPU | f64 | 128 | 687 \| 729 | 1.09 | 4.03 | 0.53 | 3.03 | 1000 |
| qpax GPU | f64 | 1024 | 3,818 \| 4,002 | 1.07 | 4.05 | 0.52 | 3.09 | 200 |
| qpax GPU | f64 | 4096 | 14.4 ms \| 14.6 ms | 1.06 | 4.27 | 0.83 | 3.31 | 200 |
| qpax GPU | f32 | 1 | 313 \| 353 | 0.52 | 4.35 | 0.60 | 4.17 | 1000 |
| qpax GPU | f32 | 16 | 170 \| 193 | 0.72 | 3.75 | 0.57 | 3.07 | 1000 |
| qpax GPU | f32 | 128 | 202 \| 215 | 0.86 | 4.07 | 0.64 | 3.42 | 1000 |
| qpax GPU | f32 | 1024 | 657 \| 680 | 0.94 | 3.85 | 0.63 | 3.21 | 200 |
| qpax GPU | f32 | 4096 | 2,147 \| 2,380 | 0.91 | 3.89 | 0.74 | 3.44 | 200 |

Constraint activity on the 4096-state dataset (CBF row active = residual ≤ 1e-5·max(1,|h|) at the solution):

| backend / dtype | ≥1 active CBF row | mean active rows | random states | joint-limit states | collision states | filter changed u | u_des outside torque bounds | solver converged |
|---|---|---|---|---|---|---|---|---|
| qpax gpu f64 | 93 % | 2.34 | 89 % | 96 % | 99 % | 97 % | 73 % | 100 % |
| qpax gpu f32 | 67 % | 1.57 | 58 % | 68 % | 84 % | 83 % | 73 % | 0 % |
| elastiqp cpu f64 | 93 % | 2.35 | 89 % | 96 % | 99 % | 97 % | 73 % | 100 % |
| qpax cpu f64 | 93 % | 2.34 | 89 % | 96 % | 99 % | 97 % | 73 % | 100 % |
| qpax cpu f32 | 66 % | 1.89 | 55 % | 72 % | 81 % | 72 % | 73 % | 25 % |

## G1: timings per batch (median | p99, µs unless marked ms)

| backend | dtype | batch | t_nominal | t_qp_data | 2a dyn_terms | 2b barriers | t_solve | t_end_to_end | sum of pieces | solve / e2e | iters mean / max (batch-max mean / max) | conv. | active CBF | compile e2e (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| elastiqp CPU | f64 | 1 | 55 \| 90 | 599 \| 743 | 48 \| 60 | 544 \| 635 | 148 \| 183 | 753 \| 904 | 802 | 20 % | 13.1 / 62 (13.1 / 62) | 100 % | 100 % | 2.5 |
| qpax CPU | f64 | 1 | 55 \| 61 | 595 \| 707 | 50 \| 82 | 558 \| 747 | 999 \| 1,204 | 1,689 \| 2,290 | 1,649 | 59 % | 17.5 / 30 (17.5 / 30) | 99 % | 100 % | 2.8 |
| qpax CPU | f32 | 1 | 49 \| 61 | 460 \| 574 | 41 \| 57 | 415 \| 507 | 1,284 \| 1,373 | 1,907 \| 2,296 | 1,793 | 67 % | 29.0 / 30 (29.0 / 30) | 14 % | 84 % | 2.8 |
| qpax GPU | f64 | 1 | 469 \| 545 | 927 \| 1,163 | 440 \| 454 | 905 \| 1,018 | 6,147 \| 7,195 | 7,133 \| 8,289 | 7,543 | 86 % | 17.5 / 30 (17.5 / 30) | 99 % | 100 % | 5.8 |
| qpax GPU | f64 | 16 | 525 \| 567 | 1,204 \| 1,330 | 500 \| 531 | 1,095 \| 1,176 | 13.7 ms \| 20.3 ms | 14.9 ms \| 21.5 ms | 15.4 ms | 92 % | 17.5 / 30 (20.6 / 30) | 99 % | 100 % | 5.8 |
| qpax GPU | f64 | 128 | 600 \| 675 | 2,557 \| 3,003 | 629 \| 678 | 2,474 \| 2,862 | 19.4 ms \| 25.4 ms | 21.3 ms \| 28.0 ms | 22.6 ms | 91 % | 17.5 / 30 (25.2 / 30) | 99 % | 100 % | 5.5 |
| qpax GPU | f64 | 1024 | 2,621 \| 2,874 | 12.1 ms \| 12.9 ms | 1,863 \| 2,111 | 11.3 ms \| 12.1 ms | 87.8 ms \| 88.1 ms | 102 ms \| 102 ms | 103 ms | 86 % | 17.5 / 30 (30.0 / 30) | 99 % | 100 % | 6.1 |
| qpax GPU | f64 | 4096 | 8,269 \| 9,187 | 50.5 ms \| 50.9 ms | 5,886 \| 6,510 | 46.8 ms \| 47.5 ms | 363 ms \| 364 ms | 424 ms \| 425 ms | 421 ms | 85 % | 17.5 / 30 (30.0 / 30) | 99 % | 100 % | 7.1 |
| qpax GPU | f32 | 1 | 281 \| 314 | 879 \| 1,012 | 372 \| 394 | 896 \| 998 | 3,749 \| 4,125 | 4,393 \| 4,798 | 4,909 | 85 % | 30.0 / 30 (30.0 / 30) | 0 % | 84 % | 5.6 |
| qpax GPU | f32 | 16 | 386 \| 412 | 1,043 \| 1,235 | 420 \| 474 | 1,006 \| 1,138 | 9,994 \| 10.2 ms | 10.8 ms \| 11.0 ms | 11.4 ms | 93 % | 30.0 / 30 (30.0 / 30) | 0 % | 84 % | 5.8 |
| qpax GPU | f32 | 128 | 406 \| 463 | 1,203 \| 1,501 | 425 \| 482 | 1,152 \| 1,273 | 10.9 ms \| 12.0 ms | 11.9 ms \| 12.9 ms | 12.5 ms | 91 % | 30.0 / 30 (30.0 / 30) | 0 % | 84 % | 6.0 |
| qpax GPU | f32 | 1024 | 528 \| 536 | 3,547 \| 3,838 | 616 \| 636 | 3,147 \| 3,548 | 25.6 ms \| 25.9 ms | 29.1 ms \| 29.2 ms | 29.7 ms | 88 % | 30.0 / 30 (30.0 / 30) | 0 % | 84 % | 6.3 |
| qpax GPU | f32 | 4096 | 1,421 \| 1,461 | 14.5 ms \| 14.9 ms | 1,290 \| 1,353 | 12.8 ms \| 14.1 ms | 133 ms \| 133 ms | 149 ms \| 149 ms | 149 ms | 89 % | 30.0 / 30 (30.0 / 30) | 0 % | 84 % | 6.9 |

Extra columns for g1: raw barrier values without derivatives (`h2_only`) and per-piece compile times.

| backend | dtype | batch | h2_only | compile nominal (s) | compile qp_data (s) | compile solve (s) | compile e2e (s) | n calls |
|---|---|---|---|---|---|---|---|---|
| elastiqp CPU | f64 | 1 | 18 \| 25 | 0.45 | 2.31 | 0.01 | 2.51 | 1000 |
| qpax CPU | f64 | 1 | 18 \| 25 | 0.39 | 2.28 | 0.22 | 2.75 | 1000 |
| qpax CPU | f32 | 1 | 17 \| 24 | 0.45 | 2.34 | 0.25 | 2.77 | 1000 |
| qpax GPU | f64 | 1 | 223 \| 250 | 1.22 | 5.23 | 1.63 | 5.75 | 1000 |
| qpax GPU | f64 | 16 | 228 \| 240 | 2.20 | 6.53 | 1.19 | 5.76 | 1000 |
| qpax GPU | f64 | 128 | 264 \| 300 | 2.39 | 6.54 | 1.14 | 5.53 | 1000 |
| qpax GPU | f64 | 1024 | 430 \| 468 | 2.48 | 7.30 | 1.35 | 6.13 | 200 |
| qpax GPU | f64 | 4096 | 1,023 \| 1,090 | 3.49 | 8.21 | 2.02 | 7.09 | 200 |
| qpax GPU | f32 | 1 | 208 \| 231 | 0.95 | 5.06 | 1.14 | 5.57 | 1000 |
| qpax GPU | f32 | 16 | 207 \| 239 | 2.33 | 6.33 | 1.24 | 5.84 | 1000 |
| qpax GPU | f32 | 128 | 214 \| 266 | 2.42 | 6.39 | 1.35 | 5.96 | 1000 |
| qpax GPU | f32 | 1024 | 236 \| 271 | 2.52 | 6.87 | 1.33 | 6.29 | 200 |
| qpax GPU | f32 | 4096 | 291 \| 333 | 3.37 | 7.50 | 1.57 | 6.92 | 200 |

Constraint activity on the 4096-state dataset (CBF row active = residual ≤ 1e-5·max(1,|h|) at the solution):

| backend / dtype | ≥1 active CBF row | mean active rows | random states | joint-limit states | collision states | filter changed u | u_des outside torque bounds | solver converged |
|---|---|---|---|---|---|---|---|---|
| qpax gpu f64 | 100 % | 5.85 | 100 % | 100 % | 100 % | 100 % | 36 % | 99 % |
| elastiqp cpu f64 | 100 % | 5.86 | 100 % | 100 % | 100 % | 100 % | 36 % | 100 % |
| qpax cpu f64 | 100 % | 5.85 | 100 % | 100 % | 100 % | 100 % | 36 % | 99 % |
| qpax cpu f32 | 84 % | 4.50 | 89 % | 67 % | 91 % | 85 % | 36 % | 14 % |
| qpax gpu f32 | 84 % | 3.10 | 80 % | 81 % | 95 % | 95 % | 36 % | 0 % |

## CPU throughput, one process per physical core

Each process pinned to its own core, batch 1, unbatched jit, frax's single-thread flags; all processes start together and run back to back for 10 s.

| problem | backend | dtype | piece | processes | total calls/s | per-process calls/s (mean / min) | single-process µs equiv. | 1-proc batch-1 median µs |
|---|---|---|---|---|---|---|---|---|
| panda | elastiqp | f64 | end_to_end | 8 | 40,924 | 5,116 / 4,821 | 195 | 142 |
| panda | elastiqp | f64 | solve | 8 | 217,945 | 27,244 / 26,071 | 37 | 34 |
| panda | qpax | f64 | end_to_end | 8 | 23,568 | 2,947 / 2,802 | 339 | 284 |
| panda | qpax | f64 | solve | 8 | 38,721 | 4,841 / 4,436 | 207 | 176 |
| panda | qpax | f32 | end_to_end | 8 | 16,689 | 2,087 / 1,899 | 479 | 402 |
| panda | qpax | f32 | solve | 8 | 23,008 | 2,877 / 2,663 | 348 | 297 |
| g1 | elastiqp | f64 | end_to_end | 8 | 10,352 | 1,295 / 1,288 | 772 | 753 |
| g1 | elastiqp | f64 | solve | 8 | 50,681 | 6,336 / 6,208 | 158 | 148 |
| g1 | qpax | f64 | end_to_end | 8 | 4,767 | 596 / 590 | 1,677 | 1,689 |
| g1 | qpax | f64 | solve | 8 | 7,372 | 923 / 899 | 1,084 | 999 |
| g1 | qpax | f32 | end_to_end | 8 | 4,241 | 531 / 522 | 1,884 | 1,907 |
| g1 | qpax | f32 | solve | 8 | 5,627 | 704 / 680 | 1,420 | 1,284 |

## GPU trace: end-to-end G1 / qpax / f64 at batch 1024

`jax.profiler` trace of 5 back-to-back calls (device stream events, means per call). The qpax `while_loop` is bracketed by its condition evaluations (`hlo_op` `while*`, one device event per iteration + 1); everything between the first and last of them is counted as the loop.

| quantity | per tick |
|---|---|
| wall time per call (ms) | 105.9 |
| device busy time (ms) | 103.2 |
| device span first→last event (ms) | 104.2 |
| device events (kernels + memcpys) | 2080 = 1831 kernels + 249 memcpys (31 D2H) |
| while-loop iterations run | 30 (qpax reports batch-max 30, mean 17.6, converged 99.1 %) |
| busy time before the loop: nominal + qp_data + solver init (ms) | 15.63 (789 events) |
| busy time inside the while loop (ms) | 87.55 (1291 events) |
| busy time after the loop (ms) | 0.00 (0 events) |
| **while-loop share of device busy time** | **84.9 %** (of span: 84.9 %) |

Top HLO ops by device time per call:

| hlo_op | launches / call | µs / call | in loop |
|---|---|---|---|
| `gemm_fusion_dot.453` | 30 | 18,048 | 100 % |
| `geqrf.0` | 90 | 13,581 | 100 % |
| `householder_product.0` | 60 | 12,941 | 100 % |
| `command_buffer_264` | 476 | 10,602 | 0 % |
| `loop_divide_fusion.1` | 30 | 6,551 | 100 % |
| `loop_add_divide_subtract_fusion` | 30 | 6,292 | 100 % |
| `triangular-solve.1` | 120 | 4,355 | 100 % |
| `triangular-solve` | 120 | 4,312 | 100 % |
| `loop_divide_fusion` | 30 | 4,124 | 100 % |
| `input_multiply_reduce_fusion` | 30 | 3,663 | 100 % |
| `input_reduce_fusion.2` | 30 | 2,988 | 100 % |
| `input_reduce_fusion.1` | 30 | 2,579 | 100 % |
| `command_buffer_1` | 200 | 1,091 | 0 % |
| `loop_divide_subtract_fusion` | 30 | 865 | 100 % |
| `input_reduce_fusion.3` | 30 | 796 | 100 % |

## Deviations from frax / cbfpy behaviour, and what did not work

**Source used.** The pip wheel `frax 0.0.4` in this environment predates the demo API
(`torque_control_matrices`, iiwa, the current collision model). The profiling therefore
runs against an unmodified git checkout of frax at `cc0a066` (pyproject version 0.1.0,
2026-10-07), cloned by `common.ensure_frax()` into `profiling/third_party/frax` and put at
the front of `sys.path`. cbfpy 0.1.0, qpax 0.1.4 and elastiqp 0.1.0 are the pip releases,
unmodified. All profiling code is in `profiling/`.

**Panda (problem 1).** `DemoOSCBFConfig` is imported from `examples/oscbf_demo.py` and the
OSC controller is a verbatim copy of `operational_space_control` minus the trailing
`safety_filter`. Deviations: (i) no MuJoCo environment, so the posture target `des_q` is
frax's Panda `default_configuration`, which is the same array the demo's `env.q_init`
holds (`franka_panda.default_q`); (ii) desired end-effector poses come from FK of random
joint vectors instead of the viewer's mocap target, with zero desired twist; (iii) the
state distribution is random inside the joint limits with 0.5 rad/s RMS velocities rather
than a closed-loop trajectory, and half of it is deliberately pushed towards joint limits
or collisions. `OSCBFTorqueConfig` hard-codes `backend="elastiqp"`; for the qpax runs the
attribute is changed on the config *instance* before `CBF.from_config` (the file is not
touched).

**G1 (problem 2).** frax has no G1 CBF demo. `g1_problem.G1OSCBFTorqueConfig` is
`cbf_utils.OSCBFTorqueConfig` with `Manipulator` replaced by `Humanoid`: the task Jacobian
is the stacked left/right hand Jacobian (12 × 29), `f`, `g`, `P`, `q` are otherwise
identical, torque bounds are the URDF efforts. Barriers are all relative degree 2: the 58
joint-limit rows plus frax's 97 G1 self-collision pairs (91 body–body pairs from
`body_sc_pairs` and 6 pairs against the fixed pelvis root from `root_sc_pairs`), evaluated
with frax's own `link_collision_data` and `_self_collision_distances_from_link_data`.
Class-K functions are α(h) = 10 h and α₂(h₂) = 10 h₂ as in the Panda demo. The nominal
controller is dual-hand OSC with a null-space posture task (default configuration), using
the Panda demo's gains per hand (kp 50/20, kd 20/10, posture kp 10, kd 5), built from the
same per-step terms as frax's `tests/timing/time_floating_base.py::osc_terms`. There is
no singularity, floor or external-obstacle barrier for the G1.

**What is in each timed piece.** `qp_data` is `cbf.qp_data(z, u_des)` exactly as cbfpy
builds it. `dyn_terms` (2a) evaluates `mass_matrix`, `jnp.linalg.inv(M)`, gravity +
Coriolis and the task Jacobian(s) the way the config's `f`, `g`, `_P` do. `barriers` (2b)
is cbfpy's `h_and_Lfh` plus `Lgh`, i.e. the barrier values with their Lie derivatives; cbfpy
evaluates `f` and `g` (hence M and M⁻¹) again inside those jvps, so 2a and 2b overlap and
do not add up to `qp_data`. `h2_only` (extra) is the raw barrier vector without
derivatives. `solve` is the solver called on precomputed `(P, q, G, h)` with cbfpy's
penalties and `solver_tol`: `qpax.solve_qp_elastic_primal(...)` (the call cbfpy makes; the
iteration counts come from `qpax.solve_qp_elastic`, which is the same function returning
its loop state — the two agree to 3e-8 on x) and `elastiqp.jax.solve(...).x` with
`eps_abs = solver_tol` and elastiqp's default dual-active-set method. `end_to_end` is the
nominal controller followed by `cbf.safety_filter` in one jit.

**elastiqp and vmap.** The handoff states elastiqp cannot run under `vmap`. In elastiqp
0.1.0 the JAX binding does accept `vmap` (`vmap_method="sequential"`: one FFI call that
loops over the batch on the host thread), so a vmapped elastiqp solve costs about batch ×
the single-instance time (checked at B = 16 during development). It is still CPU-only and
fp64-only, so, as requested, the tables report its single-instance CPU time and the
throughput of one process per physical core; no elastiqp GPU or fp32 entries exist. For
elastiqp "iterations" are active-set iterations, not interior-point iterations, so the
two backends' counts are not comparable.

**Batch 1 on CPU vs GPU.** On CPU batch 1 is run unbatched (plain `jax.jit`, like the
demo). On GPU every batch size, including 1, goes through `jax.jit(jax.vmap(...))`.

**fp32.** Only qpax is run in fp32 (`JAX_ENABLE_X64=False`, everything float32 including
the model). The convergence fraction and iteration counts in the tables show how the 1e-5
tolerance behaves at that precision; nothing was retuned.

**Iteration counts.** Under `vmap`, qpax's `while_loop` runs until every instance of the
batch has converged (or `max_iter` = 30), so the per-batch cost is set by the batch
maximum; both the per-instance distribution and the batch-max statistics (over the
consecutive windows of the shuffled 4096-state dataset) are reported. The loop count
observed in the GPU trace can differ from the standalone solve's batch-max because the
fused end-to-end program rounds differently and the 1e-5 KKT test is sensitive to it.

**G1 in fp32 and cbfpy's construction-time check.** `CBFConfig.__init__` validates P(z = 1)
with `np.allclose(P, Pᵀ, atol=1e-14)` and a Cholesky of P + 1e-14·I. For the G1 in float32
(entries of P reach ~1e3, from M⁻¹ squared) this fails by rounding alone and the G1 fp32
run aborted on first launch. `G1OSCBFTorqueConfig._is_symmetric_psd` therefore applies the
same test with float32-scaled tolerances (1e-4 × max|P|; XLA uses TF32 for float32 matmuls on this GPU by default, which leaves a relative asymmetry of about 2e-5 in P) when the matrix is float32, and
defers to cbfpy's check unchanged for float64. This affects only the one-off validation at
construction, not anything that is timed. The Panda fp32 run passed cbfpy's check as is.

**Nominal torques and the torque-bound rows.** Because the desired poses are FK of
independent random configurations, the task error is often large and the nominal OSC
torque exceeds a URDF effort limit in 73 % of the Panda states and 36 % of the G1 states
(`u_sat_frac` in `solver_stats.csv`). Those states make the torque-bound rows of the QP
active in addition to the barrier rows, which is more work for the solver than a tracking
controller close to its target would produce. The "active CBF row" column counts barrier
rows only.

**Trace bookkeeping.** In the batch-1024 trace the pre-loop device time (15.6 ms) contains
the nominal controller, `qp_data` and qpax's initial factorisation; the separately timed
`nominal` + `qp_data` at that batch is 14.7 ms, so the solver init is of the order of 1 ms.
The loop ran 30 iterations there, the same as the standalone solve's batch max; in a
development trace at batch 16 the fused program needed 21 iterations where the standalone
solve reported 30, which is the rounding sensitivity described above.

## frax vs cusadi2 as the model-evaluation backend (added after the profiling)

`bench_cusadi2_swap.py` swaps frax for cusadi2 in the Panda GPU / qpax / fp64 pipeline, batch
1, 4, 16, …, 4096 (`results/cusadi2_swap.md`, `.csv`, `.png`). `panda_casadi.py` re-implements the
model side (nominal OSC torque and cbfpy's P, q, G, h) in CasADi with Pinocchio's CasADi bindings on
frax's own `panda.urdf`; it matches frax + cbfpy to ≤ 1e-9 relative on 64 random states, and the
safe torques of the two pipelines agree to 1e-12 … 1e-5 (1e-2 at batch 4096, where one window
contains a QP at the convergence tolerance). cusadi2 compiles that function into one plan, and
`cusadi2_ffi.cu` runs it inside the JAX program as an XLA FFI custom call on XLA's stream, so the
comparison is jit against jit. The bare kernel timed with CUDA events equals the jit time at
every batch size: the FFI path adds no copies, and assembling the dense P and G from cusadi2's
column-major nonzeros costs nothing measurable.

**What did not work first.** The first CasADi version formed cbfpy's Lie derivatives as
`ca.jacobian(h, z) @ f` with `h` itself containing `ca.jacobian(h2, z) @ f`. Numerically identical,
but the symbolic second-order Jacobian through the 7×7 and 6×6 linear solves gave 485k SX nodes,
2839 levels (median level width 45 operations), and 90 KB of shared memory per block, so cusadi2's
level kernel was latency bound: ~1.1 ms per wave of one block per SM, 1.2 ms at batch 1 and 59 ms
at batch 4096 (54 waves), 2.8× slower than frax. Those results are kept in
`results/cusadi2_swap_jacobianform.*`. cbfpy does not form Jacobians; it uses `jax.jvp`. Writing the
same algebra with forward-mode directional derivatives (`ca.jtimes`) and `ca.cse` gives 37.7k nodes
and 361 levels for identical values. cusadi2's `max_live` budget (shared memory per block) then
trades depth for occupancy: 6000 slots → 361 levels, 35 KB, block 512; 3000 slots → 601 levels,
16 KB, block 256. Both are reported.


## Files

- `results/raw_timings.csv` — every timed piece (median/p99/mean/min/max µs, n, compile s).
- `results/solver_stats.csv` — iteration and activity statistics per configuration and batch.
- `results/cpu_throughput.csv` — N-process CPU throughput.
- `results/trace_g1_summary.json` — trace breakdown (raw trace under `results/trace/`, not committed).
- `results/elastiqp_f64_cpu.png` — per-piece bar chart of the published backend (`plot_elastiqp.py`).
- `results/panda_oscbf.mp4`, `results/panda_oscbf_gpu_b4096.mp4` (+ `.png` traces) — renders of the Panda scenario (single instance, demo backend; 4096 instances on the GPU with qpax), `render_panda_video.py`, `render_panda_batch_video.py`.
- `results/cusadi2_swap.md`, `.csv`, `.png` — frax vs cusadi2 as the model-evaluation backend (`bench_cusadi2_swap.py`, `panda_casadi.py`, `cusadi2_oscbf.py`, `cusadi2_ffi.cu`, `build_cusadi2_plan.py`, `report_cusadi2_swap.py`); `results/cusadi2_swap_jacobianform.*` is the first, badly formulated attempt kept for the record.
- `results/system.json`, `results/problem_*.json`, `results/run_all.log`.
- Scripts: `run_all.py` (everything), `bench.py`, `cpu_throughput.py`, `trace_g1.py`, `make_results.py`; problems in `panda_problem.py`, `g1_problem.py`; sampling in `sampling.py`; helpers in `common.py`.
