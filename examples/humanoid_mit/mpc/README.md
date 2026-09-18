# MIT humanoid MPC QP: stage-parallel cusadi2 + cuDSS

Whole-QP latency benchmark for the wip-branch ("CusADi 2.0") MIT humanoid MPC
formulation, solved with cusadi2's stage axis for the per-stage work and cuDSS
for the batched block-tridiagonal KKT solve. Compared against the same QP with
the wip's monolithic whole-horizon CasADi functions in the original CusADi
execution model (one thread per instance) feeding the same cuDSS solve, and
against CPU solves.

## Problem

The full formulation, QP, Ruiz scaling, ADMM iteration and the staged
decomposition are written out in `FORMULATION.md`.

10-stage MPC over the MIT humanoid (24 dof + 12 contact forces): 60 primal
variables and 74 constraints per stage, KKT of 1340 rows with 11,162 lower
nonzeros. OSQP-style ADMM (rho 0.1, sigma 1e-6, alpha 1.6, 2 Ruiz passes,
25 iterations, one SQP step at x = 0), exactly the wip `admm_backend`.

## Pipeline (one solve, one CUDA graph)

| step | cusadi2 function | grid | notes |
|---|---|---|---|
| assemble | `mpc_assemble` (77,859 instr, 173 levels, 64 KB smem) | B x 10 | KKT block row `[P_kk, P_{k-1,k}, A_kk, A_{k,k-1}]`, gradient, bounds; reads x_{k-1}, x_k, x_{k+1} and dt_k, dt_{k+1} through coupled inputs |
| Ruiz pass 1 | `mpc_ruiz_r1`, `mpc_ruiz_r2` + max-reduce | B x 10 | column inf-norms need the neighbour block row (coupled input); the global gamma is a 1-thread-per-instance reduction |
| Ruiz pass 2 | `mpc_ruiz_r3`, `mpc_ruiz_r4` + max-reduce | B x 10 | |
| scale | `mpc_ruiz_r5` | B x 10 | scaled KKT block row (+sigma I, -1/rho), q, D, bounds, initial rhs |
| gather | 1 kernel | | per-stage block rows -> uniform-batch lower CSR values (stage-interleaved rows `[x_k; g_k]`) |
| factorize | cuDSS `CUDSS_PHASE_FACTORIZATION` | | symmetric indefinite, default pivoting/algorithms, `CUDSS_CONFIG_UBATCH_SIZE = B` |
| 25 x iterate | cuDSS solve, `mpc_admm_iter` | B x 10 | fused ADMM step + next rhs + dx = D x, zero-copy on cuDSS's rhs/solution buffers |

Three tiny `cudaMemcpy2DAsync` calls pad stage outputs into the coupled-input
layout (cusadi2 outputs are stage-mode only). cuDSS device memory comes from a
pre-reserved `cudaMemPool` through `cudssSetDeviceMemHandler`, which is what
makes its factorization and solve phases capturable; the whole solve replays as
one graph.

## Correctness (fp64, x = 0, the wip's default parameters)

- staged assembly vs the wip `KKT_matrix`: bit-exact (10,422 nonzeros)
- staged Ruiz vs the wip `ruiz_scaling`: q and D bit-exact, scaled KKT and
  bounds within 1 ulp (4e-16 relative)
- full solve vs the wip CPU path (`solve_with_custom`, symbolic LDL): 8.5e-10
  max abs error (cuDSS orders the factorization differently)
- OSQP itself on the same QP gives 1.3e-9 vs the same reference

## Results (RTX 4080, cuDSS 0.8.0, CUDA 13.3), fp64, CUDA graph

"cusadi2" is the staged pipeline above. "cusadi" is the wip's monolithic
whole-horizon `KKT_matrix`, `ruiz_scaling`, `KKT_vector` and `ADMM_step`
functions with the same cuDSS uniform-batch factorization and solves (driver
`mpc_mono_bench.cu`, mode `cudss`). That mode runs a function in cusadi2's
serial (thread-per-instance) kernel, i.e. the original CusADi execution model,
only when its level plan does not fit shared memory: `KKT_matrix` and
`ruiz_scaling` ran serial, `KKT_vector` and `ADMM_step` (35 and 53 KB) ran in
the level kernel (the driver's `how` column read
`serial/serial/cudss/level/cudss/level`). The column is therefore not a pure
original-CusADi baseline: its assembly (97% of the B = 1 latency) is, its 25
iterations are not. Mode `cudss_serial` runs all four serially; it has not
been measured, as the wip functions were no longer at hand when it was added.
The solver is identical in both columns.

| B | cusadi2 latency | cusadi2 per QP | cusadi2 QP/s | cusadi latency | cusadi per QP | cusadi QP/s | ratio |
|---|---|---|---|---|---|---|---|
| 1 | 4.06 ms | 4.06 ms | 246 | 129.7 ms | 129.7 ms | 8 | 32x |
| 4 | 4.88 ms | 1.22 ms | 820 | 132.5 ms | 33.1 ms | 30 | 27x |
| 12 | 7.99 ms | 0.67 ms | 1502 | 137.6 ms | 11.5 ms | 87 | 17x |
| 16 | 68.8 ms | 4.30 ms | 233 | 197.7 ms | 12.4 ms | 81 | 2.9x |
| 64 | 70.0 ms | 1.09 ms | 914 | 216.2 ms | 3.38 ms | 296 | 3.1x |
| 256 | 95.0 ms | 0.37 ms | 2695 | 258.7 ms | 1.01 ms | 989 | 2.7x |
| 1024 | 200 ms | 0.20 ms | 5107 | 332.8 ms | 0.33 ms | 3077 | 1.7x |
| 4096 | 724 ms | 0.18 ms | 5656 | 718.7 ms | 0.18 ms | 5699 | 1.0x |

Latency is one batched solve (median of graph replays); throughput is B
divided by that latency, so the latency and throughput ratios coincide. Both
columns are correct to about 7e-10 (cusadi) and 9e-10 (cusadi2) against the
wip CPU reference at every B.

In the cusadi column the whole-horizon assembly and Ruiz functions cost
126 ms at B = 1 (84 + 42 ms, one thread walking 885k instructions) and
176 ms at B = 4096; the staged cusadi2 versions cost 0.68 ms and 205 ms. So
the stage axis buys two orders of magnitude of latency at small batch, and at
full occupancy the thread-per-instance model is as efficient as the level
kernel, which is why the two meet at B = 4096. The wip's original symbolic
LDL solver (a 1.28M-instruction DAG in place of cuDSS) makes the cusadi
column 872 ms at B = 1 and 1455 ms at B = 4096; those rows are kept in the
raw CSVs and appear in `results/mpc_latency.csv` when `collect.py` is run
with `KEEP_LDL=1`.

Eager breakdown at B = 1: assembly + Ruiz (11 launches) 0.68 ms, factorization
0.53 ms, 25 iterations 3.47 ms (about 130 us per cuDSS solve). Graph replay
saves 0.6 ms; the cuDSS solves are the cost, not launch overhead.

CPU, single core (Ryzen 7 5800X3D), same QP: OSQP 5.7 ms per solve (2.8 ms
CasADi-interpreted matrix evaluation, 1.9 ms refactorization, 1.0 ms for 25
iterations; its solution is 1.3e-9 from the wip reference). A CPU run of the
wip's own monolithic functions through CasADi C codegen was attempted but is
not reported: gcc needs more than 20 GB of RAM for the 34 MB `ldl_factorize.c`
even at `-O0` (the 8x smaller `ldl_solve.c` took 4.6 GB at `-O1`), which is
itself a data point on the 1.28M-instruction symbolic LDL.

Observations:

- **cuDSS uniform-batch cliff.** With default settings, 25 solves cost 3.2 ms
  at B = 2, 6.1 ms at B = 12, and 58 ms at B = 16, then stay flat to B = 64.
  For B in [16, 64] splitting into sub-batches of 12 would be several times
  faster; not done here because the benchmark uses cuDSS defaults.
- **fp32 is not usable for this QP.** The KKT mixes sigma = 1e-6 with
  equality rho = 1e2; the fp32 pipeline lands 3e-2 to 9e-2 from the fp64
  solution and the wip's own fp32 symbolic LDL is off by 0.47. The fp32 rows
  in the CSV are throughput-only (B = 4096: 465 ms, 113 us per QP).
- The monolithic `KKT_matrix` and `ruiz_scaling` run in cusadi2's serial
  (thread-per-instance) mode because their working sets (542 and 330 KB) exceed
  shared memory; `KKT_vector` and `ADMM_step` fit and ran in the level kernel
  (see above). The staged functions all fit (largest 64 KB).
- cuDSS's default mode is not bitwise deterministic across batch members in
  its small-batch path (B < 16); every instance is therefore checked against
  the reference with a tolerance rather than for equality.

## Files

- `build_mpc_stage.py`: re-expresses one generic stage of the wip controller
  symbolically, verifies against the wip functions on the CPU, writes plans
  (`plans/`), `kernel_{double,float}.cu`, `csr.bin` (lower CSR + gather map),
  `inputs/` (padded per-stage inputs, reference dx). Needs `CUSADI_WIP` to
  point at a directory holding the vendored wip package (`wip_pkg/`, `stubs/`),
  the toe/heel URDF and `humanoid_mpc_fns/` (wip functions, `p_eval.npy`,
  `dx_custom.npy`).
- `mpc_cudss_bench.cu`: the driver. Build (once per dtype):
  ```
  nvcc -std=c++17 -O3 -arch=sm_89 -DCUSADI2_SCALAR=double -I<cusadi2 include> \
       -I$NV/include mpc_cudss_bench.cu <build>/kernel_double.cu \
       -L$NV/lib -l:libcudss.so.0 -Xlinker -rpath=$NV/lib -o mpc_cudss_double
  ./mpc_cudss_double <build> 3 10 1 4 16 64 256 1024 4096
  ```
  with `NV=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13`
  (`pip install nvidia-cudss-cu13`). `CUDSS_ALLOC=pool|arena|default` picks
  the memory handler.
- `mpc_mono_bench.cu`: the cusadi baseline driver (modes `cudss`,
  `cudss_serial`, `c2` for the symbolic LDL, `c2serial`, `orig`); same build line with
  `-DNO_UNROLLED` and the plans of the wip's six functions.
- `collect.py`: merges the driver CSVs into `results/mpc_latency.csv` and
  `results/mpc_latency.png`.
- `results/`: the numbers above.

## Caveats

- Toe and heel frames are not in the public MIT humanoid URDF; they were added
  as fixed frames at the foot cylinder's ends (see `../README.md`).
- The initial guess is x = 0 and the parameters are the wip defaults (nominal
  standing pose, all contacts active), the same point the wip references use.
- The monolithic baseline's fp32 plans are the fp64 plans (a plan written
  without `max_live` is dtype-independent; one windowed to a device's shared
  memory is not), with the kernel compiled for float.
