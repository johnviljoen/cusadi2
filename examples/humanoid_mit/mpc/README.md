# MIT humanoid MPC QP: stage-parallel cusadi2 + cuDSS

Whole-QP latency benchmark for the wip-branch ("CusADi 2.0") MIT humanoid MPC
formulation, solved with cusadi2's stage axis for the per-stage work and cuDSS
for the batched block-tridiagonal KKT solve, and a receding-horizon rollout of
that controller in closed loop on the GPU. The benchmark is compared against
the same QP with the wip's monolithic whole-horizon CasADi functions in the
original CusADi execution model (one thread per instance) feeding the same
cuDSS solve, and against CPU solves.

## Problem

The full formulation, QP, Ruiz scaling, ADMM iteration, the staged
decomposition and the receding-horizon loop are written out in
`FORMULATION.md`.

10-stage MPC over the MIT humanoid (24 dof + 12 contact forces): 60 primal
variables and 74 constraints per stage, KKT of 1340 rows with 11,162 lower
nonzeros. OSQP-style ADMM (rho 0.1, sigma 1e-6, alpha 1.6, 2 Ruiz passes,
25 iterations, one SQP step at x = 0), the wip `admm_backend`.

One deliberate deviation from the wip: its left-leg length row adds the hip
position instead of subtracting it, which bounds the robot's distance from the
world origin rather than its leg length (FORMULATION.md, section 5). The sign
is corrected here, both in the staged functions and in the vendored wip
controller the references are generated from, so "the wip" below means the wip
with that one-line patch.

## Pipeline (one solve, one CUDA graph)

| step | cusadi2 function | grid | notes |
|---|---|---|---|
| assemble | `mpc_assemble` (77,707 instr, 175 levels, 64 KB smem) | B x 10 | KKT block row `[P_kk, P_{k-1,k}, A_kk, A_{k,k-1}]`, gradient, bounds; reads x_{k-1}, x_k, x_{k+1} and dt_k, dt_{k+1} through coupled inputs |
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
- staged pipeline on the CPU (wip symbolic LDL) vs the wip CPU path
  (`solve_with_custom`): 2.5e-13 max abs error
- full GPU solve vs the same reference: 6.6e-11 to 1.8e-10 over all B (cuDSS
  orders the factorization differently)
- OSQP itself on the same QP gives 3.7e-12 vs the same reference

## Results (RTX 4080, cuDSS 0.8.0, CUDA 13.3), fp64, CUDA graph

"cusadi2" is the staged pipeline above. "cusadi" is the wip's monolithic
whole-horizon `KKT_matrix`, `ruiz_scaling`, `KKT_vector` and `ADMM_step`
functions, every one in cusadi2's serial (thread-per-instance) kernel, i.e. the
original CusADi execution model, with the same cuDSS uniform-batch
factorization and solves (driver `mpc_mono_bench.cu`, mode `cudss_serial`).
"hybrid" is mode `cudss`: the same functions, but the two that fit shared
memory (`KKT_vector`, `ADMM_step`) run in the level kernel. The solver is
identical in all columns.

| B | cusadi2 latency | cusadi2 per QP | cusadi2 QP/s | cusadi latency | cusadi per QP | cusadi QP/s | ratio | hybrid latency |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.68 ms | 3.68 ms | 272 | 274 ms | 274 ms | 4 | 74x | 166 ms |
| 4 | 4.51 ms | 1.13 ms | 888 | 278 ms | 69.5 ms | 14 | 62x | 169 ms |
| 12 | 7.57 ms | 0.63 ms | 1585 | 284 ms | 23.7 ms | 42 | 38x | 174 ms |
| 16 | 67.5 ms | 4.22 ms | 237 | 345 ms | 21.6 ms | 46 | 5.1x | 235 ms |
| 64 | 68.5 ms | 1.07 ms | 934 | 382 ms | 5.96 ms | 168 | 5.6x | 261 ms |
| 256 | 89.2 ms | 0.35 ms | 2871 | 482 ms | 1.88 ms | 531 | 5.4x | 306 ms |
| 1024 | 176 ms | 0.17 ms | 5805 | 545 ms | 0.53 ms | 1880 | 3.1x | 378 ms |
| 4096 | 627 ms | 0.15 ms | 6533 | 902 ms | 0.22 ms | 4540 | 1.4x | 750 ms |

Latency is one batched solve (median of graph replays); throughput is B
divided by that latency, so the latency and throughput ratios coincide (ratio
is cusadi / cusadi2). All columns are correct to 2e-10 or better against the
wip CPU reference at every B. Measured 2026-10-01 with the corrected leg row
and the current runtime; the table of 2026-09-07 (before the correction, mode
`cudss` only) read 4.06 ms and 129.7 ms at B = 1.

In the cusadi column the whole-horizon assembly and Ruiz functions cost
163 ms at B = 1 (111 + 52 ms, one thread walking 883k instructions) and
227 ms at B = 4096; the staged cusadi2 versions cost 0.31 ms and 107 ms. Its
25 iterations cost 111 ms at B = 1 in the serial kernel against 3.3 ms in the
level kernel (the hybrid column), so the stage axis and the level kernel each
buy about two orders of magnitude of latency at small batch, and at full
occupancy the thread-per-instance model closes most of the gap. The wip's
original symbolic LDL solver (a 1.28M-instruction DAG in place of cuDSS) made
the cusadi column 872 ms at B = 1 and 1455 ms at B = 4096 on 2026-09-07; it
has not been re-measured and appears in `results/mpc_latency.csv` only when
its CSVs are present and `collect.py` is run with `KEEP_LDL=1`.

Eager breakdown at B = 1: assembly + Ruiz (11 launches) 0.31 ms, factorization
0.48 ms, 25 iterations 3.05 ms (about 120 us per cuDSS solve). Graph replay
saves 0.17 ms; the cuDSS solves are the cost, not launch overhead.

CPU, single core (Ryzen 7 5800X3D), measured 2026-09-07 before the leg-row
correction and not repeated: OSQP 5.7 ms per solve (2.8 ms CasADi-interpreted
matrix evaluation, 1.9 ms refactorization, 1.0 ms for 25 iterations). A CPU
run of the wip's own monolithic functions through CasADi C codegen was
attempted but is not reported: gcc needs more than 20 GB of RAM for the 34 MB
`ldl_factorize.c` even at `-O0` (the 8x smaller `ldl_solve.c` took 4.6 GB at
`-O1`), which is itself a data point on the 1.28M-instruction symbolic LDL.

Observations:

- **MJX pays for the assembly, not the solve.** See the next section: with
  everything but the assembly held fixed, the stage derivatives in JAX on MJX
  cost 9x to 14x the cusadi2 assembly, and the whole solve 1.1x to 2.5x.
- **cuDSS uniform-batch cliff.** With default settings, 25 solves cost 3.7 ms
  at B = 4, 6.1 ms at B = 12, and 58 ms at B = 16, then stay flat to B = 64.
  For B in [16, 64] splitting into sub-batches of 12 would be several times
  faster; not done here because the benchmark uses cuDSS defaults.
- **fp32 is not usable for this QP.** The KKT mixes sigma = 1e-6 with
  equality rho = 1e2; the fp32 pipeline lands 3e-2 to 8e-2 from the fp64
  solution (the cusadi baseline in fp32: 7e-3 to 2e-2). The fp32 rows in the
  CSV are throughput-only (B = 4096: 440 ms, 107 us per QP).
- The monolithic `KKT_matrix` and `ruiz_scaling` cannot run in the level
  kernel at all: their working sets (555 and 338 KB) exceed shared memory. The
  staged functions all fit (largest 64 KB).
- cuDSS's default mode is not bitwise deterministic across batch members in
  its small-batch path (B < 16); every instance is therefore checked against
  the reference with a tolerance rather than for equality.

## MJX baseline through the solver bridge

The same QP with the assembly written in JAX on MuJoCo MJX (`mjx_stage.py`:
MJX kinematics and RNE, the SE(3) operations in JAX, forward-mode autodiff for
the constraint Jacobians and the cost Hessian), handed on the device to the
**same** CUDA solver. `mpc_qp_lib.cu` is that solver as a shared library with
a C interface (Ruiz plans, gather, cuDSS factorization, 25 iterations, one
CUDA graph); `mpc_bridge.py` passes it the device pointers of any array with
`__cuda_array_interface__` (JAX, Warp, CuPy, PyTorch), so nothing goes through
the host. The MJX output is gathered into `mpc_assemble`'s nonzero layout, so
the solver cannot tell the two assemblies apart.

| B | cusadi2 assembly | MJX assembly | ratio | solver (both) | cusadi2 total | MJX total | ratio | solver share: cusadi2 | MJX |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.19 ms | 1.76 ms | 9.2x | 4.0 ms | 4.21 ms | 5.83 ms | 1.4x | 95% | 70% |
| 4 | 0.19 ms | 2.65 ms | 14x | 5.0 ms | 5.13 ms | 7.73 ms | 1.5x | 96% | 66% |
| 12 | 0.35 ms | 4.97 ms | 14x | 8.2 ms | 8.52 ms | 13.3 ms | 1.6x | 96% | 63% |
| 16 | 0.47 ms | 5.30 ms | 11x | 67.3 ms | 67.7 ms | 72.7 ms | 1.1x | 99% | 93% |
| 64 | 1.37 ms | 16.0 ms | 12x | 67.5 ms | 68.8 ms | 83.6 ms | 1.2x | 98% | 81% |
| 256 | 5.06 ms | 62.6 ms | 12x | 84.4 ms | 89.3 ms | 147 ms | 1.6x | 94% | 57% |
| 1024 | 19.9 ms | 254 ms | 13x | 157 ms | 177 ms | 411 ms | 2.3x | 89% | 38% |
| 4096 | 80.2 ms | 1016 ms | 13x | 548 ms | 628 ms | 1564 ms | 2.5x | 87% | 35% |

fp64, RTX 4080, `bridge_bench.py`: both systems run through one harness and
one solver instance, so the only thing that changes between the two columns is
who evaluates the stage derivatives. "Assembly" is that evaluation alone (the
cusadi2 plan `mpc_assemble` in one launch, or one jitted XLA call); "solver" is
Ruiz, gather, factorization and the 25 iterations for both, and differs by
under 2% between the two runs. Median of 10 after 3 warm-up solves, wall clock
around synchronous calls. These cusadi2 totals are 0.5 ms (B = 1) to 1 ms
(B = 12) above the single-graph latencies of the first table, which is the
cost of the bridge (below); from B = 16 up the two agree within 1%.

The whole solve is close between the two because the solver dominates it: 87%
to 99% of the cusadi2 solve is Ruiz, cuDSS and ADMM, so a 9x to 14x slower
assembly moves the total by 1.1x to 2.5x. Between B = 16 and 64 the cuDSS
cliff hides the assembly almost entirely.

- Correctness: the MJX stage constraints match the CasADi ones to 3e-14 at the
  benchmark point and to 1e-7 at states from the rollout; the assembled
  Jacobian entries to 3e-7 on a scale of 1e3; the solved step is 3.6e-10 from
  the wip reference at every B (cusadi2: 6.6e-11 to 1.8e-10).
- Compilation: XLA takes 31 to 48 s per batch size (the function is retraced
  for every B); the cusadi2 plans are batch-independent and build in under a
  second.
- Memory: forward-mode AD through MJX carries 120 tangents of every
  intermediate; unchunked it needs 23.7 GB at B = 1024 and 97 GB at B = 4096
  and fails on the 16 GB card. `mjx_stage.py` therefore evaluates 256 instances
  at a time above B = 256 (`lax.map`), which is what the last two rows measure.
- Bridge overhead: two synchronous calls instead of one graph (device copies
  of the inputs, a Python call and a stream synchronisation each) cost the
  cusadi2 solve 0.5 ms at B = 1 and 1 ms at B = 12; both systems pay it
  equally here.
- mujoco-warp: the bridge accepts Warp arrays (`python mpc_bridge.py <build>`
  solves the benchmark QP from `wp.array`s, 7.6e-11 from the reference), but
  there is no Warp assembly. mujoco-warp 3.12 is fp32 only and compiled with
  `enable_backward=False` throughout; its only analytical derivatives are the
  body Jacobian and d(RNE)/d(velocity). The constraint Jacobian here needs
  d(RNE)/dq and d(J^T F)/dq, so a Warp version would have to finite-difference
  in fp32, on a QP for which fp32 is already unusable.

## Receding horizon (`mpc_rollout.cu`)

12 robots in closed loop for 4 s against the MPC's own model, with per-instance
commands (forward speed 0 to 0.6 m/s, yaw rate -0.5, 0, +0.5 rad/s) and a
time-based walking gait. One control step is one stage of the plan
(dt_0 = 20.92 ms) and one real-time iteration:

1. solve the staged QP above, linearized at the current plan instead of at rest;
2. take the SQP step, x+ = x + dx;
3. `mpc_advance`: stage 0 of x+ becomes the new state (q_0, v_0);
4. `mpc_shift`: re-sample x+ one control period later on the geometric grid
   and re-base it onto the new q_0; this is the next linearization point.

The references, contact schedule and swing phase depend on time only, so
`gen_batch12.py` precomputes them for all 192 steps and they are uploaded
once; the step itself (the solve, one small kernel, two more cusadi2 launches
and two copies) is captured into one CUDA graph and replayed.

Results (B = 12, fp64, RTX 4080; `rollout_report.py`, `results/rollout_report.csv`):

- 192 steps in 1.47 s of wall time, 2.7x real time; step latency 7.6 ms
  median, 8.5 ms worst, against a 20.9 ms control period.
- All 12 stay upright for the whole rollout: base height 0.63 to 0.65 m, tilt
  below 0.04 rad.
- Tracking after the command ramp: forward speed within 3% of the command
  (0.588 for 0.6 m/s), yaw rate within 0.5%.
- Residuals of the executed stage, nonlinear, typical median / worst after
  the first 10 steps: integration 3e-4 / 1.5e-3, floating-base dynamics 0.1 / 20 N,
  swing height 3e-5 / 2.4e-3 m, stance-foot velocity 1e-5 / 0.09 m/s,
  inequalities 1e-2 / 0.06. The first steps are larger (dynamics up to 46 N):
  the plan starts from x = 0.
- The GPU rollout matches a CPU run of the same loop (staged CasADi functions,
  SuperLU for the KKT) to 6e-11 in q over all 193 states.

What it took, and what it did not:

- Linearizing at rest every step, as the benchmark does, diverges within the
  rollout; the shifted plan as the linearization point is what makes it work.
- With the wip's left-leg row as coded the robots cannot leave the origin
  (0.5 m covered at 0.6 m/s with 25 ADMM iterations, 0.2 m with 400); this is
  what prompted the correction.
- Solver effort is the benchmark's: 25 ADMM iterations, one SQP step, cold
  ADMM start. 50, 100 and 400 iterations change the travelled distance by about
  1% and do not lower the residuals, so the ADMM duals are not warm-started;
  a second SQP step per control step lowers the dynamics residual about 8x.

`results/batch12_closed_loop.mp4` shows the rollout, one frame per control
step at 47.79 fps (= 1 / dt_0, real time), rows by yaw rate and columns by
forward speed; `results/batch12_closed_loop.png` is a contact sheet at
t = 0, T/3, 2T/3, T.

## Files

- `vendor_setup.py`: fetches what lives outside this repo into `vendor/`
  (gitignored): the wip branch of se-hwan/cusadi (patched as above), the MIT
  humanoid URDF and meshes from hojae-io/LearningHumanoidArmMotion-RAL2025-Code
  with toe and heel frames added, import stubs, and the wip's reference
  functions (`p_eval.npy`, `dx_custom.npy`). Run it once.
- `build_mpc_stage.py`: re-expresses one generic stage of the wip controller
  symbolically, verifies against the wip functions on the CPU, writes plans
  (`plans/`), `kernel_{double,float}.cu`, `csr.bin` (lower CSR + gather map),
  `inputs/` (padded per-stage inputs, reference dx), and the functions
  themselves (`fns/`). `CUSADI_WIP` overrides the vendor directory.
- `mpc_cudss_bench.cu`: the benchmark driver. Build (once per dtype):
  ```
  nvcc -std=c++17 -O3 -arch=sm_89 -DCUSADI2_SCALAR=double -I<cusadi2 include> \
       -I$NV/include mpc_cudss_bench.cu <build>/kernel_double.cu \
       -L$NV/lib -l:libcudss.so.0 -Xlinker -rpath=$NV/lib -o mpc_cudss_double
  ./mpc_cudss_double <build> 3 10 1 4 12 16 64 256 1024 4096
  ```
  with `NV=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13`
  (`pip install nvidia-cudss-cu13`). `CUDSS_ALLOC=pool|arena|default` picks
  the memory handler.
- `build_mpc_mono.py`, `mpc_mono_bench.cu`: plans for the wip's six functions
  and the cusadi baseline driver (modes `cudss_serial`, `cudss`, `c2` for the
  symbolic LDL, `c2serial`, `orig`); same build line with `-DNO_UNROLLED`.
- `mjx_stage.py`: the stage and its assembly in JAX on MJX.
- `bridge_bench.py`: the assembly-swap sweep, cusadi2 and MJX through the same
  harness and solver (`python bridge_bench.py <build> 3 10 1 4 ...`, CSV to
  `<staged_dir>/bridge_double.csv`).
- `mpc_qp_lib.cu`, `mpc_bridge.py`: the QP solver and the cusadi2 assembly as
  a shared library (add
  `-shared -Xcompiler -fPIC`, output `<build>/libmpc_qp_double.so`) and its
  Python bridge.
- `collect.py`: merges the driver CSVs into `results/mpc_latency.csv` and
  `results/mpc_latency.png`.
- `gen_batch12.py`: commands, gait and references for the rollout
  (`<build>/rollout/p_stage.f64`).
- `mpc_rollout.cu`: the closed-loop driver; same build line as the benchmark,
  then `./mpc_rollout_double <build> 12`. Writes the states, every step's plan
  and the step latencies to `<build>/rollout/`.
- `rollout_report.py`: tracking, residual and latency report for a rollout.
- `render_batch12.py`: headless MuJoCo render of the rollout (needs `ffmpeg`).
- `results/`: the numbers above.

## Caveats

- Toe and heel frames are not in the public MIT humanoid URDF; they are added
  as fixed frames at the foot cylinder's ends (`vendor_setup.py`).
- The benchmark's initial guess is x = 0 and its parameters are the wip
  defaults (nominal standing pose, all contacts active), the same point the
  wip references use.
- The rollout is a nominal closed loop: the plant is the MPC's own model
  (stage 0 of the plan becomes the next state), with no contact physics, model
  mismatch or disturbance. It shows that the controller re-plans stably at
  this solver effort, not that it would balance a simulated or real robot.
- The monolithic baseline's fp32 plans are the fp64 plans (a plan written
  without `max_live` is dtype-independent; one windowed to a device's shared
  memory is not), with the kernel compiled for float.
