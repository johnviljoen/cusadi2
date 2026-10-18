# Safe predictive sampling on the Panda, MJPC style: 3 iterations x 256 CBF-filtered rollouts per 20 ms tick, best sample applied, no re-filter

`examples/panda_safe_mppi` (safe MPPI, the paper's key result) averages the K CBF-filtered rollouts and then has to
filter the applied torque once more. `examples/panda_safe_ps` replaced the average by the best sample so that the
feasibility established inside the rollouts reaches the robot unchanged, and stalled at the obstacle on every seed.
This directory re-does the best-sample controller the way MuJoCo MPC's sampling planner runs it (Howell et al. 2022,
Predictive Sampling, `mjpc/planners/sampling`): several sequential random-search iterations per control period on a
cubic 7-knot joint-target spline, the best candidate replacing the nominal each time, MJPC's resampling of the plan,
with the one-step CBF filter inside every rollout step and nothing else between the sampler and the robot. The
formulation is in `FORMULATION.md`; the driver is `safe_mppi.cu` (all iterations and the resample in one CUDA graph)
with `driver.py` on top.

Two results are reported, in the order they happened: the MJPC-faithful configuration (MJPC's own Panda settings,
nothing tuned) fails the task; the tuned configuration (John's go-ahead: "do the tuning, I want this working well with
only the CBF on the candidates") reaches almost twice as many waypoints per minute as the paper's MPPI at 20 ms per tick.

## 1. The MJPC-faithful run (phase 1): FAIL

| | MJPC (Panda task settings / defaults) | `panda_safe_ps` | phase 1 here |
|---|---|---|---|
| planner iterations per control period | as many as fit (asynchronous loop) | 1 | 4, fixed by the 20 ms budget |
| samples per iteration | 60 (`sampling_trajectories`) | 256 | 64 |
| horizon | 0.5 s (`agent_horizon`) | 2.0 s | 0.5 s (T = 25 steps of 20 ms) |
| spline | cubic Hermite, 7 knots, resampled evenly over [t, t + H] | linear, 4 knots, shifted | cubic Hermite, 7 knots, MJPC resample |
| noise std | 0.1 x half the control range on every knot | 0.1 rad | 0.1 x half the joint range (about 0.29 rad) |
| update | best of the N candidates replaces the nominal, nominal wins ties | same | same |

Gates (`verify_ps.py`, `results/verify_ps.json`) all pass: the CasADi spline weights inside the plans equal the numpy
reference and a direct cubic Hermite with MJPC's slopes (0 and 1e-15); the applied torque, gathered from step 0 of
the selected rollout, equals model + projection recomputed at the plant state bit for bit; within a tick the nominal's
cost at iteration i + 1 is bitwise the cost min of iteration i and never increases; the resample and projection kernels
match their numpy references to 1e-7 (fp32).

Closed loop (MuJoCo plant, 30 s, 50 Hz, 5 seeds, pass rule fixed in advance: every attempted waypoint reached within
6 s and barriers above -0.01 on every seed; `closed_loop.py --K 64 --T 25 --n-iter 4 --sigma mjpc --knot-scale "" --par ""`):

| seed | reached / timed out | min barrier | max joint speed | tick median / p99 |
|---:|---:|---:|---:|---:|
| 1 | 3 / 3 | -0.020 (obstacle) | 5.7 rad/s | 14.5 / 15.1 ms |
| 2 | 7 / 1 | -0.003 | 3.2 | 14.5 / 14.9 |
| 3 | 6 / 1 | -0.015 (floor) | 4.1 | 14.6 / 14.7 |
| 4 | 3 / 3 | -0.010 | 3.6 | 14.6 / 15.1 |
| 5 | 5 / 3 | -0.013 | 4.9 | 14.5 / 15.6 |

Why (`diagnose.py`): the 0.5 s horizon does not see the way round the sphere, so the arm parks 5-45 cm short of the far
waypoints; only about one iteration in four improves the nominal and on 40-50 % of the ticks none does, the gain within
a tick being cancelled by the rise between ticks; and MJPC's 0.29 rad noise on knot 0 through the kp = 50 PD gives torque
spikes, joint speeds of 3-6 rad/s and barrier dips to -0.02. Predictive sampling is jittery by construction (the plan
jumps by the noise scale whenever a different sample wins); MJPC's smooth demos come from per-task tuning, and its
planner runs far more iterations per period on a CPU than the 4 that the depth of this model graph allows here.

## 2. Budget: what fits in 20 ms (`timing_grid.py`, `results/timing_grid.json`, cusadi2, fp32)

| T (horizon) | K = 64 | K = 256 | K = 512 | K = 1024 |
|---|---:|---:|---:|---:|
| 25 (0.5 s) | 3.5 ms, 4 fit | 3.8 ms, 5 fit | 6.2 ms, 3 fit | 8.9 ms, 2 fit |
| 35 (0.7 s) | | 5.3 ms, 3 fit | | |
| 50 (1.0 s) | 7.1 ms, 2 fit | 7.4 ms, 2 fit | 12.4 ms, 1 fits | 17.8 ms, 1 fits |
| 100 (2.0 s) | 14.4 ms, 1 fits | 15.3 ms, 1 fits | 24.9 ms, none | 35.6 ms, none |

One iteration costs about T x 0.14 ms whatever the batch below 256 (the model call is latency-bound: 207 levels of a
30.6k-node graph), so iterations are paid in depth and samples up to 256 are free.

## 3. Tuning (`sweep.py`, grids 1-6, `results/closed_loop_ps_<tag>_seed<s>.json`, `logs/sweep_grid*.log`)

The structure was kept fixed (best sample applied directly, CBF filter only inside the rollouts, no re-filter, cusadi2,
one tick within 20 ms); horizon, sample count, iteration count, knot noise (absolute std and per-knot scale), gains and
cost weights were tuned. 46 configurations, 5 seeds each for the later grids, 30 s per run. What mattered, in order:

| lever | effect (reached / timed out per seed) |
|---|---|
| freezing knot 0 (no noise on the knot at the current time, so the applied torque stays continuous) | T 25, 5 iterations, 0.10 rad: 5.0 / 3.0 -> 10.0 / 0 on seeds 1-2; 7.8 / 1.2 on 5 seeds |
| iterations over samples | 5 x 256 (7.8 / 1.2) > 3 x 512 (6.2 / 1.4) > 2 x 1024 (5.2 / 2.2) at T 25 |
| noise | 0.07 rad 7.6 / 1.2, 0.10 rad 7.8 / 1.2, 0.15 rad 4.0 / 2.8, MJPC's 0.29 rad fails; ramping the noise over the knots does not help |
| horizon | T 35 with 3 iterations (9.2 / 1.6) > T 25 with 5 (7.8 / 1.2 but slower) > T 50 with 2, T 100 with 1 |
| gains and weights alone | kp 30: 8.8 / 1.0; w_pos 100: 10.2 / 2.0 with speed spikes; w_v 0.5 or 2 and w_rot 0 hurt; no-nominal sampling hurts |
| the combination T 35 x 3 iterations, kp 30, w_pos 100 | 21.6 / 0.2, 1.3 s mean to reach |
| kp 20 instead of 30 | 20.4 / 0.0 on every seed, min barrier -0.014 |
| around that point | w_pos 150: faster (24-26 / 0.2-0.4) but one seed with 2 timeouts; sigma 0.08, 96 projection iterations, w_u 1e-3, w_pos 80: zero timeouts, same -0.015 dip |

## 4. The tuned configuration: 3 iterations x 256 rollouts x 0.7 s, 0.10 rad noise with knot 0 frozen, kp 20, w_pos 100

`closed_loop.py` defaults (`driver.TUNED`). 30 s, 50 Hz, 5 seeds, nothing re-filtered:

| seed | reached / timed out | time to reach (mean) | min barrier (obstacle rows; all other groups > 0 except floor -0.001) | max joint speed | tick median / p99 |
|---:|---:|---:|---:|---:|---:|
| 1 | 14 / 0 | 2.0 s | -0.014 | 2.3 rad/s | 15.9 / 16.0 ms |
| 2 | 25 / 0 | 1.2 s | -0.009 | 2.5 | 15.3 / 16.0 |
| 3 | 26 / 0 | 1.1 s | -0.009 | 2.7 | 15.2 / 16.0 |
| 4 | 18 / 0 | 1.6 s | -0.006 | 2.6 | 15.9 / 16.1 |
| 5 | 19 / 0 | 1.6 s | -0.008 | 2.5 | 15.9 / 16.1 |

Against the other controllers on the same task and seed 1 (30 s):

| controller | reached / timed out | time to reach | min barrier | tick |
|---|---:|---:|---:|---:|
| this: safe predictive sampling, MJPC style, tuned (3 x 256 x T 35, best sample, no re-filter) | 14 / 0 (20.4 / 0 over 5 seeds) | 2.0 s (1.5 s) | -0.014 | 15.9 ms |
| safe MPPI, the paper's key result (K 256, T 100, averaged, re-filtered) | 12 / 0 | 2.5 s | -0.002 | 35 ms |
| one-iteration best sample, `panda_safe_ps` (K 256, T 100) | 12 / 1 (stalls on every seed) | 1.9 s | -0.006 | 15 ms |
| MJPC-faithful, phase 1 above | 3 / 3 | 2.6 s | -0.020 | 14.5 ms |
| frax's filtered OSC (OSC + one-step CBF-QP, qpax) | 23 / 0 | 1.3 s | 0.004 | 3.8 ms |

Every attempted waypoint is reached on every seed, in 1.1-2.0 s on average, at 20 ms per tick with the CBF filter only
inside the rollouts. The one part of the pass rule not met is the barrier floor: the obstacle rows dip to -0.014 on one
seed (1.4 cm into the padded collision margin; the other groups stay positive). Doubling the projection iterations does
not change it, so it is not the filter failing to converge but the one-step CBF condition on a 20 ms step against a 2 ms
plant at 2.5 rad/s; the MPPI run, slower (2.5 s per waypoint) and re-filtered, stayed at -0.002. Raising w_pos buys
speed and loses completion on one seed; lowering it or the noise buys nothing. This is where the tuning stopped.

Video: `results/closed_loop_ps.mp4` (seed 1, tuned; yellow = the final nominal's rollout, blue = 31 of the last
iteration's samples shaded by cost, green = end-effector trail, overlay with the per-iteration time and how many of the
3 iterations improved the plan).

## Files

- `model.py` CasADi model and integrate functions, MJPC's cubic spline as knot weights (`interp_weights`, shared with
  the JAX twins), numpy references `spline_np`, `resample_np`, `project_greedy_np`.
- `safe_mppi.cu` / `driver.py` the native driver: `n_iter` iterations per tick, per-knot noise, `k_resample`,
  per-iteration statistics (`iter_stats`), `time_iteration`; `build.py` (plans, lib, exports, unrolled, cpu).
- `verify_ps.py` the gates; `timing_grid.py` the budget table; `sweep.py` the tuning grids (`run_sweep.sh` detached);
  `closed_loop.py` (`--sigma mjpc --knot-scale "" --par "" --K 64 --T 25 --n-iter 4` reproduces phase 1), `diagnose.py`,
  `summarize.py`, `smoke.py`, `run_phase1.sh`.
- `verify_evaluators.py`, `bench.py` (per-iteration timing, K and T sweeps), `frax_model.py`, `mjx_model.py` (JAX twins
  with the same spline, not yet re-exported): the all-evaluator benchmark of phase 2.
- `results/` closed_loop_ps.{json,npz,mp4} (tuned, seed 1), closed_loop_ps_seed2..5 (phase 1, MJPC-faithful),
  closed_loop_ps_<tag>_seed<s> (tuning grids), verify_ps.json, timing_grid.json; `logs/`.
