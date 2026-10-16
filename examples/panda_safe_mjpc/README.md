# MJPC-faithful safe predictive sampling on the Panda: phase 1 (cusadi2 only), closed loop FAILS the pass rule

`examples/panda_safe_mppi` (safe MPPI, the paper's key result) averages the K CBF-filtered rollouts and then has to
filter the applied torque once more. `examples/panda_safe_ps` replaced the average by the best sample so that the
feasibility established inside the rollouts reaches the robot unchanged, and stalled at the obstacle on every seed.
This directory re-does the best-sample controller exactly as MuJoCo MPC's sampling planner runs it (Howell et al. 2022,
Predictive Sampling, `mjpc/planners/sampling`), which is what the one-iteration variant was not:

| | MJPC (Panda task settings / defaults) | `panda_safe_ps` | this example |
|---|---|---|---|
| planner iterations per control period | as many as fit (asynchronous loop) | 1 | `n_iter` = 4, fixed by the 20 ms budget |
| samples per iteration | 60 (`sampling_trajectories`) | 256 | 64 |
| horizon | 0.5 s (`agent_horizon`) | 2.0 s | 0.5 s (T = 25 steps of 20 ms) |
| spline | cubic Hermite, 7 knots, resampled evenly over [t, t + H] | linear, 4 knots, shifted | cubic Hermite, 7 knots, MJPC resample |
| noise std | 0.1 x half the control range on every knot | 0.1 rad | 0.1 x half the joint range (about 0.29 rad) |
| update | best of the N candidates replaces the nominal, nominal wins ties | same | same |

The CBF safety filter inside every rollout step, the inertia-shaped PD nominal controller, the cost and the task are
those of the MPPI example. The formulation is in `FORMULATION.md`; the driver is `safe_mppi.cu` (all `n_iter`
iterations and the resample in one CUDA graph) with `driver.py` on top. All parameters were frozen before the runs
(MJPC's values where it has one, the MPPI example's otherwise); only `n_iter` was set by measurement. fp32 throughout.

## Gates (`verify_ps.py`, `results/verify_ps.json`): PASS

| gate | result |
|---|---|
| CasADi spline weights (inside the plans) vs numpy reference vs a direct cubic Hermite with MJPC's slopes | 0 and 1.3e-15 |
| applied torque = gathered step-0 filtered torque of the selected rollout vs model + projection recomputed at the plant state | 0 (bitwise) |
| within a tick: cost of the nominal at iteration i + 1 = cost min of iteration i; cost min non-increasing | 0 (bitwise), true on every tick |
| resample kernel (`k_resample`) vs `model.resample_np` of the winning sample | 1.1e-7 (fp32) |
| projection kernel vs numpy reference | 1.8e-7 (fp32) |

## Timing (cusadi2 evaluator, K = 64, T = 25, model plant, `smoke.py`)

| | median | p99 |
|---|---:|---:|
| one iteration (sample, init, 25-step rollout, select) | 3.6 ms | 4.0 ms |
| model call (64 instances) | 0.090 ms | |
| integrate call | 0.045 ms | |
| projection | 0.003 ms | |
| tick with n_iter = 4 | 14.2 ms | 15.2 ms |
| tick with n_iter = 5 | 18.0 ms | 18.1 ms |

The rule fixed in advance, `n_iter = floor(20 ms / p99 iteration time)` = floor(20 / 4.04) = 4. (Five iterations
would also fit the whole-tick budget at p99 18.1 ms.) At K = 64 the model call is latency-bound, 207 levels of a
30.6k-node graph on 64 blocks; a 25-step rollout step costs 0.14 ms whatever the batch below a few hundred, so an
iteration with K = 256 would cost about the same as one with 64. That is the reason only 4 iterations fit where MJPC's
CPU planner runs many: the price of an iteration here is its depth, not its width.

## Closed loop (MuJoCo plant, 30 s, 50 Hz, 5 seeds, `closed_loop.py`): FAIL

Pass rule fixed in advance: every attempted waypoint reached within 6 s, barriers above -0.01, on all five seeds.

| seed | reached / timed out | time to reach (mean, max) | min barrier | max joint speed | tick median / p99 | improving iterations per tick (of 4) | ticks with no improvement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (video) | 3 / 3 | 2.6 s, 5.4 s | -0.020 (obstacle) | 5.7 rad/s | 14.5 / 15.1 ms | 0.98 | 44 % |
| 2 | 7 / 1 | 3.1 s, 5.2 s | -0.003 | 3.2 | 14.5 / 14.9 | 0.75 | 52 % |
| 3 | 6 / 1 | 3.4 s, 5.5 s | -0.015 (floor) | 4.1 | 14.6 / 14.7 | 0.83 | 52 % |
| 4 | 3 / 3 | 3.0 s, 4.4 s | -0.010 | 3.6 | 14.6 / 15.1 | 0.99 | 46 % |
| 5 | 5 / 3 | 2.0 s, 3.5 s | -0.013 | 4.9 | 14.5 / 15.6 | 1.10 | 38 % |

For comparison on the same task, seed 1: safe MPPI (K = 256, T = 100, averaged, re-filtered) 12 / 0, 2.5 s mean,
min barrier -0.002; one-iteration predictive sampling (K = 256, T = 100, best sample) 12 / 1 (seeds 2-5: 12/1, 6/3,
8/2, 7/1), min barrier -0.006; frax's filtered OSC 23 / 0.

What the logs say (`diagnose.py`):

- **Timeouts are stalls short of the waypoint, 5-45 cm away**, mostly on the lower waypoints and the return legs
  around the sphere: with a 0.5 s horizon the rollouts do not see the way round the obstacle, and the sequential
  search has nowhere to go (the last frame of the video, t = 20 s, shows the arm parked above the sphere with the plan
  cost unchanged, 1.5 of 4 iterations "improving" by amounts that do not move it).
- **The search mostly treads water.** About one iteration in four improves the nominal, and on 40-50 % of the ticks
  none does; the improvement found within a tick (plan cost 4.7 -> 3.7 on seed 1) is on average cancelled by the rise
  between ticks (the plant step plus the resample of a plan whose knot 0 was perturbed by 0.29 rad, 0.7 of ticks rise,
  mean rise 1.0). MJPC's remedy is many more iterations; four is what the depth of the model graph allows at 20 ms.
- **MJPC's noise scale breaks the barrier margin.** Perturbing every knot, knot 0 included, by 0.29 rad through a
  kp = 50 PD gives torque spikes (|u| p90 67 N m against a median of 36) and joint speeds of 3-5.7 rad/s; at those
  speeds the one-step CBF rows plus the greedy 48-iteration projection let the obstacle barrier dip to -0.02 (seed 1,
  t = 19.5-19.7 s, |v| 4.1 rad/s) and the floor barrier to -0.015 (seed 3). The MPPI and one-iteration runs, with
  0.1 rad noise, stayed above -0.006.

Nothing was tuned. The result stands as a negative one for the MJPC recipe on this task at this compute budget;
the decision on what to do next (report it as is, change the frozen parameters, or stop here) is the author's.

Video: `results/closed_loop_ps.mp4` (seed 1; yellow = the final nominal's rollout, blue = 31 of the last iteration's
64 samples shaded by cost, green = end-effector trail; the overlay title of this render runs past the box and was
shortened in the script afterwards).

## Files

- `model.py` CasADi model and integrate functions, MJPC's cubic spline as knot weights (`interp_weights`, shared with
  the JAX twins), numpy references `spline_np`, `resample_np`, `project_greedy_np`.
- `safe_mppi.cu` / `driver.py` the native driver: `n_iter` iterations per tick, MJPC noise, `k_resample`, per-iteration
  statistics (`iter_stats`), `time_iteration`; `build.py` (plans, lib, exports, unrolled, cpu).
- `verify_ps.py` the gates above; `verify_evaluators.py`, `bench.py` (per-iteration timing, K and T sweeps) for phase 2.
- `closed_loop.py`, `diagnose.py`, `summarize.py`, `smoke.py`, `run_phase1.sh`.
- `frax_model.py`, `mjx_model.py` the JAX twins with the same spline (not yet re-exported: phase 2).
- `results/` closed_loop_ps{,_seed2..5}.{json,npz}, closed_loop_ps.mp4, verify_ps.json; `logs/`.
