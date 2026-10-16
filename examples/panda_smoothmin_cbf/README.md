# Merged smooth-min CBF for the Panda safe filter (study only)

Question: can the 64 relative-degree-2 barrier rows of the Panda safe-MPPI demo (`examples/panda_safe_mppi`) be
merged into **one** barrier, H(q) = -(1/kappa) log sum_i exp(-kappa s_i h2_i(q)), so that the per-rollout safety filter is a
single half-space projection inside the torque box (closed form, no greedy iterations, no QP), without losing what the
greedy-48 kernel and cbfpy's elastic QP achieve on the rollout metric?

The study reuses panda_safe_mppi's frax + cbfpy twin, test states and metrics (`projection_study.py` there) and does not
modify anything in that directory: `safe_mppi.cu`, the plans and the committed results are untouched. The HOCBF row of H is
built with cbfpy's cascade (h = dH/dt + alpha H, alpha = 10, L_f h and L_g h by forward-mode jvp like cbfpy's rows), the
filter is the exact projection of u_nom onto {a u <= b} ∩ box by bisection on lambda, and the sweep is
kappa in {25, 50, 100, 200, 400, 800} at dt = 20 ms. The singularity row gets the scale s = 4 (joint limits and spheres s = 1);
the choice and the success criterion fixed before the run are in the docstring of `projection_study.py` and in
`results/report.md`.

| file | what |
|---|---|
| `projection_study.py` | the study: baselines re-run (box only, greedy 48, elastic QP) + the smooth-min sweep -> `results/projection_study_smoothmin.json` |
| `report.py` | `results/report.md`: the table next to the existing rows, the composite diagnostics, the verdict |
| `trace.py` | step-by-step trace of the failing rollouts -> `results/trace.txt` |
| `logs/` | the runs |

Run (cusadi2 conda env, GPU, fp64): `python projection_study.py && python report.py`.

## Result (2026-10-13): FAIL, with the reason

`results/report.md` has the tables; `results/trace.txt` the step-by-step traces of the failing rollouts. The success
criterion fixed before the run (some kappa with 0 % of rollouts below -0.01, worst h2 >= -0.0045, mean |u - u_nom| on the
random states <= 1.5x the elastic QP's) is met by no kappa, with or without capping the multiplier at cbfpy's penalty:

| | rollouts below -0.01 | worst h2 | mean \|u - u_nom\| vs QP |
|---|---:|---:|---:|
| greedy 48 (the kernel) / elastic QP | 0.0 % | -0.0045 | 0.91x / 1.00x |
| smooth-min, best kappa (50) | 0.3 % (one rollout, diverged) | -0.0027 over the rest | 0.70x |
| smooth-min, kappa 100-400 | 0.9-1.5 % | -0.27 to -0.45 | 0.62x |
| smooth-min, kappa 25 or 800 | 2.3-4.1 % | -0.60 to -0.75 | 0.62-0.89x |

The composite is cheaper than the QP everywhere (0.6-0.9x the deviation) because it enforces one row where the QP enforces
64: after the filter 31-35 % of the random states still violate an original row (greedy 48: 15 %, QP: 2.5 %). What breaks the
rollouts is not cancellation between opposing barriers (rho stays above 0.13, and |a| ~ 0 with a violated row occurs on
0.15 % of states at most) but which barrier the softmin hands the row to: the barrier with the smallest scaled value is often
one the torque can barely move (a floor sphere on a base link, the singularity barrier with |dh/dq| 0.07 /rad), so |a| of the
composite falls below 0.05 on 11 % (kappa 50) to 37 % (kappa 800) of the rollout steps, and the joint-limit or obstacle row
that is actually approaching is invisible until the weights switch, by which time b is -20 to -100 and the row is infeasible
inside the box. The exact projection then runs lambda to saturation (every joint with a non-zero a_j gets full torque, wrist
joints reach 50-100 rad/s within a few steps, the explicit Euler plant diverges); the capped variant limits that to a 100-190
N m kick along a near-zero direction instead, and the state crashes through a joint limit a few steps later (first barrier
below -0.01 in the 53 non-diverged failing rollouts of both sweeps: a joint limit in 50, an obstacle sphere in 3). Small kappa does not help: at kappa = 25 the offset
H >= min - log(64)/kappa = min - 0.166 makes 20 % of the initially safe states count as unsafe already and b is more negative
still. The singularity scale (s = 4 here; 14.6 would match the joint-limit rows) moves the weights between these two failure
modes but not out of them.
