# MJPC-faithful safe predictive sampling on the Panda: the formulation

One Franka Panda, torque controlled at 50 Hz, reaching a sequence of end-effector waypoints around a spherical
obstacle under the 64 control-barrier functions of frax's OSCBF demo (joint limits, singularity, obstacle, self-collision,
floor). The controller is predictive sampling exactly as MuJoCo MPC's sampling planner runs it (Howell et al. 2022,
`mjpc/planners/sampling`): a nominal joint-target plan on a cubic spline is improved by sequential random-search
iterations, each the best of K noisy copies of the nominal rolled out over the horizon, with no averaging anywhere.
Every rollout step passes its torque through the closed-form CBF safety filter, so every sampled trajectory is safe
over the whole horizon, the best one becomes the nominal, and the torque sent to the robot is the first filtered torque
of the final nominal's rollout. Nothing is averaged and nothing is re-filtered. There is no QP solver and no linear solve
anywhere except the 7 x 7 Cholesky of the forward dynamics.

This example is `examples/panda_safe_ps` (one iteration of best-sample predictive sampling per tick, linear 4-knot
spline) with the four things MJPC does differently: several planner iterations per control period, MJPC's cubic spline
with 7 knots and its resampling rule, MJPC's noise scaling, and MJPC's Panda task sizes (K = 64 of its 60, 0.5 s
horizon). It runs in fp32 only.

## 1. Model (the evaluator)

State `x = [q (7), v (7), tau, c]`: joint positions and velocities, the plan clock and the accumulated cost. Two pure
model functions are evaluated per rollout step, each by one of five interchangeable evaluators: cusadi2 (CasADi graph,
`model.py`, level-parallel plans), frax + cbfpy (JAX, `frax_model.py`), MuJoCo XLA + cbfpy (JAX, `mjx_model.py`), the
original CusADi's unrolled kernels of the CasADi graph, or CasADi's C code generation of it on the CPU:

    model(x, Y, rp, par)      -> (u_nom (7), G (64 x 7), h (64))
    integrate(x, u, rp, par)  -> x1

**Plan.** `Y` holds N_KNOT = 7 knots of 7 joint targets at times j * knot_dt, knot_dt = (T - 1) dt / 6 (MJPC's
`UpdateNominalPolicy` spreads the knots evenly over the planner's time horizon). `q_des(tau)` is MJPC's cubic spline
(`TimeSpline`, `kCubicSpline`): cubic Hermite on each segment with the slopes MJPC uses, one-sided differences at the
two ends and the mean of the two secants at interior knots, the end values held outside the knots. The spline is linear
in the knots, so it is a weight vector, `q_des(tau) = sum_j w_j(tau) Y_j` (`model.interp_weights`, one formula shared by
the CasADi graph, the JAX twins and the driver's resample kernel).

**Nominal controller.** Inertia-shaped PD with gravity and Coriolis compensation,

    u_nom = clip(M(q) (kp (q_des - q) - kd v) + g(q) + c(q, v), -u_max, u_max),   kp = 50, kd = 20.

(MJPC's Panda samples position targets for MuJoCo's stiff joint servos; the inertia-shaped PD is kept from the MPPI
example because the CBF rows, the torque box and the 20 ms semi-implicit Euler step are built around it.)

**CBF rows.** cbfpy's relative-degree-2 rows of the demo's 64 barriers `h2(q)` at the current state, built with
forward-mode directional derivatives exactly as cbfpy does (`G_qp`, `h_qp`; CasADi `jtimes` on the cusadi2 side):

    h = Lf h2 + alpha h2,   G = -Lg h,   hq = alpha h + Lf h,   alpha = 10,

so that `G u <= hq` is the one-step CBF condition on the torque.

**Dynamics and cost.** Semi-implicit Euler, dt = 20 ms (the control period), Pinocchio / frax rigid-body terms on
frax's `panda.urdf`:

    a = M^-1 (u - g - c),  v1 = v + dt a,  q1 = q + dt v1,  c1 = c + dt l(x, u),
    l = w_pos |p_ee - rp|^2 + w_rot |e_rot|^2 + w_v |v|^2 + w_u |u|^2,  (50, 2, 1, 1e-4)

with `e_rot` the orientation error against the demo's tool-down orientation and `rp` the current waypoint. MJPC sums
the per-step costs; the dt factor does not change the argmin.

## 2. The safety filter (shared driver kernel, `k_project` in `safe_mppi.cu`)

For every rollout, with the box rows `|u_j| <= u_max,j`,

    u <- clip(u_nom)
    repeat 48 times:
        i* = argmax_i (G_i u - hq_i) / |G_i|            the most violated half-space (signed distance; lowest index on ties)
        if G_i* u <= hq_i*: stop
        u <- u - (G_i* u - hq_i*) / |G_i*|^2 G_i*          projection onto that half-space
        u <- clip(u, -u_max, u_max)                        projection onto the box

One warp per rollout, lane l owns rows l and l + 32 in registers, the argmax is a shuffle reduction. It is
closed-form arithmetic, it is the same code for every evaluator, and it costs a few microseconds for 64 rollouts.
`examples/panda_safe_mppi/projection_study.py` compares it with cbfpy's elastic QP on 20-step rollouts from initially
safe random states: the greedy filter keeps the barriers above -0.01 as often as the QP does.

## 3. Predictive sampling, MJPC's planner (`safe_mppi.cu`)

MJPC runs two asynchronous loops: the agent applies the current nominal, the planner repeats `OptimizePolicy` as fast as
it can, each call one random-search iteration from the state it reads. Here the plant is stepped by the closed loop, so
the asynchrony is emulated synchronously: every 20 ms tick measures the plant state and runs a fixed number `n_iter` of
iterations from that state, with `n_iter` chosen so that the tick fits the period on the cusadi2 evaluator
(`n_iter = floor(20 ms / p99 iteration time)`, measured; see the README). One iteration, MJPC's `OptimizePolicy`:

    Y_0 = Y0 (the nominal, candidate 0, no noise)
    Y_k = clip(Y0 + sigma eps_k, joint range),  k = 1..K-1,  sigma_j = 0.1 x (hi_j - lo_j) / 2 on every knot of joint j
    roll out all K from the measured plant state (plan clock and cost at zero), T = 25 steps (0.5 s)
    k* = argmin_k c_k (finite costs, lowest index on ties: the nominal wins ties, as in MJPC),   Y0 <- Y_k*

K = 64 (MJPC's Panda task uses 60 `sampling_trajectories`; 64 keeps whole warps). The noise is MJPC's
`AddNoiseToPolicy`: a Gaussian of standard deviation `sampling_exploration = 0.1` times half the control range on every
spline parameter, clamped to the range; for the Panda the control range is the joint range, so sigma is about 0.29 rad.
Philox counters make the samples reproducible and identical across evaluators.

After the last iteration the torque applied to the robot is `u_0^(k*)`, the filtered torque of step 0 of the final
nominal's rollout. Step 0 of every rollout starts from the measured plant state, so this is exactly
`project(model(x_plant, Y0))`, the plant-side filtered torque of the plan; the driver gathers it from the rollout
instead of recomputing it, and `verify_ps.py` checks the two agree bit for bit. Because candidate 0 is the nominal, the
plan's cost never increases from one iteration to the next, and `verify_ps.py` checks that the nominal's cost at
iteration i + 1 is bitwise the cost min of iteration i.

Then the plan is resampled one control period later, MJPC's `UpdateNominalPolicy` (non-sliding): the new knot j is the
current spline evaluated at `dt + j knot_dt`, the end value held past the last knot (`k_resample`; `model.resample_np`
is its reference). MJPC resamples at the start of every iteration; within a tick the time does not advance and the
spline interpolates its own knots, so resampling once per tick is the same operation.

The whole tick of the cusadi2 pipeline, all `n_iter` iterations and the resample, is one CUDA graph; the other
evaluators run the same kernels around their own model calls.

Why MJPC's planner rather than the one-iteration best sample of `panda_safe_ps`: with one batch per tick the nominal
is replaced only when one of the K perturbations beats it, and at the obstacle none did (the plan was kept on 71-81 %
of the ticks and the arm stalled against the sphere on every seed). MJPC's planner is sequential random search: each
iteration that finds a better sample moves the nominal before the next batch is drawn, so a tick gets `n_iter` chances to
improve along a path rather than one, with three times the noise and a C1 spline that keeps the torque smooth.

## 4. Task and plant

MuJoCo, frax's demo scene (`panda.xml` + `obstacle_scene.xml`: the sphere at (0.5, 0.5, 0.5), r = 0.3), 2 ms
physics with a zero-order hold over the 20 ms tick. Waypoints cycle through (0.55, 0.05, 0.55), (0.05, 0.55, 0.55),
(0.55, 0.05, 0.30), (0.05, 0.55, 0.30): the straight line between consecutive ones passes through the obstacle, so
the arm has to go around. A waypoint counts as reached within 4 cm and is abandoned after 6 s. The comparison
controller is frax's own filtered OSC (operational-space PD + cbfpy's one-step CBF-QP through qpax) on the same
waypoints.

The parameters are frozen in advance (MJPC's values where it has one, the MPPI example's gains and weights otherwise);
only `n_iter` is set by measurement. The pass rule fixed before the runs: every attempted waypoint reached within 6 s over
30 s with the barriers above -0.01, on all five seeds. A failure is reported next to the MPPI and one-iteration
predictive-sampling results, not tuned away.

## 5. Precision

fp32 throughout (plans, exported programs with `JAX_DEFAULT_MATMUL_PRECISION=highest`, driver). The MPPI version of this
example checked that fp32 and fp64 give the same closed loop (torques 1e-6 apart, same waypoints, same barrier minima).
MuJoCo Warp, the paper's remaining GPU baseline, has no automatic differentiation and so cannot form the CBF rows, and
has no row here.

## 6. The MJX twin

`mjx_model.py` keeps cbfpy's row construction and the demo's barrier definitions and replaces every frax call by MJX:
`mjx.kinematics` for the body frames (the end-effector pose, the collision-sphere centres in the frames frax attaches them
to), `mjx.crb` + `mjx.full_m` for M, `mjx.rne` for g + c, `mjx.jac` for the end-effector Jacobian of the singularity
barrier. MuJoCo's URDF importer drops the rotational inertia of frax's massless `panda_link8` (0.1 kg m^2 on each axis)
and its own `fullinertia` path runs an iterative 3 x 3 eigensolver that leaves 4e-9 in the mass matrix, so the MJCF is
generated from the Pinocchio model of the same URDF with the composite inertias in principal form; MJX then agrees with
frax to 1e-14 on u_nom, G, h and the integration step.
