# Safe MPPI on the Panda: the formulation

One Franka Panda, torque controlled at 50 Hz, reaching a sequence of end-effector waypoints around a spherical
obstacle under the 64 control-barrier functions of frax's OSCBF demo (joint limits, singularity, obstacle, self-collision,
floor). The controller is MPPI over K sampled joint-target plans; every rollout step passes its torque through a
closed-form CBF safety filter, so the sampler only ever sees safe trajectories, and the torque sent to the robot is
filtered the same way. There is no QP solver and no linear solve anywhere except the 7 x 7 Cholesky of the forward
dynamics.

## 1. Model (the evaluator)

State `x = [q (7), v (7), tau, c]`: joint positions and velocities, the plan clock and the accumulated cost. Two pure
model functions are evaluated per rollout step, each by one of five interchangeable evaluators: cusadi2 (CasADi graph,
`model.py`, level-parallel plans), frax + cbfpy (JAX, `frax_model.py`), MuJoCo XLA + cbfpy (JAX, `mjx_model.py`), the
original CusADi's unrolled kernels of the CasADi graph, or CasADi's C code generation of it on the CPU:

    model(x, Y, rp, par)      -> (u_nom (7), G (64 x 7), h (64))
    integrate(x, u, rp, par)  -> x1

**Plan.** `Y` holds N_KNOT = 4 knots of 7 joint targets at times j * knot_dt, knot_dt = T dt / 3, linearly
interpolated in the plan clock: `q_des(tau) = sum_j w_j(tau) Y_j`.

**Nominal controller.** Inertia-shaped PD with gravity and Coriolis compensation,

    u_nom = clip(M(q) (kp (q_des - q) - kd v) + g(q) + c(q, v), -u_max, u_max),   kp = 50, kd = 20.

**CBF rows.** cbfpy's relative-degree-2 rows of the demo's 64 barriers `h2(q)` at the current state, built with
forward-mode directional derivatives exactly as cbfpy does (`G_qp`, `h_qp`; CasADi `jtimes` on the cusadi2 side,
agreement 5e-12, `verify.py`):

    h = Lf h2 + alpha h2,   G = -Lg h,   hq = alpha h + Lf h,   alpha = 10,

so that `G u <= hq` is the one-step CBF condition on the torque.

**Dynamics and cost.** Semi-implicit Euler, dt = 20 ms (the control period), Pinocchio / frax rigid-body terms on
frax's `panda.urdf`:

    a = M^-1 (u - g - c),  v1 = v + dt a,  q1 = q + dt v1,  c1 = c + dt l(x, u),
    l = w_pos |p_ee - rp|^2 + w_rot |e_rot|^2 + w_v |v|^2 + w_u |u|^2,  (50, 2, 1, 1e-4)

with `e_rot` the orientation error against the demo's tool-down orientation and `rp` the current waypoint.

## 2. The safety filter (shared MPPI kernel, `k_project` in `safe_mppi.cu`)

For every rollout, with the box rows `|u_j| <= u_max,j`,

    u <- clip(u_nom)
    repeat 48 times:
        i* = argmax_i (G_i u - hq_i) / |G_i|            the most violated half-space (signed distance; lowest index on ties)
        if G_i* u <= hq_i*: stop
        u <- u - (G_i* u - hq_i*) / |G_i*|^2 G_i*          projection onto that half-space
        u <- clip(u, -u_max, u_max)                        projection onto the box

One warp per rollout, lane l owns rows l and l + 32 in registers, the argmax is a shuffle reduction. It is
closed-form arithmetic, it is the same code for both evaluators, and it costs about 55 us for 256 rollouts, a
fifth of one cusadi2 model call. `projection_study.py` compares it with cbfpy's elastic QP on 20-step rollouts
from initially safe random states (results in `results/report.md`): the greedy filter keeps the barriers above -0.01
as often as the QP does, while an exact small-active-set projection (three most violated rows, closed-form KKT) was
tried and rejected: nearly parallel rows (neighbouring obstacle spheres) make its exact solution jump far from the
nominal torque in infeasible situations.

Why this filter rather than the QP inside the rollouts: a QP per rollout step (K T of them per tick) would put a
solver back on the critical path; the point of the demo is that with the filter closed-form, the tick is model
evaluation, where the two backends differ.

## 3. MPPI (`safe_mppi.cu`)

Per tick (20 ms plant period), K = 256 samples `Y_k = clip(Y0 + sigma eps_k, joint range)` (sample 0 is Y0 itself, sigma = 0.1 rad on
every knot, Philox noise), rolled out T = 100 steps (2 s, knot spacing 0.67 s) from the measured plant state with the plan clock and
cost at zero. With the final costs `c_k`,

    w_k = exp(-(c_k - min c) / (temp std c)),  temp = 0.3,     Y0 <- sum_k w_k Y_k / sum_k w_k.

The torque applied to the robot is `project(model(x_plant, Y0))`, the filtered PD torque of the mean plan at the
plant state. Then the plan is re-sampled one period later (knot j <- lerp(knot j, knot j + 1, dt / knot_dt), the last
knot held). Vanilla MPPI, one iteration. The whole tick of the cusadi2 pipeline is one CUDA graph; the frax pipeline
runs the same kernels around PJRT executions of the exported programs.

## 4. Task and plant

MuJoCo, frax's demo scene (`panda.xml` + `obstacle_scene.xml`: the sphere at (0.5, 0.5, 0.5), r = 0.3), 2 ms
physics with a zero-order hold over the 20 ms tick. Waypoints cycle through (0.55, 0.05, 0.55), (0.05, 0.55, 0.55),
(0.55, 0.05, 0.30), (0.05, 0.55, 0.30): the straight line between consecutive ones passes through the obstacle, so
the arm has to go around. A waypoint counts as reached within 4 cm and is abandoned after 6 s. The comparison
controller is frax's own filtered OSC (operational-space PD + cbfpy's one-step CBF-QP through qpax) on the same
waypoints.

## 5. Precision

fp64 throughout (plans, exported programs, driver). cusadi2's level kernel is latency-bound at K = 256 (one block
per rollout, one barrier per level), so fp64 costs it little; XLA's fp64 on a consumer GPU is several times its fp32,
which is the regime where the two earlier demos' claims also held. MuJoCo Warp, the paper's remaining GPU baseline, is fp32-only and therefore has no row here.

## 6. The MJX twin

`mjx_model.py` keeps cbfpy's row construction and the demo's barrier definitions and replaces every frax call by MJX:
`mjx.kinematics` for the body frames (the end-effector pose, the collision-sphere centres in the frames frax attaches them
to), `mjx.crb` + `mjx.full_m` for M, `mjx.rne` for g + c, `mjx.jac` for the end-effector Jacobian of the singularity
barrier. MuJoCo's URDF importer drops the rotational inertia of frax's massless `panda_link8` (0.1 kg m^2 on each axis)
and its own `fullinertia` path runs an iterative 3 x 3 eigensolver that leaves 4e-9 in the mass matrix, so the MJCF is
generated from the Pinocchio model of the same URDF with the composite inertias in principal form; MJX then agrees with
frax to 1e-14 on u_nom, G, h and the integration step.
