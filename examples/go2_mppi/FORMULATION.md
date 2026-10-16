# Go2 parkour with MPPI: the formulation

Everything the GPU evaluates is one CasADi SX function, `go2_step` (`go2_model.py`): one physics step of
the Unitree Go2 on the course, with the PD controller, the penalty contacts and the running cost inside
it. MPPI (`mppi_rollout.cu`) only chains that function and reduces over samples.

## 1. Model

**Rigid-body dynamics.** The Menagerie `unitree_go2/go2.xml` is loaded through Pinocchio's MJCF parser
(the repaired load path of `examples/menagerie_bench/model.py`): free-flying base, 12 revolute joints,
nq = 19 (base position, quaternion xyzw, joints), nv = 18. The forward dynamics is Pinocchio's
articulated-body algorithm in CasADi, which matches MuJoCo's `mj_forward` (contacts and constraints
disabled, joint damping applied as a torque) to 2e-13 in `qacc`.

MuJoCo's joint damping d = 2 N m s/rad and armature 0.01 kg m² come from the XML. The damping is treated
implicitly for free: with the armature set to a + Δt d, `aba(q, v, τ − d v)` solves
(M + Δt D) a = τ − D v − h(q, v), which is the backward-Euler damping update. Joint friction loss
(0.2 N m) is dropped (nominal plant, so this is a modelling choice, not an approximation of a reference).

**Integration.** Semi-implicit Euler with Δt = 4 ms: v⁺ = v + Δt a, q⁺ = q ⊕ Δt v⁺ (`pinocchio.integrate`).
Five steps make one 20 ms control period. Δt was chosen on the CPU: at 5 ms a standing robot chatters
from the regularised friction (0.5 rad/s joint velocities), at 4 ms with v_eps = 0.1 m/s the chatter is
0.36 rad/s and the robot stays put, at 2.5 ms (or v_eps = 0.2) it is gone. The runs use 4 ms and
v_eps = 0.1.

**Control.** The plan is N_KNOT = 5 knots of 12 joint-target offsets a_j from the Menagerie `home`
pose, at times j · KNOT_DT (KNOT_DT = horizon / 4 = 0.1 s), linearly interpolated in the plan clock τ:
q_des(τ) = clip(home + Σ_j w_j(τ) a_j, joint range). The torque is the Menagerie position actuator,
τ = clip(kp (q_des − q) − kd q̇, ±τ_max) with kp = 50, kd = 0.5, τ_max = (23.7, 23.7, 45.43) N m per leg.

**Terrain.** `course.height(x, y)` is a single analytic function: boxes are products of tanh steps of
scale EDGE = 12 mm, ramps are clamped linear functions, features are combined with `fmax`. The contact
normal is its gradient, n = (−h_x, −h_y, 1) / |·|, from `ca.jacobian`.

**Contact.** 13 points: the 4 foot spheres (r = 22 mm, go2.xml), the 4 knees (calf origin, r = 30 mm),
4 trunk corners and the belly. For a point p with velocity v (both from the same forward kinematics
the ABA uses), with h = h(p_x, p_y) and n_z the normal's vertical component,

    d   = r + (h − p_z) n_z                      penetration along the normal
    f_n = max(0, k d (1 − a v_n))   if d > 0      Hunt-Crossley-like, k = 15 kN/m, a = 1 s/m
    f_t = −μ f_n v_t / sqrt(|v_t|² + v_eps²)     regularised Coulomb, μ = 0.8, v_eps = 0.1 m/s

f = f_n n + f_t is applied as an external spatial force on the parent joint (`aba(..., fext)`). The
regularisation makes a loaded foot creep at a few cm/s; the PD plus MPPI absorb it.

## 2. State and function signature

    x = [q (19), v (18), t, τ, c]      t: gait clock, τ: time since the plan started, c: accumulated cost
    go2_step(x, knots (60), rp (4), par) -> x⁺

`rp` = (goal_x, goal_y, goal_yaw, gait phase) is per robot; `par` holds every gain, contact and cost
parameter (`go2_model.PARAMS`), so tuning never rebuilds the plan. The cost is accumulated inside the
state (c⁺ = c + Δt ℓ), so a rollout is just H × 5 chained evaluations and the sample cost is read off
the last state. Two plans are built, one physics step and five fused steps; they time the same, and the
driver uses the single step.

## 3. Running cost ℓ (per second)

The weights and parameters used are the defaults in `go2_model.PARAMS` (w_vel 3, w_yaw 0.5, w_ang 0.02,
w_up 10, w_h 100, w_gait 150, w_foot 5, w_body 10, w_slip 0.1, w_pos 0.2, w_ctrl 0.1, w_tau 1e-5,
w_jv 1e-4; v_max 0.6 m/s, d_slow 0.5 m, h0 0.29 m). They were tuned on flat ground and on each approach
in isolation with eight copies of one robot carrying different values (`gen_run.py`).

With e = goal − p_xy, dist = |e|, u = e / dist, the speed command v_cmd = v_max u · min(1, dist / d_slow),
R the base rotation, v_w = R v_body:

| term | expression | role |
|---|---|---|
| vel | w_vel |v_w,xy − v_cmd|² | progress toward the goal |
| yaw | w_yaw wrap(yaw − goal_yaw)² | arrive facing the centre |
| ang | w_ang |ω|² | damping |
| up | w_up ((R_20 − s)² + R_21²) | roll zero, pitch following the terrain slope s under the feet |
| h | w_h (p_z − h_feet − h0)² | body height above the support under the feet |
| gait | w_gait Σ_i (z_i − z_tar,i)² | trot clock foot clearance (section 4) |
| foot | w_foot Σ_i |p_foot,i,xy − p_hip,i,xy|² | feet under their hips (no leg crossing) |
| body | w_body Σ_non-foot f_n / 100 | knees and trunk off the terrain |
| slip | w_slip Σ_feet in contact |v_t|² | no skating |
| ctrl, tau, jv, pos | small regularisers on q_des − home, τ, q̇, dist | |

**Support height.** For foot i at p, with `BRIDGE` = (0.2, 0.4) m, the terrain is sampled behind and
ahead of the foot along u, and

    sup_i = max(h(p), min(max(h(p − 0.2u), h(p − 0.4u)), max(h(p + 0.2u), h(p + 0.4u))))

so a hole narrower than about 0.4 m is level with its rims (the gap does not pull the height
reference down). h_feet is the mean of sup over the four feet and the slope s is the front/rear
difference of sup over the hip-to-hip length. (`body_look` = 1 would make both references use the
terrain `look` ahead of each foot as well; it was tried for the tall step and made the stairs worse,
so it is off.)

## 4. Gait prior

A trot clock at gait_hz = 2 Hz with phase offsets (FL, FR, RL, RR) = (0, ½, ½, 0) and duty = 0.45
(a long swing: 0.55 gave shorter strides and more gap failures). During a foot's swing phase
(s = (φ − duty)/(1 − duty) > 0) the target height above the support is a sine bump of amplitude
swing · move (swing = 0.15 m), where move = min(1, (|v_cmd| + 0.3 |yaw error|) / 0.15) switches the
gait off at the goal. A swing foot measures its bump from the terrain `look` = 0.2 m ahead of it as
well, so it lifts before a riser:

    z_tar,i = max(sup_i, [swing] h(p_i + look u)) + z_ref,i(φ)

An unreachable target (a 0.2 m riser from a 0.29 m stance) is deliberately left as a steep cost:
saturating the error (`gait_clip`) removed the incentive to climb at all.

The term is a timing and clearance preference only: where the feet land, and how the body gets up an
obstacle, are left to the sampling.

## 5. MPPI

Per control step, `iters` annealed sampling iterations (DIAL-MPC style). Iteration i draws, for each
of the R robots, K − 1 samples Y = Y0 + σ_i,j ε around the current plan (sample 0 is Y0 itself), with
σ_i,j = σ · anneal^i · hfac^(N_KNOT−1−j): the noise shrinks over iterations and is smaller on the
knots executed first. Samples are clipped to the joint range. Every sample starts from the robot's
plant state with τ = c = 0 and is rolled out for H = 20 control periods (0.4 s, 100 evaluations of
`go2_step` over the batch of R · K). With the costs c_k of one robot (non-finite rollouts get zero
weight),

    w_k = exp(−(c_k − min c) / (temp · std c)),   Y0 ← Σ_k w_k Y_k / Σ_k w_k

Then the plant advances one control period under Y0 (the same function, batch R), and the plan is
re-sampled one period later (knot j ← lerp(knot j, knot j+1, 0.02 / KNOT_DT), the last knot held).
The whole control step, iterations included, is captured once as a CUDA graph and replayed.

Vanilla MPPI is `iters=1`.

## 6. Precision

The rollouts run in fp32 (kernel and driver built with `CUSADI2_SCALAR=float`); the fp64 build is
3x slower and is used only for the cross-check in the README.
