# Safe predictive sampling on the Panda, MJPC style: the formulation in equations

The controller of `examples/panda_safe_mjpc` in its tuned configuration (`driver.TUNED`, the `closed_loop.py` default).
Every symbol below maps onto one line of `model.py` or `safe_mppi.cu`. Prose is kept to what the equations cannot say.

## 1. Sizes and constants

| symbol | value | meaning |
|---|---|---|
| $n_v$ | 7 | joints |
| $\Delta t$ | 0.02 s | control period = rollout step |
| $T$ | 35 | rollout steps, horizon $T\Delta t = 0.7$ s |
| $N$ | 7 | spline knots |
| $\kappa$ | $(T-1)\Delta t/(N-1) = 0.1133$ s | knot spacing |
| $K$ | 256 | rollouts per iteration (index 0 = nominal) |
| $n_\mathrm{it}$ | 3 | planner iterations per tick |
| $\sigma$ | 0.10 rad | knot noise std, knots $1\ldots 6$; knot 0: 0 |
| $k_p,\,k_d$ | 20, 20 | PD gains |
| $w_p,\,w_R,\,w_v,\,w_u$ | 100, 2, 1, $10^{-4}$ | cost weights |
| $\alpha$ | 10 | CBF class-$\mathcal K$ gain, both levels |
| $n_\mathrm{proj}$ | 48 | greedy projection iterations |
| $m$ | 64 | barrier rows |

## 2. State, plan, spline

$$
x = (q, v, \tau, c) \in \mathbb{R}^{16}, \qquad
Y = (Y_0, \ldots, Y_{N-1}) \in \mathbb{R}^{N \times n_v}, \qquad Y_j \in [q_{\min}, q_{\max}]
$$

MJPC's cubic Hermite spline (`TimeSpline`, `kCubicSpline`) on the plan clock $\tau$, knot $j$ at $j\kappa$:

$$
s = \operatorname{clip}\!\left(\tfrac{\tau}{\kappa}, 0, N-1\right), \qquad
i = \min(\lfloor s \rfloor, N-2), \qquad t = s - i
$$

$$
\kappa\, m_0 = Y_1 - Y_0, \qquad
\kappa\, m_j = \tfrac{1}{2}(Y_{j+1} - Y_{j-1}) \;\; (0<j<N-1), \qquad
\kappa\, m_{N-1} = Y_{N-1} - Y_{N-2}
$$

$$
q_\mathrm{des}(\tau; Y) = h_{00}(t)\,Y_i + h_{10}(t)\,\kappa m_i + h_{01}(t)\,Y_{i+1} + h_{11}(t)\,\kappa m_{i+1}
= \sum_{j=0}^{N-1} w_j(\tau)\, Y_j
$$

$$
h_{00} = 2t^3 - 3t^2 + 1,\quad h_{10} = t^3 - 2t^2 + t,\quad h_{01} = -2t^3 + 3t^2,\quad h_{11} = t^3 - t^2
$$

Values are held outside $[0, (N-1)\kappa]$. In particular $w(0) = e_0$, so $q_\mathrm{des}(0; Y) = Y_0$.

## 3. Rigid-body model and nominal torque

$$
M(q)\,\dot v + C(q,v)\,v + g(q) = u, \qquad
\dot z = f(z) + B(z)\,u, \quad z = (q, v), \quad
f = \begin{pmatrix} v \\ -M^{-1}(Cv + g) \end{pmatrix}, \quad
B = \begin{pmatrix} 0 \\ M^{-1} \end{pmatrix}
$$

$$
u_\mathrm{nom}(x; Y) = \operatorname{sat}_{\bar u}\!\Big( M(q)\big(k_p\,(q_\mathrm{des}(\tau;Y) - q) - k_d\, v\big) + C(q,v)\,v + g(q) \Big)
$$

## 4. Barriers and the one-step CBF rows (cbfpy, relative degree 2)

$m = 64$ position barriers $h_2(q) \in \mathbb{R}^{m}$ on frax's demo scene, collision spheres $p_k(q)$ with radii $r_k$:

$$
\begin{aligned}
&\text{joint limits (14):} && q_{\max} - q,\;\; q - q_{\min} \\
&\text{singularity (1):} && \sqrt{\det\big(J(q)J(q)^\top\big)} - 10^{-3} \\
&\text{obstacle:} && \|p_k(q) - p_o\| - r_k - r_o, \qquad p_o = (0.5, 0.5, 0.5),\; r_o = 0.3 \\
&\text{self-collision:} && \|p_b(q) - p_a(q)\| - r_a - r_b - \delta_{ab} \\
&\text{floor:} && p_{k,z}(q) - r_k - 0.01
\end{aligned}
$$

Cascade to relative degree 1 and the affine condition on the torque:

$$
h(z) = \nabla_z h_2 \cdot f(z) + \alpha\, h_2(q) = \frac{\partial h_2}{\partial q}\, v + \alpha\, h_2
$$

$$
\dot h + \alpha h \ge 0 \;\Longleftrightarrow\; L_f h + L_B h\, u + \alpha h \ge 0
\;\Longleftrightarrow\; G(z)\, u \le b(z), \qquad
G = -L_B h = -\frac{\partial h}{\partial v} M^{-1}, \qquad b = \alpha h + L_f h
$$

Safe torque set at state $z$:

$$
\mathcal{U}(z) = \{\, u : G(z)\,u \le b(z),\;\; |u_j| \le \bar u_j \,\}
$$

## 5. The safety filter $\Pi$ (`k_project`)

Greedy alternating projection onto the most violated half-space, then the box, $n_\mathrm{proj}$ times:

$$
u^{(0)} = \operatorname{sat}_{\bar u}(u_\mathrm{nom}), \qquad
i^\star_\ell = \arg\max_i \frac{G_i u^{(\ell)} - b_i}{\|G_i\|} \quad(\text{lowest index on ties})
$$

$$
u^{(\ell+1)} =
\begin{cases}
u^{(\ell)} & \text{if } G_{i^\star} u^{(\ell)} \le b_{i^\star} \quad(\text{stop}) \\[4pt]
\operatorname{sat}_{\bar u}\!\Big( u^{(\ell)} - \dfrac{G_{i^\star} u^{(\ell)} - b_{i^\star}}{\|G_{i^\star}\|^2}\, G_{i^\star}^\top \Big) & \text{otherwise}
\end{cases}
\qquad \ell = 0, \ldots, n_\mathrm{proj}-1
$$

$$
\Pi(x; Y) := u^{(n_\mathrm{proj})}
$$

No QP, no linear solve. On 64 rows it matches cbfpy's elastic QP on the rollout metric (`panda_safe_mppi/projection_study.py`).

## 6. One rollout step and the cost

$$
F(x, u):\quad
a = M^{-1}(u - Cv - g),\quad
v^+ = v + \Delta t\, a,\quad
q^+ = q + \Delta t\, v^+,\quad
\tau^+ = \tau + \Delta t,\quad
c^+ = c + \Delta t\, \ell(x, u)
$$

$$
\ell(x,u) = w_p\,\|p_\mathrm{ee}(q) - r\|^2 + w_R\,\|e_R(q)\|^2 + w_v\,\|v\|^2 + w_u\,\|u\|^2,
\qquad
e_R = -\tfrac{1}{2}\sum_{i=1}^{3} R_i(q) \times R^\mathrm{des}_i
$$

$R^\mathrm{des} = \operatorname{diag}(1,-1,-1)$ (tool down), $r$ the current waypoint.

Rollout of a plan $Y$ from the measured plant state $x_p = (q_p, v_p)$:

$$
x_0 = (q_p, v_p, 0, 0), \qquad
u_s = \Pi(x_s; Y), \qquad
x_{s+1} = F(x_s, u_s), \qquad s = 0, \ldots, T-1
$$

$$
J(Y; x_p) = c_T = \Delta t \sum_{s=0}^{T-1} \ell(x_s, u_s)
$$

Every $u_s \in \mathcal U(z_s)$ by construction: a rollout is a CBF-feasible trajectory over the whole horizon.

## 7. The planner (MJPC's sampling planner, synchronous)

One control tick, given the plant state $x_p$ and the nominal plan $Y$:

$$
\text{for } i = 1, \ldots, n_\mathrm{it}:\quad
\begin{cases}
Y^{(0)} = Y, \\[2pt]
Y^{(k)}_j = \operatorname{clip}_{[q_{\min}, q_{\max}]}\!\big(Y_j + \sigma_j\, \varepsilon^{(k)}_j\big),
\quad \varepsilon^{(k)}_j \sim \mathcal N(0, I_{n_v}),\quad k = 1, \ldots, K-1, \\[2pt]
\sigma_0 = 0, \quad \sigma_j = \sigma \;\;(j \ge 1), \\[2pt]
J_k = J\big(Y^{(k)}; x_p\big), \qquad
k^\star = \arg\min_{k:\, J_k < \infty} J_k \quad(\text{lowest index on ties}), \\[2pt]
Y \leftarrow Y^{(k^\star)}
\end{cases}
$$

Applied torque: the step-0 filtered torque of the final nominal's rollout, gathered, not recomputed:

$$
u_\mathrm{applied} = u_0^{(k^\star)} = \Pi(x_p; Y)
$$

Because $k = 0$ is the unperturbed nominal, the plan cost is monotone within a tick:

$$
J(Y_{i}; x_p) \le J(Y_{i-1}; x_p), \qquad i = 1, \ldots, n_\mathrm{it}
$$

No weights, no temperature, no averaging, no second filter: what the rollouts certified is what the robot receives.

Resample one period later (MJPC `UpdateNominalPolicy`, non-sliding):

$$
Y_j \leftarrow q_\mathrm{des}(\Delta t + j\kappa;\, Y), \qquad j = 0, \ldots, N-1
$$

At $T = 35$ ($\Delta t/\kappa = 0.176$) the new knot 0 and 1 in terms of the old knots are

$$
Y_0^+ = 0.811\,Y_0 + 0.202\,Y_1 - 0.013\,Y_2, \qquad
Y_1^+ = -0.060\,Y_0 + 0.930\,Y_1 + 0.142\,Y_2 - 0.013\,Y_3
$$

All $n_\mathrm{it}$ iterations and the resample are one CUDA graph per tick.

## 8. Plant, task, budget

$$
\text{MuJoCo, } \delta t = 2\ \mathrm{ms}, \qquad u(t) = u_\mathrm{applied}\ \text{held for } \Delta t = 20\ \mathrm{ms}
$$

$$
r \in \{(0.55, 0.05, 0.55),\ (0.05, 0.55, 0.55),\ (0.55, 0.05, 0.30),\ (0.05, 0.55, 0.30)\}\ \text{cyclic},
\qquad \text{reached: } \|p_\mathrm{ee} - r\| < 0.04,\quad \text{timeout } 6\ \mathrm{s}
$$

The straight line between consecutive waypoints crosses the obstacle sphere.

$$
t_\mathrm{iter} \approx 0.14\ \mathrm{ms} \times T \quad(\text{latency-bound: 207 levels, flat in } K \le 256), \qquad
n_\mathrm{it} = \left\lfloor \frac{20\ \mathrm{ms}}{t^{p99}_\mathrm{iter}} \right\rfloor = 3 \ \text{at } T = 35
$$

Measured tick: 15.5 to 15.9 ms median, 16.0 to 16.1 ms p99 (cusadi2, fp32).

## 9. Result (5 seeds, 30 s each)

| seed | reached / timed out | mean time to reach | min $h_2$ | max $\|v\|_\infty$ |
|---:|---:|---:|---:|---:|
| 1 | 14 / 0 | 2.0 s | -0.014 | 2.3 rad/s |
| 2 | 25 / 0 | 1.2 s | -0.009 | 2.5 |
| 3 | 26 / 0 | 1.1 s | -0.009 | 2.7 |
| 4 | 18 / 0 | 1.6 s | -0.006 | 2.6 |
| 5 | 19 / 0 | 1.6 s | -0.008 | 2.5 |

Reference points on seed 1: safe MPPI (averaged, re-filtered, $T=100$) 12 / 0 at 2.5 s; one-iteration best sample
($T = 100$) 12 / 1; MJPC-faithful settings 3 / 3; frax's filtered OSC 23 / 0 at 1.3 s.

## 10. The hyperparameter search and why the winner works

Structure fixed throughout: best sample applied, filter only inside the rollouts, no re-filter, tick $\le 20$ ms.
46 configurations (`sweep.py`, grids 1 to 6), 2 seeds for the first grid and 5 for the rest, 30 s each. Mean over
seeds, reached / timed out:

| step | change | result | note |
|---|---|---|---|
| 0 | MJPC-faithful: $K=64$, $T=25$, $n_\mathrm{it}=4$, $\sigma = 0.29$ rad on every knot, $k_p = 50$, $w_p = 50$ | 4.8 / 2.2 | $h_2$ to -0.020, $\|v\|$ to 5.7 rad/s |
| 1 | $\sigma = 0.10$ rad absolute, $K = 256$, $n_\mathrm{it} = 5$ at $T = 25$ | 5.0 / 3.0 (seeds 1-2) | |
| 2 | freeze knot 0 | 10.0 / 0 (seeds 1-2), 7.8 / 1.2 (5 seeds) | two seeds; did not survive the controlled test below |
| 3 | iterations vs samples: $5\times256$ / $3\times512$ / $2\times1024$ | 7.8 / 1.2, 6.2 / 1.4, 5.2 / 2.2 | iterations win |
| 4 | $\sigma$: 0.07 / 0.10 / 0.15 | 7.6 / 1.2, 7.8 / 1.2, 4.0 / 2.8 | |
| 5 | horizon: $T=35$, $n_\mathrm{it}=3$ | 9.2 / 1.6 | $T=50$, $n_\mathrm{it}=2$ and $T=100$, $n_\mathrm{it}=1$ worse |
| 6 | $w_p = 100$ (at $T=25$) | 10.2 / 2.0 | faster, speed spikes to 5.7 rad/s |
| 7 | $T=35$, $w_p=100$, $k_p=30$ | 21.6 / 0.2 | $h_2$ to -0.016 on one seed |
| 8 | **$k_p = 20$** | **20.4 / 0.0** | $h_2$ to -0.014 on one seed, 2.7 rad/s; the final configuration |
| 9 | around it: $w_p = 150$ | 26.2 / 0.2 | faster, but one seed times out |
| 9 | $n_\mathrm{proj} = 96$ | 20.6 / 0.0, same -0.015 dip | the filter is converged |
| 9 | $w_v = 2$, $w_v = 0.5$, $w_R = 0$, no nominal in the batch | all worse | |

**A controlled test of the two levers the sweep credited.** The sweep's ordering came from one lever at a time on
early configurations, two seeds for the first grid. `knot0_study.py` re-runs the final configuration with everything
fixed except the noise on knot 0 and $k_p$, 5 seeds each, 30 s (`results/knot0_study_*.json`):

| mean over 5 seeds | $k_p = 20$, knot 0 frozen (final) | $k_p = 20$, knot 0 noisy | $k_p = 50$, frozen | $k_p = 50$, noisy |
|---|---:|---:|---:|---:|
| reached / timed out | 20.4 / 0.0 | 18.0 / 0.4 | 17.2 / 0.6 | 17.6 / 0.8 |
| per seed | 14, 25, 26, 18, 19 | 18, 11+2, 26, 18, 17 | 19+1, 22, 18, 16, 11+2 | 26, 12+1, 18, 14+2, 18+1 |
| torque jump between ticks, median / p99 (N m) | 1.3 / 8.3 | 6.7 / 28.0 | 1.5 / 12.0 | 10.8 / 53.3 |
| max joint speed (rad/s) | 2.7 | 3.0 | 3.4 | 4.5 |
| min $h_2$ | -0.014 | -0.016 | -0.014 | -0.014 |
| candidates beating the nominal | 2.9 % | 2.4 % | 0.9 % | 1.0 % |
| ticks where no iteration improves | 26 % | 26 % | 48 % | 41 % |
| winner's $\|Y_0 - q\|$ minus the nominal's | 0 | +0.03 rad | 0 | +0.03 rad |

**What freezing knot 0 does.** From $w(0) = e_0$,

$$
u_0^{(k)} = \Pi\big(x_p;\, Y^{(k)}\big) \;\text{ depends on } Y^{(k)} \text{ only through } Y^{(k)}_0 ,
$$

so with $\sigma_0 = 0$ every candidate shares the applied torque of the tick, and a selection reaches the robot one
period later through the resample of section 7, scaled by the weight 0.20 on knot 1:

$$
\delta u_\mathrm{applied}^{\,+} \approx M k_p\,(0.202\, \delta Y_1 - 0.013\, \delta Y_2),
\qquad \|\delta Y_1\| \sim \sigma = 0.1\ \mathrm{rad} \;\Rightarrow\; \|\delta Y_0^+\| \sim 0.02\ \mathrm{rad}.
$$

With $\sigma_0 = \sigma$ the perturbation of the winning sample enters the torque directly as $M k_p\, \delta Y_0$. The
table confirms the arithmetic: the between-tick torque jump is 5 times larger at $k_p = 20$ and 7 times at $k_p = 50$,
and the peak joint speed rises with it. It also shows what freezing does **not** do: the search is unchanged (the
acceptance statistics are the same, candidates that pull knot 0 toward $q$ are no cheaper than those that push it
away, winners do not drift toward $q$), and at $k_p = 50$ the completion counts are the same with and without it. The
sweep's "5 / 3 to 10 / 0" was two seeds on an early configuration ($T = 25$, $w_p = 50$). Knot 0 is kept frozen for
the smoother torque and the lower peak speed, which cost nothing; it is not the reason the controller works.

**Why iterations beat samples.** One iteration costs its graph depth, not its batch ($t_\mathrm{iter}$ flat in $K$ up
to 256), so samples are free up to 256 and iterations are the only thing the budget buys. Within a tick each
iteration is an accept-if-better step at scale $\sigma$ from the current nominal; $n_\mathrm{it}$ iterations compound
along a path, one iteration of $n_\mathrm{it} K$ samples does not. Three such steps per tick are enough to walk the
plan round the sphere; one per tick (`panda_safe_ps`, $T = 100$) stalled with the plan kept on 71 to 81 % of ticks.

**Why $T = 35$.** The 0.5 s horizon of MJPC's Panda task does not see the way round the sphere (the arm parked 5 to
45 cm short of the far waypoints); 0.7 s does, and still fits 3 iterations. Longer horizons cost iterations
(2 at 1.0 s, 1 at 2.0 s) and lose more than they gain.

**Why $\sigma = 0.10$ rad.** Smaller explores too little; larger makes most candidates worse than the nominal (the
acceptance rate falls) and the accepted ones faster, which is where the barrier dips come from.

**Why $k_p = 20$ with $w_p = 100$.** These, with the horizon, are what moved completion. At $k_p = 50$ about 1 % of
candidates beat the nominal and on 41 to 48 % of ticks no iteration improves anything; at $k_p = 20$ it is 2.4 to 2.9 %
and 26 %. A softer PD makes the rollout cost a smoother function of the knots, so more of the $\sigma$-ball is an
improvement and the sequential iterations find a path; the barrier dips, always obstacle rows at speed, fall with the
speed. $w_p = 100$ pays for the slower tracking by making progress worth more than velocity in the cost. $w_v$ in either
direction breaks the balance: $w_v = 2$ penalises the motion the task needs, $w_v = 0.5$ allows the speeds that dip.

**Why the nominal must be in the batch.** Dropping $k = 0$ removes the monotonicity of section 7 and the plan drifts
by noise when no candidate is good: 7.0 / 1.4 against 7.8 / 1.2, with barrier dips twice as deep.

**What remains.** The obstacle rows dip to -0.014 on one seed, unchanged at $n_\mathrm{proj} = 96$: the one-step
condition $G u \le b$ is imposed on a 20 ms step against a 2 ms plant at 2.5 rad/s. It is a discretisation margin of the
filter, not a planner or projection failure.

## 11. Draft paragraph for the paper

The controller is MuJoCo MPC's sampling planner, predictive sampling [Howell et al. 2022]: a nominal joint-target plan
on a cubic spline with 7 knots, $K - 1$ Gaussian perturbations of it per iteration, every candidate rolled out over the
horizon, the lowest-cost candidate replacing the nominal (the nominal wins ties), and MJPC's resampling of the plan by
one control period. We make two changes to the algorithm. First, a control-barrier-function safety filter acts on the
torque at every step of every candidate rollout: the sampled plan sets a PD torque, the filter projects that torque
onto the one-step CBF condition of the 64 barriers of the scene (joint limits, singularity, obstacle, self-collision,
floor) inside the torque box, and the filtered torque is what the dynamics integrate. Every rollout is therefore a
CBF-feasible trajectory over the whole horizon, and the torque sent to the robot is the first torque of the selected
rollout, without averaging and without a second filter. The plans themselves are never filtered, only clamped to the
joint range; a plan that aims into a barrier is rolled out as the arm being held at the barrier and loses the selection
on cost. Second, the planner runs a fixed three iterations per 20 ms control tick, synchronously from the same measured
state, in place of MJPC's asynchronous as-many-as-fit loop; three is the number of 5.3 ms iterations that fit the period
on the cusadi2 evaluator, and fixing it keeps the tick deterministic and the evaluators comparable. The parameters
differ from MJPC's Panda task (Table X): 256 samples instead of 60, a 0.7 s horizon instead of 0.5 s, a knot noise of
0.10 rad instead of 0.1 times half the joint range, an inertia-shaped PD torque with gravity and Coriolis compensation
as the nominal controller in place of MJPC's position actuators, with $k_p = 20$ and a position weight of 100, and no
noise on the first knot. The MJPC-faithful configuration, run first with the pass rule fixed in advance, fails the task
on all five seeds (Section Y); of the parameter changes, the horizon, the PD gain and the position weight are what
restore completion, while leaving the first knot unperturbed reduces the between-tick torque jump five to seven times
and the peak joint speed without changing the completion rate (a controlled test, Section Z).

## 12. Where the filter sits: ablation on the same planner (2026-10-17)

Same tuned planner (knot 0 frozen), same seeds, 30 s; only the constraint handling changes. `closed_loop.py` flags:
`--proj-iters 0` (no filter), `--filter-output-only`, `--select feasible` (a candidate whose unfiltered torque violates the
one-step CBF condition anywhere along its rollout ranks below every feasible one; if none is feasible the least violating
wins). Results `results/closed_loop_ps_{nocbf,outfilter,reject,reject_outfilter}_seed<s>.json`.

| | reached / timed out | per seed | time to reach | min $h_2$ | max $\|v\|_\infty$ | ticks with obstacle $h_2 < 0$ |
|---|---:|---|---:|---:|---:|---:|
| **ours: filter in every rollout step** | 20.4 / 0.0 | 14, 25, 26, 18, 19 | 1.52 s | -0.014 | 2.7 rad/s | 12 % |
| no filter | 10.0 / 1.2 | 12+1, 7+3, 10+1, 9+1, 12 | 2.03 s | -0.076 | 8.1 | 69 % |
| filter on the planner's output only | 6.8 / 2.6 | 8+2, 8+2, 6+3, 6+3, 6+3 | 1.44 s | -0.031 | 10.5 | 36 % |
| rejection (infeasible = infinite cost), no filter | 17.0 / 0.0 | 14, 19, 14, 19, 19 | 1.75 s | -0.006 | 1.9 | 2.7 % |
| rejection + output filter | 17.4 / 0.0 | 17, 17, 15, 19, 19 | 1.67 s | -0.008 | 1.9 | 2.8 % |

No filter: the plant enforces the obstacle, floor and joint limits by contact (the model has none), the arm is pressed
against the sphere on 59 to 80 % of ticks. Output filter only: the unaware planner keeps choosing the line through the
sphere and the filter holds the arm against it, the same three waypoints time out on three seeds (deadlock). Rejection:
117 of 256 candidates feasible per iteration, none on 9 to 10 % of ticks; conservative (only plans whose nominal torque
is already safe), half the barrier dip of ours, 30 % lower peak speed, 17 % fewer waypoints. It needs the same CBF rows
per rollout step as ours, so the evaluator comparison is unchanged; its tick (20 ms vs 15.8) is an unoptimised
one-thread-per-rollout violation kernel, not the method.
