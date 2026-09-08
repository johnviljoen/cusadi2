# MIT humanoid MPC: formulation, QP, ADMM, and the staged GPU pipeline

Everything below is read off the wip-branch ("CusADi 2.0") code
(`controllers/mit_humanoid_mpc.py`, `optimization/optimization_problem.py`,
`optimization/qp_backends/admm_backend.py`, commits by se-hwan, Jan–Mar 2026)
and off `build_mpc_stage.py` / `mpc_cudss_bench.cu` in this directory. Where
the code does something unusual it is stated as coded, not idealised.

## 1. Model and notation

MIT humanoid, floating base, Pinocchio conventions.

| symbol | size | meaning |
|---|---|---|
| $q$ | 25 | configuration: base position (3), base quaternion $(x,y,z,w)$ (4), 18 joint angles |
| $v$ | 24 | generalized velocity, base part in the base (body) frame |
| $n_v$ | 24 | tangent dimension |
| $F$ | 12 | contact forces at 4 point contacts, world frame, **normalised by body weight** $m g$ |

Joint order: right leg (hip yaw, hip ab/ad, hip pitch, knee, ankle), left leg
(same), right arm (shoulder pitch, ab/ad, yaw, elbow), left arm (same).
Contact order: right toe, left toe, right heel, left heel; $F$ is stacked
$[F^{rt}; F^{lt}; F^{rh}; F^{lh}]$, each $(x,y,z)$. Toe and heel frames are
fixed frames on each foot link at $(0.105, 0, -0.04)$ and $(-0.045, 0, -0.04)$
(added here; the public URDF has none).

Manifold operations, all Pinocchio: $q \oplus \delta$ is `integrate`
(SE(3) exponential in the base frame for the floating base, addition for
joints); $q_2 \ominus q_1$ is `difference(q_1, q_2)` (the tangent vector
taking $q_1$ to $q_2$). Frame kinematics $p_c(q)$, $\dot p_c(q, v)$ and
frame Jacobians $J_c(q) \in \mathbb{R}^{3 \times 24}$ are `LOCAL_WORLD_ALIGNED`
(world-aligned axes at the frame origin). $\mathrm{RNEA}(q, v, a)$ is
inverse dynamics with rotor inertia, gear ratio and armature set from the
actuator table.

## 2. Horizon and decision variables

$N = 10$ stages, $k = 0, \dots, 9$, with a geometric time grid
$\Delta t_k = \Delta t_0 r^k$ summing to $T = 1.2$ s:

$\Delta t = (0.0209, 0.0285, 0.0387, 0.0526, 0.0716, 0.0973, 0.1324, 0.1801, 0.2449, 0.3330)$ s, $r \approx 1.36$.

Per-stage decision variable, $n_s = 60$:

$$x_k = \begin{bmatrix} \delta q_k \\ v_k \\ F_k \end{bmatrix} \in \mathbb{R}^{24 + 24 + 12},
\qquad x = [x_0; \dots; x_9] \in \mathbb{R}^{600}.$$

The configuration at stage $k$ is *not* a decision variable; it is
reconstructed from the measured configuration through a tangent-space
increment:

$$q_k = q_0 \oplus \delta q_k .$$

Note the increment is always relative to $q_0$, not to $q_{k-1}$.
Accelerations are finite differences of the body-frame velocities,

$$a_k = \frac{v_k - v_{k-1}}{\Delta t_k}, \qquad v_{-1} := v_0 ,$$

so $a_k$ couples stages $k-1$ and $k$; this is the only coupling in the cost.

## 3. Parameters ($n_p = 806$)

| name | size | benchmark value |
|---|---|---|
| $\Delta t$ | 10 | the grid above |
| $q_0$ | 25 | nominal stance: base $(0,0,0.65)$, identity quaternion, legs $(0,-0.1,-0.6,1,-0.5)$, arms $(0.2,\mp 0.1,0,-0.5)$ |
| $v_0$ | 24 | 0 |
| $q^{des}_k$, $v^{des}_k$, $F^{des}_k$ | 25, 24, 12 per stage | $q_0$, 0, 0 |
| $\phi_k$ (swing phase) | 4 per stage | 0 |
| $c_k$ (contact flags) | 4 per stage | 1 (all feet in stance) |
| $Q_{fb}, Q_{leg}, Q_{arm}$ | 6, 5, 4 | $(0,0,1000,1200,1200,1200)$, $(100,100,50,25,20)$, $(20,20,20,20)$ |
| $Q^v_{fb}, Q^v_{leg}, Q^v_{arm}$ | 6, 5, 4 | $(800,800,400,100,100,100)$, $(5,5,0.1,0.1,0.1)$, $(5,5,5,5)$ |
| $Q^a_{\cdot}$ | 6, 5, 4 | 0 |
| $R_F$ | 12 | 0.1 |

Weight matrices are assembled in joint order,
$Q_q = \mathrm{diag}(Q_{fb}, Q_{leg}, Q_{leg}, Q_{arm}, Q_{arm})$ and likewise
$Q_v$, $Q_a$. With the benchmark values $Q_a = 0$, so the acceleration cost is
structurally present but numerically zero (it still shapes the sparsity).

## 4. Cost

$$f(x; p) = \sum_{k=0}^{9} \Delta t_k \Big[ e_{q,k}^\top Q_q\, e_{q,k} + e_{v,k}^\top Q_v\, e_{v,k} + a_k^\top Q_a\, a_k + e_{F,k}^\top \mathrm{diag}(R_F)\, e_{F,k} \Big]$$

with $e_{q,k} = \tilde q^{des}_k \ominus q_k \in \mathbb{R}^{24}$,
$e_{v,k} = v^{des}_k - v_k$, $e_{F,k} = F^{des}_k - F_k$. The desired
quaternion is sign-aligned to the current one before the difference:
$\tilde q^{des}_k$ has its quaternion multiplied by $\mathrm{sign}(\langle
\mathrm{quat}(q_k), \mathrm{quat}(q^{des}_k)\rangle)$ (an `if_else`, so it is
piecewise smooth).

## 5. Constraints, 74 per stage ($n_g = 740$)

In order of the rows of $g_k(x_{k-1}, x_k; p)$. $\mu = 0.7$.

| rows | name | expression | type |
|---|---|---|---|
| 0–23 | pos_integration | $q_k \ominus \big(q_{k-1} \oplus (v_{k-1} + a_k \Delta t_k)\,\Delta t_k\big) = 0$, $q_{-1} := q_0$ | eq |
| 24–29 | dynamics | $\big[\mathrm{RNEA}(q_k, v_k, a_k)\big]_{1:6} - \big[J_c(q_k)^\top (m g\, F_k)\big]_{1:6} = 0$ | eq |
| 30–33 | foot_swing | $p_{c,z}(q_k) - h(\phi_{k,c}) = 0$ for each contact $c$ | eq |
| 34–37 | friction_x lower | $-\mu F_{z}/\sqrt2 - F_{x} \le 0$ | ineq |
| 38–41 | friction_x upper | $F_{x} - \mu F_{z}/\sqrt2 \le 0$ | ineq |
| 42–49 | friction_y | same for $F_y$ | ineq |
| 50–53 | contact_x_vel | $\dot p_{c,x}(q_k, v_k)\, c_{k,c} = 0$ | eq |
| 54–57 | contact_y_vel | $\dot p_{c,y}(q_k, v_k)\, c_{k,c} = 0$ | eq |
| 58–69 | contact_F | $F^{c}_k (1 - c_{k,c}) = 0$ (3 rows per contact) | eq |
| 70 | toe_lateral | $\big[R_{yaw}(q_k)(p_{lt} - p_{rt})\big]_y - r^2 - o \ge 0$ | ineq |
| 71 | heel_lateral | same with heels | ineq |
| 72 | right_leg_max | $\|\tfrac12(p_{rt}+p_{rh}) - p_{r,hip}\|^2 - L^2 \le 0$ | ineq |
| 73 | left_leg_max | $\|\tfrac12(p_{lt}+p_{lh}) + p_{l,hip}\|^2 - L^2 \le 0$ | ineq |

Details:

- **pos_integration.** Since $v_{k-1} + a_k \Delta t_k = v_k$ identically,
  the row is $q_k = q_{k-1} \oplus v_k \Delta t_k$: implicit-Euler integration
  of the body-frame velocity on the manifold, written through $a_k$ in the
  DAG. It couples stage $k$ to $x_{k-1}$ (through $\delta q_{k-1}$ and
  $v_{k-1}$). The stage-0 row uses the parameters $q_0$, $v_0$ directly.
- **dynamics.** Only the six floating-base rows of the equations of motion
  are imposed (no torque variables; joint torques are implicit). Contact
  wrenches enter through the stacked 3×24 contact Jacobians; forces are in
  units of body weight, hence the factor $m g$.
- **foot_swing.** $h(\phi)$ is a quartic Bézier in the phase
  $\phi \in [0,1]$ with control points $(0, 0, v_{TO}/4, \tfrac83(h_{sw} - v_{TO}/16 + v_{TD}/16), -v_{TD}/4)$,
  $v_{TO} = 0.02$, $v_{TD} = -0.02$, $h_{sw} = 0.05$ m, i.e.
  $(0, 0, 0.005, 0.1267, 0.005)$. With $\phi = 0$ (stance) it pins the
  contact height to 0. This is an equality on all four feet at every stage.
- **friction.** A pyramid $|F_x|, |F_y| \le \mu F_z / \sqrt 2$, written as two
  one-sided rows each (Opti canonicalises $l \le e \le u$ with expression
  bounds into $l - e \le 0$ and $e - u \le 0$). No explicit $F_z \ge 0$ row
  (commented out in the wip).
- **contact velocity / force complementarity** is scheduled, not
  optimised: with $c_{k,c} = 1$ the foot's horizontal velocity is zero and the
  force is free; with $c_{k,c} = 0$ the force is zero and the foot is free.
- **lateral spacing.** $R_{yaw}$ is the rotation by minus the base yaw
  (yaw extracted from the quaternion by `atan2`), $r = 0.15$ m,
  $o = 0.2$ m; as coded the term is $r^2 = 0.0225$, not $r$.
- **leg length.** $L = 0.58$ m. **As coded the left-leg row uses
  `- - p_left_hip_yaw`, i.e. adds the hip position instead of subtracting
  it.** This is a sign error in the wip formulation and is reproduced
  unchanged here so that the benchmark matches the reference bit-for-bit; it
  makes that row a near-inactive bound rather than a leg-length limit.

Equality rows: 54 (0–33, 50–69); inequality rows: 20. Bounds per row:
equalities $l = u = 0$; friction and leg-length rows $(-\infty, 0]$; lateral
rows $[0, \infty)$.

## 6. The QP (one SQP step)

The wip `SQPInterface` with `sqp_cfg={'max_iter': 1}` takes **one** step from
the initial guess $x^{(0)} = 0$ (all increments, velocities and forces zero,
i.e. the robot standing still at $q_0$). Let $g(x) = [g_0; \dots; g_9]$ and

$$P = \nabla^2_{xx} f(x^{(0)}), \qquad c = \nabla_x f(x^{(0)}), \qquad A = \partial_x g(x^{(0)}),$$

$$l = l_g - g(x^{(0)}), \qquad u = u_g - g(x^{(0)}) .$$

$P$ is the exact Hessian of the **cost only** (`ca.hessian(f, x)`); constraint
curvature (the Lagrangian's $\sum \lambda_i \nabla^2 g_i$) is not included.
The step is the solution of

$$\min_{d \in \mathbb{R}^{600}} \ \tfrac12 d^\top P d + c^\top d \quad \text{s.t.} \quad l \le A d \le u ,$$

and the controller output is $x^{(1)} = x^{(0)} + d = d$. Multipliers are
discarded. With the benchmark parameters $P$ has 966 upper-triangular
nonzeros, $A$ has 9456, and the nonzero pattern is block-tridiagonal in the
stage index: $P_{k,k}$ (75 upper nnz), $P_{k-1,k}$ (24), $A_{k,k}$ (735),
$A_{k,k-1}$ (234), nothing at distance 2.

## 7. Ruiz equilibration (as coded, 2 passes)

Form $M = \begin{bmatrix} P & A^\top \\ A & 0 \end{bmatrix} \in \mathbb{R}^{1340 \times 1340}$,
$S = I$, $\gamma_{tot} = 1$, $\bar q = c$. For each of the 2 passes:

1. $\delta_j = \big(\max_i |M_{ij}|\big)^{-1/2}$ for every column $j$, with
   $\delta_j = 1$ if the column is zero; $D = \mathrm{diag}(\delta_{1:600})$,
   $E = \mathrm{diag}(\delta_{601:1340})$.
2. $\bar P \leftarrow D \bar P D$, $\bar A \leftarrow E \bar A D$, $\bar q \leftarrow D \bar q$.
3. $\gamma = 1 / \max\big(\max_j \|\bar P_{:,j}\|_\infty,\ \|\bar q\|_\infty\big)$;
   $\bar P \leftarrow \gamma \bar P$, $\bar q \leftarrow \gamma \bar q$.
4. $S \leftarrow \mathrm{diag}(\delta)\, S$, $\gamma_{tot} \leftarrow \gamma\, \gamma_{tot}$; rebuild $M$ from $\bar P, \bar A$.

Then with $D = \mathrm{diag}(S_{1:600})$, $E = \mathrm{diag}(S_{601:1340})$, $c_s = \gamma_{tot}$:

$$P_s = c_s D P D, \quad A_s = E A D, \quad q_s = c_s D c, \quad l_s = E l, \quad u_s = E u .$$

Maxima are taken over the full symmetric columns of $M$ (both $A$ and
$A^\top$ blocks). Operation order matters for bit-exact reproduction: every
scaled entry is computed as $(\text{row scale} \cdot \text{entry}) \cdot
\text{column scale}$, and $c_s D$ is formed before multiplying into $P$.

## 8. ADMM (OSQP iteration with a per-row $\rho$)

Constants: $\bar\rho = 0.1$, $\sigma = 10^{-6}$, $\alpha = 1.6$. Per-row
penalty $\rho_i = \bar\rho\, r_i$ with $r_i = 10^3$ on equality rows and 1
otherwise, so $\rho = 100$ on equalities and $0.1$ on inequalities. $\rho$ is
fixed (no adaptation), the problem is the scaled one ($P_s, q_s, A_s, l_s,
u_s$), and the iteration starts from $x^0 = y^0 = z^0 = 0$.

Factor once:

$$K = \begin{bmatrix} P_s + \sigma I & A_s^\top \\ A_s & -\mathrm{diag}(1/\rho) \end{bmatrix} .$$

For $t = 0, \dots, 24$ (exactly 25 iterations, no termination test):

$$K \begin{bmatrix} \tilde x \\ \nu \end{bmatrix} = \begin{bmatrix} \sigma x^t - q_s \\ z^t - y^t / \rho \end{bmatrix},$$

$$w = \alpha \nu + (1-\alpha) y^t, \qquad
x^{t+1} = \alpha \tilde x + (1-\alpha) x^t,$$

$$z^{t+1} = \Pi_{[l_s, u_s]}\big(z^t + w / \rho\big), \qquad
y^{t+1} = w + \rho\,(z^t - z^{t+1}) .$$

(Divisions and products by $\rho$ are row-wise.) This is algebraically the
standard OSQP update with $\tilde z = z^t + (\nu - y^t)/\rho$ and relaxation
$\alpha$; the wip merely pre-combines the relaxed quantities into $w$. The
output is the unscaled primal step

$$d = D\, x^{25} .$$

$y$ and $z$ are not unscaled or returned. OSQP itself on the same problem with
`max_iter = 25` returns a solution within $1.3\times10^{-9}$ of this loop,
which confirms the loop is OSQP.

Numerical remark: $K$ is quasi-definite ($P_s + \sigma I \succ 0$,
$-\mathrm{diag}(1/\rho) \prec 0$), so $LDL^\top$ exists for any symmetric
permutation. The entries span $\sigma = 10^{-6}$ to $1/\rho = 10$ after
scaling; in fp32 the 25-iteration loop lands $3\times10^{-2}$ to
$9\times10^{-2}$ from the fp64 solution (and the wip's own fp32 symbolic LDL
$0.47$), so all reported accuracy is fp64.

## 9. The wip's GPU realisation (the "cusadi" baseline)

Six whole-horizon CasADi functions, each evaluated one thread per QP
instance (the original CusADi execution model):

| function | inputs → outputs | instructions |
|---|---|---|
| `KKT_matrix` | $(x, p) \to \mathrm{triu}\begin{bmatrix} P & A^\top \\ A & 0\end{bmatrix}$ nonzeros (10,422) | 607,593 |
| `ruiz_scaling` | $(\text{kkt nz}, x, p, \sigma, \bar\rho) \to (\mathrm{triu}\,K$ (11,162 nz)$, q_s, D, l_s, u_s)$ | 277,164 |
| `ldl_factorize` | $\mathrm{triu}\,K \to (D_{LDL}, L, \pi)$, symbolic sparse LDL (`ca.ldl`, fill 11,162 → 27,533) | 1,277,725 |
| `KKT_vector` | $(q_s, x, y, z, \sigma, \bar\rho) \to$ rhs | 7,243 |
| `ldl_solve` | $(\text{rhs}, D_{LDL}, L) \to (\tilde x, \nu)$ | 143,025 |
| `ADMM_step` | $(\tilde x, \nu, x, y, z, l_s, u_s, \alpha, \bar\rho) \to (x, y, z)$ | 16,725 |

In the results table the "cusadi" column keeps the first, second, fourth and
sixth of these (working sets 542 KB, 330 KB, 35 KB, 53 KB) and replaces the
two LDL functions by cuDSS; the symbolic-LDL variant is also measured
(872 ms at $B=1$) and kept in the raw CSVs.

## 10. The staged realisation (cusadi2 + cuDSS)

### 10.1 One generic stage

A single CasADi function `mpc_assemble` evaluates, for stage $k$,

inputs: $x_{k-1}, x_k, x_{k+1}, \Delta t_k, \Delta t_{k+1}, \mathbb{1}_{k=0}, \mathbb{1}_{k<N-1}, p^{stage}_k, p^{shared}$

outputs: $\mathrm{triu}\,P_{k,k}$ (75), $P_{k-1,k}$ (24, rows $x_{k-1}$, cols $x_k$), $A_{k,k}$ (735), $A_{k,k-1}$ (234), $c_k$ (60), $l_k$, $u_k$ (74 each)

where $p^{stage}_k = (q^{des}_k, v^{des}_k, F^{des}_k, \phi_k, c_k)$ (69) and
$p^{shared} = (q_0, v_0, \text{weights})$ (106). The stage cost used for the
derivatives is

$$J_k = \ell_k(x_{k-1}, x_k) + \mathbb{1}_{k<N-1}\; \Delta t_{k+1}\, a_{k+1}^\top Q_a a_{k+1},$$

i.e. the stage's own term plus the part of stage $k+1$'s acceleration term
that depends on $v_k$; then $P_{k,k} = \nabla^2_{x_k x_k} J_k$,
$c_k = \nabla_{x_k} J_k$ (needs $x_{k+1}$ and $\Delta t_{k+1}$),
$P_{k-1,k} = \partial_{x_k} \nabla_{x_{k-1}} J_k$, $A_{k,k} = \partial_{x_k} g_k$,
$A_{k,k-1} = \partial_{x_{k-1}} g_k$. The indicator $\mathbb{1}_{k=0}$
switches $q_{k-1}, v_{k-1}$ to the parameters $q_0, v_0$ (an `if_else`, so
the untaken branch cannot poison the value); $\mathbb{1}_{k<N-1}$ masks the
next-stage term and the padded $x_{N}$ slot. Assembled over the ten stages this
reproduces the wip `KKT_matrix` output **bit-for-bit** (10,422 nonzeros,
identical pattern).

The function has 77,859 instructions, 173 levels and an 8,197-slot working
set (64 KB in fp64); it runs as one level-parallel cusadi2 launch with grid
$B \times 10$. Neighbour stages are read through cusadi2's *coupled* inputs:
the buffer $[x_{-1}; x_0; \dots; x_9; x_{10}]$ is passed three times with
offsets 0, 1, 2 (and $\Delta t$ with offsets 0, 1, padded with 1 to avoid a
division by zero in the masked term).

### 10.2 Staged Ruiz

Column $j$ of $M$ in stage $k$ touches block rows $k-1, k, k+1$, so each
Ruiz quantity is computed from a *window* $(\text{blk}_k, \text{blk}_{k+1})$
plus the neighbours' scales. The two passes become five stage-parallel
launches and two reductions:

| launch | reads | writes |
|---|---|---|
| `ruiz_r1` | blk$_k$, blk$_{k+1}$ | $\delta^{(1)}_k$ (134) |
| `ruiz_r2` | blk$_{k,k+1}$, $\delta^{(1)}_{k-1,k,k+1}$, $c_k$ | $m^{(1)}_k = \max(\max_j \|\bar P^{(1)}_{:,j}\|_\infty, \|\bar q^{(1)}_k\|_\infty)$ |
| reduce | $m^{(1)}_{0..9}$ | $\gamma_1 = 1/\max_k m^{(1)}_k$ |
| `ruiz_r3` | blk, $\delta^{(1)}$, $\gamma_1$ | $\delta^{(2)}_k$ |
| `ruiz_r4` | blk, $\delta^{(1)}, \delta^{(2)}$, $\gamma_1$, $c_k$ | $m^{(2)}_k$ |
| reduce | $m^{(2)}$ | $\gamma_2$ |
| `ruiz_r5` | blk$_k$, $\delta^{(1),(2)}_{k-1,k}$, $\gamma_{1,2}$, $c_k, l_k, u_k, \sigma, \bar\rho$ | scaled block row of $K$ (1,142 values incl. $\sigma$ and $-1/\rho$ diagonals), $q_{s,k}$, $D_k$, $l_{s,k}$, $u_{s,k}$, rhs$^0_k$ |

Because every reduction is a maximum, the staged Ruiz is order-independent
and reproduces the wip's scaled $K$, $l_s$, $u_s$ to 1 ulp and $q_s$, $D$
bit-for-bit. Three `cudaMemcpy2DAsync` calls pad stage outputs into the
coupled-input layout between launches.

### 10.3 KKT ordering and cuDSS

Rows are ordered stage-interleaved, $[x_0; g_0; x_1; g_1; \dots]$, so $K$ is
block tridiagonal with $134 \times 134$ blocks and its lower triangle has
11,162 nonzeros ($10 \times 1142 - (24 + 234)$, stage 0 having no $k-1$ blocks). One gather kernel
maps `ruiz_r5`'s per-stage output into the uniform-batch CSR values
$(B \times 11162)$; the pattern is shared and analysed by cuDSS once
(`CUDSS_MTYPE_SYMMETRIC`, `CUDSS_MVIEW_LOWER`, `CUDSS_CONFIG_UBATCH_SIZE = B`,
default pivoting and reordering). Factorisation once per solve, then 25
solves. Because the rhs and solution are stage-contiguous, `KKT_vector` and
`ADMM_step` fuse into one stage-parallel launch `admm_iter` per iteration:

inputs $(\tilde x_k, \nu_k), x^t_k, y^t_k, z^t_k, q_{s,k}, l_{s,k}, u_{s,k}, D_k, \sigma, \bar\rho, \alpha$
→ outputs $x^{t+1}_k, y^{t+1}_k, z^{t+1}_k$, next rhs$_k$, $d_k = D_k x^{t+1}_k$.

The whole solve (11 launches + factorisation + 25 × (solve + iterate)) is
captured into one CUDA graph; cuDSS's device allocations are served from a
pre-reserved `cudaMemPool` via `cudssSetDeviceMemHandler`, which is what makes
its phases capturable.

### 10.4 What is verified

At $x^{(0)} = 0$ with the benchmark parameters, fp64:

| quantity | staged vs wip |
|---|---|
| $\mathrm{triu}\begin{bmatrix} P & A^\top \\ A & 0\end{bmatrix}$ (10,422 nz) | bit-exact |
| $q_s$, $D$ | bit-exact |
| scaled $K$, $l_s$, $u_s$ | $\le 1$ ulp ($4\times10^{-16}$ rel.) |
| $d$ after 25 iterations, CPU, wip LDL | $1.4\times10^{-10}$ |
| $d$ on the GPU with cuDSS, every instance, all $B$ | $8.5\times10^{-10}$ |
| OSQP on the same QP | $1.3\times10^{-9}$ |

## 11. Things to state when reporting this as a result

- It is a single SQP iteration linearised at rest ($x^{(0)} = 0$ at the nominal
  stance), not a converged nonlinear MPC solve; the benchmark measures the
  cost of one QP of the controller's inner loop.
- 25 fixed ADMM iterations, no convergence check, fixed $\rho$, cold start.
- $P$ is the cost Hessian, not the Lagrangian Hessian.
- Contact schedule and swing height are parameters (given gait), so the QP has
  no complementarity; friction is a pyramid; no $F_z \ge 0$ row.
- Two wip quirks are reproduced verbatim: the left-leg length sign and the
  squared collision radius in the lateral-spacing bound.
- Toe/heel frames are reconstructed, not from the published model.
- fp32 is not accurate enough for this scaled KKT; all accuracy claims are fp64.
