## Deviations from frax / cbfpy behaviour, and what did not work

**Source used.** The pip wheel `frax 0.0.4` in this environment predates the demo API
(`torque_control_matrices`, iiwa, the current collision model). The profiling therefore
runs against an unmodified git checkout of frax at `cc0a066` (pyproject version 0.1.0,
2026-10-07), cloned by `common.ensure_frax()` into `profiling/third_party/frax` and put at
the front of `sys.path`. cbfpy 0.1.0, qpax 0.1.4 and elastiqp 0.1.0 are the pip releases,
unmodified. All profiling code is in `profiling/`.

**Panda (problem 1).** `DemoOSCBFConfig` is imported from `examples/oscbf_demo.py` and the
OSC controller is a verbatim copy of `operational_space_control` minus the trailing
`safety_filter`. Deviations: (i) no MuJoCo environment, so the posture target `des_q` is
frax's Panda `default_configuration`, which is the same array the demo's `env.q_init`
holds (`franka_panda.default_q`); (ii) desired end-effector poses come from FK of random
joint vectors instead of the viewer's mocap target, with zero desired twist; (iii) the
state distribution is random inside the joint limits with 0.5 rad/s RMS velocities rather
than a closed-loop trajectory, and half of it is deliberately pushed towards joint limits
or collisions. `OSCBFTorqueConfig` hard-codes `backend="elastiqp"`; for the qpax runs the
attribute is changed on the config *instance* before `CBF.from_config` (the file is not
touched).

**G1 (problem 2).** frax has no G1 CBF demo. `g1_problem.G1OSCBFTorqueConfig` is
`cbf_utils.OSCBFTorqueConfig` with `Manipulator` replaced by `Humanoid`: the task Jacobian
is the stacked left/right hand Jacobian (12 × 29), `f`, `g`, `P`, `q` are otherwise
identical, torque bounds are the URDF efforts. Barriers are all relative degree 2: the 58
joint-limit rows plus frax's 97 G1 self-collision pairs (91 body–body pairs from
`body_sc_pairs` and 6 pairs against the fixed pelvis root from `root_sc_pairs`), evaluated
with frax's own `link_collision_data` and `_self_collision_distances_from_link_data`.
Class-K functions are α(h) = 10 h and α₂(h₂) = 10 h₂ as in the Panda demo. The nominal
controller is dual-hand OSC with a null-space posture task (default configuration), using
the Panda demo's gains per hand (kp 50/20, kd 20/10, posture kp 10, kd 5), built from the
same per-step terms as frax's `tests/timing/time_floating_base.py::osc_terms`. There is
no singularity, floor or external-obstacle barrier for the G1.

**What is in each timed piece.** `qp_data` is `cbf.qp_data(z, u_des)` exactly as cbfpy
builds it. `dyn_terms` (2a) evaluates `mass_matrix`, `jnp.linalg.inv(M)`, gravity +
Coriolis and the task Jacobian(s) the way the config's `f`, `g`, `_P` do. `barriers` (2b)
is cbfpy's `h_and_Lfh` plus `Lgh`, i.e. the barrier values with their Lie derivatives; cbfpy
evaluates `f` and `g` (hence M and M⁻¹) again inside those jvps, so 2a and 2b overlap and
do not add up to `qp_data`. `h2_only` (extra) is the raw barrier vector without
derivatives. `solve` is the solver called on precomputed `(P, q, G, h)` with cbfpy's
penalties and `solver_tol`: `qpax.solve_qp_elastic_primal(...)` (the call cbfpy makes; the
iteration counts come from `qpax.solve_qp_elastic`, which is the same function returning
its loop state — the two agree to 3e-8 on x) and `elastiqp.jax.solve(...).x` with
`eps_abs = solver_tol` and elastiqp's default dual-active-set method. `end_to_end` is the
nominal controller followed by `cbf.safety_filter` in one jit.

**elastiqp and vmap.** The handoff states elastiqp cannot run under `vmap`. In elastiqp
0.1.0 the JAX binding does accept `vmap` (`vmap_method="sequential"`: one FFI call that
loops over the batch on the host thread), so a vmapped elastiqp solve costs about batch ×
the single-instance time (checked at B = 16 during development). It is still CPU-only and
fp64-only, so, as requested, the tables report its single-instance CPU time and the
throughput of one process per physical core; no elastiqp GPU or fp32 entries exist. For
elastiqp "iterations" are active-set iterations, not interior-point iterations, so the
two backends' counts are not comparable.

**Batch 1 on CPU vs GPU.** On CPU batch 1 is run unbatched (plain `jax.jit`, like the
demo). On GPU every batch size, including 1, goes through `jax.jit(jax.vmap(...))`.

**fp32.** Only qpax is run in fp32 (`JAX_ENABLE_X64=False`, everything float32 including
the model). The convergence fraction and iteration counts in the tables show how the 1e-5
tolerance behaves at that precision; nothing was retuned.

**Iteration counts.** Under `vmap`, qpax's `while_loop` runs until every instance of the
batch has converged (or `max_iter` = 30), so the per-batch cost is set by the batch
maximum; both the per-instance distribution and the batch-max statistics (over the
consecutive windows of the shuffled 4096-state dataset) are reported. The loop count
observed in the GPU trace can differ from the standalone solve's batch-max because the
fused end-to-end program rounds differently and the 1e-5 KKT test is sensitive to it.

**G1 in fp32 and cbfpy's construction-time check.** `CBFConfig.__init__` validates P(z = 1)
with `np.allclose(P, Pᵀ, atol=1e-14)` and a Cholesky of P + 1e-14·I. For the G1 in float32
(entries of P reach ~1e3, from M⁻¹ squared) this fails by rounding alone and the G1 fp32
run aborted on first launch. `G1OSCBFTorqueConfig._is_symmetric_psd` therefore applies the
same test with float32-scaled tolerances (1e-4 × max|P|; XLA uses TF32 for float32 matmuls on this GPU by default, which leaves a relative asymmetry of about 2e-5 in P) when the matrix is float32, and
defers to cbfpy's check unchanged for float64. This affects only the one-off validation at
construction, not anything that is timed. The Panda fp32 run passed cbfpy's check as is.

**Nominal torques and the torque-bound rows.** Because the desired poses are FK of
independent random configurations, the task error is often large and the nominal OSC
torque exceeds a URDF effort limit in 73 % of the Panda states and 36 % of the G1 states
(`u_sat_frac` in `solver_stats.csv`). Those states make the torque-bound rows of the QP
active in addition to the barrier rows, which is more work for the solver than a tracking
controller close to its target would produce. The "active CBF row" column counts barrier
rows only.

**Trace bookkeeping.** In the batch-1024 trace the pre-loop device time (15.6 ms) contains
the nominal controller, `qp_data` and qpax's initial factorisation; the separately timed
`nominal` + `qp_data` at that batch is 14.7 ms, so the solver init is of the order of 1 ms.
The loop ran 30 iterations there, the same as the standalone solve's batch max; in a
development trace at batch 16 the fused program needed 21 iterations where the standalone
solve reported 30, which is the rounding sensitivity described above.

## frax vs cusadi2 as the model-evaluation backend (added after the profiling)

`bench_cusadi2_swap.py` swaps frax for cusadi2 in the Panda GPU / qpax / fp64 pipeline, batch
1, 4, 16, …, 4096 (`results/cusadi2_swap.md`, `.csv`, `.png`). `panda_casadi.py` re-implements the
model side (nominal OSC torque and cbfpy's P, q, G, h) in CasADi with Pinocchio's CasADi bindings on
frax's own `panda.urdf`; it matches frax + cbfpy to ≤ 1e-9 relative on 64 random states, and the
safe torques of the two pipelines agree to 1e-12 … 1e-5 (1e-2 at batch 4096, where one window
contains a QP at the convergence tolerance). cusadi2 compiles that function into one plan, and
`cusadi2_ffi.cu` runs it inside the JAX program as an XLA FFI custom call on XLA's stream, so the
comparison is jit against jit. The bare kernel timed with CUDA events equals the jit time at
every batch size: the FFI path adds no copies, and assembling the dense P and G from cusadi2's
column-major nonzeros costs nothing measurable.

**What did not work first.** The first CasADi version formed cbfpy's Lie derivatives as
`ca.jacobian(h, z) @ f` with `h` itself containing `ca.jacobian(h2, z) @ f`. Numerically identical,
but the symbolic second-order Jacobian through the 7×7 and 6×6 linear solves gave 485k SX nodes,
2839 levels (median level width 45 operations), and 90 KB of shared memory per block, so cusadi2's
level kernel was latency bound: ~1.1 ms per wave of one block per SM, 1.2 ms at batch 1 and 59 ms
at batch 4096 (54 waves), 2.8× slower than frax. Those results are kept in
`results/cusadi2_swap_jacobianform.*`. cbfpy does not form Jacobians; it uses `jax.jvp`. Writing the
same algebra with forward-mode directional derivatives (`ca.jtimes`) and `ca.cse` gives 37.7k nodes
and 361 levels for identical values. cusadi2's `max_live` budget (shared memory per block) then
trades depth for occupancy: 6000 slots → 361 levels, 35 KB, block 512; 3000 slots → 601 levels,
16 KB, block 256. Both are reported.
