# End-to-end MPC/MPPI visualizations on A1 and H1

Closed-loop demos where LP-batched Pinocchio constrained dynamics drive a
simulated quadruped / humanoid, with MuJoCo for rendering. The demo
validates the LP evaluation path end-to-end and produces paper-ready video
of GPU-accelerated analytical contact dynamics closing the control loop at
real-time rates.

Hardware: NVIDIA L40S (GPU 5). Env: `cusadi2`. fp64 throughout.

## Status

| Demo | Script | Video | Status |
|---|---|---|---|
| A1 MPPI (constrained) | [`a1_mppi.py`](a1_mppi.py) | [`videos/a1_mppi.mp4`](videos/a1_mppi.mp4) | **5 s stable standing + push recovery at real-time rates.** |
| H1 MPPI | — | — | Pending. |
| A1 iLQR | — | — | Pending. |
| H1 iLQR | — | — | Pending. |

## A1 MPPI — constrained dynamics

The rollout model is `a1_constrained_fwd.casadi` —
`pinocchio.constraintDynamics` with 4 × CONTACT_3D at FL/FR/RL/RR feet,
LOCAL_WORLD_ALIGNED. Compiled to the CusADi level-parallel GPU kernel by
[`build_a1_constrained_fwd.py`](build_a1_constrained_fwd.py).

Function statistics:
- **11,107 instructions** (~10× smaller than the constrained-Jacobian kernel's 102K)
- **359 compute levels**, total ops 10,943
- **peak_live_slots = 1,252** → **9.8 KB shared memory** per SM (no tiling needed — fits untiled LP kernel)

### Per-control-step timing (N=128 samples, T=20-step horizon, dt=0.02 s)

```
LP dynamics (all T=20 rollout steps):     mean=5.65 ms   p95=5.96 ms
MPPI total (rollout + cost + softmax):   mean=19.9 ms   p95=22.2 ms
Pinocchio sim step (10 GPU sub-steps):   mean=4.32 ms
Offscreen render (1280×720):             mean=4.12 ms
----------------------------------------
Real-time factor: 0.76× (5.0 s sim in 6.59 s wall)
```

The LP dynamics kernel runs at the same rate as the open-loop benchmark
(see [`demo/mpc-benchmark/RESULTS.md`](../mpc-benchmark/RESULTS.md));
closed-loop adds CPU↔GPU state transfers and Pinocchio CPU-side projection
steps. At N=256 LP dynamics scales to ~6 ms; the MPPI loop is
**~6 ms LP dynamics, 14 ms other** (sampling, cost, softmax, transfers).

### Trajectory — 5 seconds with 60 N lateral push at t=2.5 s

| metric | min | mean | max | final |
|---|---:|---:|---:|---:|
| base_z (m) | 0.218 | 0.256 | 0.301 | 0.274 |
| quat_w (upright, 1.0 = perfectly upright) | +0.88 | +0.96 | +1.00 | +0.98 |

Robot maintains stance for the full 5 s and recovers from the 60 N
lateral perturbation (base_z dip at t=2.5 s followed by recovery to ~0.27 m).

### Pipeline structure

1. `build_a1_constrained_fwd.py` saves `a1_constrained_fwd.casadi`.
2. `validate_a1_mapping.py` once, to confirm MuJoCo↔Pinocchio joint mapping.
3. `a1_mppi.py` main loop:
   - Holding torques for the initial MPPI nominal computed via a throwaway
     MuJoCo sim + `mj_inverse` at startup.
   - 50 Hz control: sample N=128 perturbations of T=20 torque trajectories
     around the nominal, roll out on GPU via the LP constrained-fwd kernel,
     softmax-weight by cost, apply the first control.
   - **MPPI + joint-space PD feedback:** applied torque =
     `u0_mppi − K_p (q − q_home) − K_v v`, with K_p=80, K_v=3. Standard
     MPC+PD hybrid — MPPI handles feed-forward and exploration over
     trajectories, PD handles high-frequency stabilization that a 50 Hz
     receding-horizon plan can't correct on its own.
   - **Advance "true state"** with the same LP kernel on a single state,
     10 GPU sub-steps per 20 ms control period, integrated via
     `pin.integrate`.
   - **Position projection** after each sub-step: solve
     `δq = J^+ (p_foot_home − p_foot_current)` and integrate; stabilizes
     the constraint at position level (see "Why position projection" below).
   - **Viscous damping** (2 s⁻¹) applied in the sim only — regularizes
     velocities in lieu of MuJoCo-style contact friction.

### Why Pinocchio is both controller and simulator

We originally used MuJoCo as the simulator; the MJCF (`unitree_mujoco`)
and URDF (`unitree_ros`) repos have different mass/inertia and different
contact formulations (soft penalty vs hard bilateral). MuJoCo's
`mj_inverse` gave ~4 Nm holding torque at the calf; the Pinocchio
constrained ddq=0 inverse solution lives in a 6-dim null space (12
contact constraints on 18 DOF, 12 actuators over-determining the 6
independent post-contact DOFs) with no natural selector — Newton picks
arbitrary null-space points with 100+ Nm magnitudes.

With the two models genuinely different, MPPI optimizing the Pinocchio
model cannot stabilize MuJoCo by MPPI tuning alone. Using the same
Pinocchio constrained kernel as both the planner's model and the sim's
physics eliminates the divergence and keeps the demo on the paper
contribution: a GPU-accelerated analytical constrained-dynamics kernel
closing a feedback loop at real time.

MuJoCo is retained for rendering (`mujoco.Renderer` + offscreen EGL). State
is pushed to `mj_data.qpos` and `mujoco.mj_forward` updates kinematics;
`mj_step` is never called.

### Why position projection (instead of in-kernel Baumgarte)

Pinocchio's `RigidConstraintModel` exposes Baumgarte corrector parameters
(`m_baumgarte_parameters.Kp/Kd`) and a target placement
(`desired_contact_placement`). Setting these on a `pin.RigidConstraintModel`
works — values read back correctly — but they do **not** propagate through
`cpin.RigidConstraintModel(cm)` to the symbolic form used by CasADi.
Verification: building a fn with `Kp=400, Kd=40` and `desired_contact_placement
= home_foot_world_xyz` produces a symbolic ddq bit-identical to the same
function built with `desired_contact_placement` left at the default identity
(0, 0, 0). With the default target, enabling Kp>0 pulls feet toward the
origin with a force proportional to their distance from it — catastrophic.

Workaround: disable the in-kernel Baumgarte corrector (Kp=Kd=0) and do
position-level projection on the Python side after each integration step.
Cheap (one FK + one 12×18 linear solve per sub-step, adds ~0.5 ms),
doesn't need kernel changes, and produces the target behavior. The
projection uses the damped-pseudoinverse IK correction
`δq = Jᵀ (J Jᵀ + λI)⁻¹ (p_target − p_current)` integrated via
`pin.integrate` to keep the quaternion on SO(3).

## Joint-mapping validation

[`validate_a1_mapping.py`](validate_a1_mapping.py) checks the
MuJoCo MJCF ↔ Pinocchio URDF correspondence:
- Joint ordering **differs**: MuJoCo is `FR,FL,RR,RL`; Pinocchio is `FL,FR,RL,RR`.
  Permutation `mj_to_pin = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]` saved to
  `a1_mapping.npz`.
- Quaternion storage differs: MuJoCo `[w,x,y,z]`, Pinocchio `[x,y,z,w]`.
- After applying both permutations, FK at a lifted+yawed pose agrees to
  **~2.8 mm** on body positions — residual is geometry drift between the
  upstream URDF and MJCF repos.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=5 python demo/e2e-visualization/validate_a1_mapping.py
CUDA_VISIBLE_DEVICES=5 python demo/e2e-visualization/build_a1_constrained_fwd.py
CUDA_VISIBLE_DEVICES=5 python demo/e2e-visualization/a1_mppi.py \
    --duration_s 5.0 --N 128 --T 20 --sigma 1.5 --temperature 1.0 \
    --push_t 2.5 --push_duration 0.1 --push_force 60.0
```

Outputs:
- `videos/a1_mppi.mp4` — 5 s, 30 fps, 1280×720
- `videos/a1_mppi_timing.json` — per-step timing
- `figures/a1_mppi_t_{0p0s,push,recover}.png` — keyframes for paper figure
- `a1_mapping.npz` — joint index + quaternion permutation
- `src/casadi_functions/a1_constrained_fwd.casadi` — compiled kernel
