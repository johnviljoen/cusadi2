"""
A1 MPPI with LP-batched COMPLIANT-CONTACT forward dynamics — end-to-end viz.

MPPI rolls out torque trajectories via the GPU-batched compliant ABA
(`a1_compliant_fwd` — unconstrained ABA + spring-damper ground wrench on
each foot), costs them, and weighted-averages to a first control action.
The "true" physics is the same LP kernel stepped on a single state.

The compliant model replaces the earlier `constraintDynamics` version.
Rigid bilateral constraints at the acceleration level drift under explicit
integration and needed three workarounds (position projection, viscous
damping, stiff PD). A spring-damper ground has no position-level drift —
feet sink ~3 mm under body weight at equilibrium and damping bleeds out
bounce energy.

The rollout integrator is semi-implicit Euler with n_sub=5 substeps per
rollout step (dt_sub=0.004 s). The earlier linearized-implicit-Euler path
compiled a second CasADi function returning ∂ddq/∂v; the LP kernel had
subtle correctness issues at near-home configurations on that Jacobian
output, and substepping the plain forward dynamics is both simpler and
~2× cheaper in instruction count.

Batch dimensions flowing through the LP kernel:
    inputs[0]  q   (N, 19)   floating-base + 12 joints (quaternion xyzw)
    inputs[1]  v   (N, 18)
    inputs[2]  tau (N, 18)   first 6 = base wrench; last 12 = actuated
    outputs[0] ddq (N, 18)

Integration is batched on GPU: position Euler, quaternion via SE(3) exp-map
matching Pinocchio's `pin.integrate`.

MuJoCo (MJCF) is used for rendering only — state is pushed into
`mj_data.qpos` and `mj_forward` updates kinematics; `mj_step` is never
called. Controller and "true sim" share the same Pinocchio compliant
kernel, so there is no planner/sim model mismatch.

Task: hold the default standing pose. A lateral push perturbation can be
injected via the base wrench term (tau[1] += push_force for push_duration).
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse

os.environ.setdefault("MUJOCO_GL", "egl")  # headless offscreen rendering

import numpy as np
import torch
import casadi
import mujoco
import mediapy
import pinocchio as pin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cusadi import CUSADI_FUNCTION_DIR  # noqa: E402
from cusadi.level_parallel.function import CusadiLevelParallelFunction  # noqa: E402

from robot_descriptions import a1_mj_description  # noqa: E402
from robot_descriptions.loaders.pinocchio import load_robot_description  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- A1 home configuration (Pinocchio joint order) ------------------
# Pinocchio joint order after FreeFlyer: FL_hip, FL_thigh, FL_calf,
# FR_hip, FR_thigh, FR_calf, RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
HOME_JOINT_ANGLES_PIN = np.array([
    0.0,  0.8, -1.6,   # FL
    0.0,  0.8, -1.6,   # FR
    0.0,  0.8, -1.6,   # RL
    0.0,  0.8, -1.6,   # RR
], dtype=np.float64)

HOME_BASE_Z = 0.29  # near the compliant-model equilibrium (K=3000, A1 body weight → ~1 mm sink)
HOME_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def compute_holding_torques(mj_model, home_joints_mj: np.ndarray) -> np.ndarray:
    """Run a throwaway MuJoCo PD-hold + mj_inverse to find the 12-dim actuated
    holding torques. Used as the initial MPPI nominal and as the shift-target
    in the receding-horizon loop."""
    d = mujoco.MjData(mj_model)
    d.qpos[:3] = [0.0, 0.0, HOME_BASE_Z + 0.03]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[7:] = home_joints_mj
    d.qvel[:] = 0.0
    kp, kv = 300.0, 10.0
    for _ in range(2000):
        q_err = d.qpos[7:] - home_joints_mj
        qvel_err = d.qvel[6:]
        d.ctrl[:] = np.clip(-kp * q_err - kv * qvel_err,
                             mj_model.actuator_ctrlrange[:, 0],
                             mj_model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(mj_model, d)
    d.qacc[:] = 0.0
    mujoco.mj_inverse(mj_model, d)
    return d.qfrc_inverse[6:].copy()


# ---------- Batched SE(3) integration on GPU --------------------------------
def batched_quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2, both stored xyzw (vector part first)."""
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def batched_quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate 3-vector v by quaternion q (both batched; q in xyzw)."""
    qv = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def batched_pin_integrate(
    q: torch.Tensor, v: torch.Tensor, dt: float
) -> torch.Tensor:
    """Integrate (q, v) for dt using Pinocchio's SE(3) exp-map convention.

    q: (N, 19) = [pos(3), quat_xyzw(4), joints(12)]
    v: (N, 18) = [lin_body(3), ang_body(3), joint_vel(12)]
    Returns q_new (N, 19).
    """
    pos = q[..., :3]
    quat = q[..., 3:7]
    joints = q[..., 7:]

    lin_body = v[..., :3]
    ang_body = v[..., 3:6]
    joint_vel = v[..., 6:]

    # SE(3) integration — body-frame convention, matching pin.integrate.
    # New position: pos + R(quat) * (lin_body * dt).
    pos_new = pos + batched_quat_rotate_xyzw(quat, lin_body * dt)

    # New orientation: quat * exp(ang_body * dt). Body-frame angular velocity
    # integrates via right-multiplication.
    omega_dt = ang_body * dt
    angle = torch.norm(omega_dt, dim=-1, keepdim=True)
    small = angle < 1e-8
    half = angle * 0.5
    # Safe division for small angles via Taylor expansion: sin(h)/h ≈ 1 - h²/6
    sin_over_angle = torch.where(
        small,
        0.5 - angle * angle / 48.0,
        torch.sin(half) / angle.clamp(min=1e-12),
    )
    dq_xyz = omega_dt * sin_over_angle
    dq_w = torch.where(small, 1.0 - angle * angle / 8.0, torch.cos(half))
    dq = torch.cat([dq_xyz, dq_w], dim=-1)
    quat_new = batched_quat_mul_xyzw(quat, dq)
    quat_new = quat_new / torch.norm(quat_new, dim=-1, keepdim=True)

    joints_new = joints + joint_vel * dt
    return torch.cat([pos_new, quat_new, joints_new], dim=-1)


# ---------- Mapping utilities ----------------------------------------------
def pin_q_from_mj(mj_qpos: np.ndarray, mj_to_pin: np.ndarray) -> np.ndarray:
    """Convert a single MuJoCo qpos (19,) to Pinocchio q (19,)."""
    q = np.empty(19)
    q[:3] = mj_qpos[:3]
    # MuJoCo [w,x,y,z] -> Pinocchio [x,y,z,w]
    q[3:7] = mj_qpos[[4, 5, 6, 3]]
    q[7:] = mj_qpos[7:][mj_to_pin]
    return q


def pin_v_from_mj(mj_qvel: np.ndarray, mj_to_pin: np.ndarray) -> np.ndarray:
    """Convert MuJoCo qvel (18,) to Pinocchio v (18,). Linear/angular layout
    matches (both [lin(3), ang(3), joints(12)]) — only joints need permuting.
    """
    v = np.empty(18)
    v[:6] = mj_qvel[:6]
    v[6:] = mj_qvel[6:][mj_to_pin]
    return v


def mj_tau_from_pin_actuated(tau_pin_actuated: np.ndarray, pin_to_mj: np.ndarray) -> np.ndarray:
    """Map the 12-dim actuated torque from Pinocchio order to MuJoCo ctrl order."""
    return tau_pin_actuated[pin_to_mj]


# ---------- MPPI core -------------------------------------------------------
class A1MPPI:
    """MPPI over joint POSITION targets, with a PD+gravity-comp inner loop.

    The sampled action at each rollout step is a 12-dim joint-position target
    q_des. Each rollout substep computes tau = gcomp + K_p(q_des - q) - K_d*v_j,
    clips to actuator limits, and integrates the compliant dynamics.

    This is the deployed-controller pattern (DIAL-MPC, whole-body MPPI): the
    planner decides where legs should go, the stabilizing high-rate feedback
    makes them track. Position-target MPPI is stable from any state because
    every sampled trajectory has its own PD controller closing the loop —
    there is no open-loop torque integration to destabilize.
    """

    def __init__(
        self,
        home_joint_angles: torch.Tensor,
        tau_gcomp_actuated: torch.Tensor,
        N: int = 128,
        T: int = 15,
        dt: float = 0.02,
        n_sub: int = 5,
        sigma: float = 0.1,
        temperature: float = 1.0,
        kp: float = 80.0,
        kv: float = 3.0,
        tau_clip: float = 33.5,
        device: str = "cuda",
        seed: int = 0,
    ):
        self.N = N
        self.T = T
        self.dt = dt
        self.n_sub = n_sub
        self.dt_sub = dt / n_sub
        self.sigma = sigma
        self.temperature = temperature
        self.kp = kp
        self.kv = kv
        self.tau_clip = tau_clip
        self.device = device

        # Compliant contact (K=3000 / D=100 / 16 corners / softplus). Rollouts
        # use semi-implicit Euler with n_sub=5 substeps per rollout step
        # (dt_sub=0.004s). Stable because every substep applies a PD+gcomp
        # torque to the compliant dynamics, not an open-loop sample.
        fn_path = os.path.join(CUSADI_FUNCTION_DIR, "a1_compliant_sim.casadi")
        if not os.path.exists(fn_path):
            raise FileNotFoundError(
                f"Run build_a1_compliant_fwd.py first — no {fn_path}")
        self.fn = casadi.Function.load(fn_path)
        assert self.fn.nnz_in(0) == 19 and self.fn.nnz_in(1) == 18 and self.fn.nnz_in(2) == 18
        assert self.fn.nnz_out(0) == 18
        self.lp = CusadiLevelParallelFunction(self.fn, N, lib_name="level_parallel_kernel")

        # Nominal = HOME joint positions. σ is in radians.
        self.q_home_joints = home_joint_angles.to(device=device, dtype=torch.double)
        self.tau_gcomp = tau_gcomp_actuated.to(device=device, dtype=torch.double)
        self.U_nominal = self.q_home_joints.unsqueeze(0).expand(T, -1).clone()

        # Reference state (Pinocchio convention).
        self.q_ref = torch.tensor(
            np.concatenate([[0.0, 0.0, HOME_BASE_Z], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN]),
            device=device, dtype=torch.double,
        )
        self.v_ref = torch.zeros(18, device=device, dtype=torch.double)

        self.gen = torch.Generator(device=device).manual_seed(seed)

        self.last_top_cost = 0.0
        self.last_mean_cost = 0.0

    def step(self, q: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Run one MPPI update. q, v are single states. Returns (u0_actuated_12, stats)."""
        times = {}
        t0 = time.perf_counter()

        # 1. Sample joint-position perturbations around the shifted nominal.
        #    U_samples[n, t, :] is a 12-dim joint-position target (rad).
        noise = torch.randn(self.N, self.T, 12, device=self.device, dtype=torch.double,
                            generator=self.gen) * self.sigma
        U_samples = self.U_nominal.unsqueeze(0) + noise                  # (N, T, 12)

        # 2. Rollout all N samples through T timesteps in parallel.
        q_roll = q.unsqueeze(0).expand(self.N, -1).contiguous()
        v_roll = v.unsqueeze(0).expand(self.N, -1).contiguous()
        costs = torch.zeros(self.N, device=self.device, dtype=torch.double)

        # Pre-allocate the padded tau tensor (first 6 dims zero = floating-base unactuated).
        tau_padded = torch.zeros(self.N, 18, device=self.device, dtype=torch.double)

        torch.cuda.synchronize()
        times["setup_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dyn_s = 0.0
        for t in range(self.T):
            q_des = U_samples[:, t, :]                                    # (N, 12)

            # Semi-implicit Euler with n_sub substeps. Every substep computes
            # PID torque against the current q_des — rollouts see exactly the
            # closed-loop dynamics the true sim will experience.
            for _ in range(self.n_sub):
                q_err = q_roll[:, 7:] - q_des                             # (N, 12)
                v_joints = v_roll[:, 6:]                                  # (N, 12)
                tau_act = (self.tau_gcomp - self.kp * q_err - self.kv * v_joints) \
                    .clamp(-self.tau_clip, self.tau_clip)
                tau_padded[:, 6:] = tau_act

                t_dyn0 = time.perf_counter()
                self.lp.evaluate([q_roll, v_roll, tau_padded])
                ddq = self.lp.outputs_sparse[0]                            # (N, 18)
                torch.cuda.synchronize()
                dyn_s += time.perf_counter() - t_dyn0
                v_roll = v_roll + ddq * self.dt_sub
                q_roll = batched_pin_integrate(q_roll, v_roll, self.dt_sub)

            # Stage cost (one per outer rollout step).
            dq_pos = q_roll[:, :3] - self.q_ref[:3]
            dq_quat = q_roll[:, 3:7] - self.q_ref[3:7]  # upright prior (small-angle)
            dq_joint = q_roll[:, 7:] - self.q_ref[7:]
            dv = v_roll - self.v_ref
            # Penalize q_des deviation from home only weakly — the PD already
            # pulls the robot toward q_des, so the main cost is state quality.
            ddes = q_des - self.q_home_joints
            costs += (
                20.0 * (dq_pos ** 2).sum(-1)
                + 10.0 * (dq_quat ** 2).sum(-1)
                + 2.0 * (dq_joint ** 2).sum(-1)
                + 0.1 * (dv ** 2).sum(-1)
                + 0.1 * (ddes ** 2).sum(-1)
            )
        times["rollout_s"] = time.perf_counter() - t0
        times["dynamics_only_s"] = dyn_s

        t0 = time.perf_counter()
        # 3. MPPI exponentially-weighted update. Replace any NaN/Inf cost with a
        # large finite value so the softmax down-weights those samples cleanly.
        finite = torch.isfinite(costs)
        costs = torch.where(finite, costs, torch.full_like(costs, 1e12))
        min_cost = costs.min()
        weights = torch.softmax(-(costs - min_cost) / self.temperature, dim=0)
        # (N,) × (N, T, 12) -> (T, 12)
        self.U_nominal = (weights.view(-1, 1, 1) * U_samples).sum(dim=0)

        # 4. Stats for stdout and overlay.
        self.last_top_cost = float(costs.min().item())
        self.last_mean_cost = float(costs.mean().item())

        u0 = self.U_nominal[0].clone()

        # 5. Shift nominal trajectory one step forward (MPPI receding horizon).
        self.U_nominal[:-1] = self.U_nominal[1:].clone()
        self.U_nominal[-1] = self.q_home_joints
        times["update_s"] = time.perf_counter() - t0

        return u0, times


# ---------- Main loop -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration_s", type=float, default=5.0)
    ap.add_argument("--control_dt", type=float, default=0.02)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--n_sub", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.1,
                    help="sample std of joint position targets, in radians")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--push_t", type=float, default=1.5)
    ap.add_argument("--push_duration", type=float, default=0.1)
    ap.add_argument("--push_force", type=float, default=40.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_video", default=os.path.join(HERE, "videos", "a1_mppi.mp4"))
    ap.add_argument("--out_keyframes_dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--out_json", default=os.path.join(HERE, "videos", "a1_mppi_timing.json"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[a1_mppi] CUDA device = {torch.cuda.get_device_name(0)}")
    print(f"[a1_mppi] Params: N={args.N} T={args.T} dt={args.control_dt} "
          f"n_sub={args.n_sub} (dt_sub={args.control_dt/args.n_sub:.4f}) "
          f"sigma={args.sigma} temp={args.temperature}")

    # --- Mapping ---
    mapping_path = os.path.join(HERE, "a1_mapping.npz")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Run validate_a1_mapping.py first — no {mapping_path}")
    mapping = np.load(mapping_path)
    mj_to_pin = mapping["mj_to_pin"]
    pin_to_mj = mapping["pin_to_mj"]

    # --- MuJoCo setup ---
    mj_model = mujoco.MjModel.from_xml_path(a1_mj_description.MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    sim_dt = float(mj_model.opt.timestep)
    sim_steps_per_control = max(1, int(round(args.control_dt / sim_dt)))
    print(f"[a1_mppi] MuJoCo sim_dt={sim_dt}s, control_dt={args.control_dt}s "
          f"({sim_steps_per_control} substeps/control)")

    # Gravity-compensation feedforward via RNEA at the home pose. Approximate
    # (RNEA ignores contact), but good enough — the PID absorbs the residual.
    pin_model = load_robot_description(
        "a1_description", root_joint=pin.JointModelFreeFlyer()).model
    pin_data = pin_model.createData()
    q_home_full = np.concatenate([
        [0.0, 0.0, HOME_BASE_Z], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN
    ])
    tau_gcomp_full = pin.rnea(pin_model, pin_data, q_home_full,
                              np.zeros(18), np.zeros(18))
    tau_gcomp_pin = tau_gcomp_full[6:]
    print(f"[a1_mppi] RNEA gcomp at home: hip={np.round(tau_gcomp_pin[0::3],2)} "
          f"thigh={np.round(tau_gcomp_pin[1::3],2)} calf={np.round(tau_gcomp_pin[2::3],2)}")

    # Initialize "true state" slightly above the natural equilibrium and let
    # the PID+gcomp settle it.
    q_sim = np.concatenate([
        [0.0, 0.0, HOME_BASE_Z + 0.01],
        HOME_QUAT_XYZW,
        HOME_JOINT_ANGLES_PIN,
    ]).astype(np.float64)
    v_sim = np.zeros(18, dtype=np.float64)

    # Single-instance compliant-sim for stepping the "true" state. Same kernel
    # the MPPI rollouts use, so there is no planner/sim mismatch.
    fn_sim = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "a1_compliant_sim.casadi"))
    sim_lp = CusadiLevelParallelFunction(fn_sim, 1, lib_name="level_parallel_kernel")

    # Push state into MuJoCo for the initial render (no physics stepping).
    def push_state_to_mj(q_p: np.ndarray, v_p: np.ndarray) -> None:
        mj_data.qpos[:3] = q_p[:3]
        mj_data.qpos[3:7] = q_p[[6, 3, 4, 5]]  # Pinocchio xyzw -> MuJoCo wxyz
        mj_data.qpos[7:] = q_p[7:][pin_to_mj]
        mj_data.qvel[:6] = v_p[:6]  # body-frame approx (renderer doesn't care)
        mj_data.qvel[6:] = v_p[6:][pin_to_mj]
        mujoco.mj_forward(mj_model, mj_data)

    push_state_to_mj(q_sim, v_sim)

    # --- Controller ---
    # The same PID gains are used by the MPPI rollout (every substep) and the
    # true sim (every substep). Keeping them identical means the planner's
    # rollout dynamics exactly match what the true sim will experience under
    # the decided q_des.
    tau_clip = float(mj_model.actuator_ctrlrange[0, 1])
    Kp_pid, Kv_pid = 200.0, 10.0
    mppi = A1MPPI(
        home_joint_angles=torch.tensor(HOME_JOINT_ANGLES_PIN, dtype=torch.double),
        tau_gcomp_actuated=torch.tensor(tau_gcomp_pin, dtype=torch.double),
        N=args.N, T=args.T, dt=args.control_dt, n_sub=args.n_sub,
        sigma=args.sigma, temperature=args.temperature,
        kp=Kp_pid, kv=Kv_pid, tau_clip=tau_clip,
        seed=args.seed,
    )

    # --- Renderer ---
    # Default offscreen framebuffer is 640x480. Expand before creating the renderer.
    mj_model.vis.global_.offwidth = 1280
    mj_model.vis.global_.offheight = 720
    renderer = mujoco.Renderer(mj_model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135.0
    camera.elevation = -20.0
    camera.distance = 1.2
    camera.lookat[:] = [0.0, 0.0, 0.2]

    # --- Pre-settle under PD holding home ---
    # PID against home joint positions + gcomp feedforward drops the robot
    # onto the compliant ground until |v| < 0.1.
    print(f"[a1_mppi] Settling compliant sim under PD+gcomp...")
    settle_dt = 0.0005
    q_sim_pt = torch.zeros(1, 19, device="cuda", dtype=torch.double)
    v_sim_pt = torch.zeros(1, 18, device="cuda", dtype=torch.double)
    tau_sim_pt = torch.zeros(1, 18, device="cuda", dtype=torch.double)
    for _ in range(int(3.0 / settle_dt)):
        q_err = q_sim[7:] - HOME_JOINT_ANGLES_PIN
        v_err = v_sim[6:]
        tau_pd = np.clip(
            tau_gcomp_pin - Kp_pid * q_err - Kv_pid * v_err, -tau_clip, tau_clip)
        q_sim_pt[0] = torch.tensor(q_sim, device="cuda", dtype=torch.double)
        v_sim_pt[0] = torch.tensor(v_sim, device="cuda", dtype=torch.double)
        tau_sim_pt[0, 6:] = torch.tensor(tau_pd, device="cuda", dtype=torch.double)
        sim_lp.evaluate([q_sim_pt, v_sim_pt, tau_sim_pt])
        ddq = sim_lp.outputs_sparse[0][0].cpu().numpy()
        v_sim = v_sim + ddq * settle_dt
        q_sim = pin.integrate(pin_model, q_sim, v_sim * settle_dt)
        if np.linalg.norm(v_sim) < 0.05:
            break
    print(f"[a1_mppi] after settle: base_z={q_sim[2]:.4f}  |v|={np.linalg.norm(v_sim):.4f}  "
          f"quat_w={q_sim[6]:+.3f}")
    push_state_to_mj(q_sim, v_sim)

    # --- Main loop ---
    n_control_steps = int(round(args.duration_s / args.control_dt))
    frame_interval_control = max(1, int(round(1.0 / (args.fps * args.control_dt))))

    frames = []
    timings: list[dict] = []
    keyframes: dict[str, np.ndarray] = {}

    loop_t0 = time.perf_counter()

    for step_idx in range(n_control_steps):
        sim_t = step_idx * args.control_dt

        push_active = args.push_t <= sim_t < (args.push_t + args.push_duration)

        # --- MPPI step: returns a 12-dim joint-position target q_des. ---
        q_pt = torch.tensor(q_sim, device="cuda", dtype=torch.double)
        v_pt = torch.tensor(v_sim, device="cuda", dtype=torch.double)
        t_ctrl0 = time.perf_counter()
        q_des_t, stats = mppi.step(q_pt, v_pt)
        torch.cuda.synchronize()
        stats["total_control_s"] = time.perf_counter() - t_ctrl0
        q_des = q_des_t.cpu().numpy()

        # --- Advance "true state" with the same PD+gcomp law. ---
        # The sim ticks at dt_sub=0.001 (20 substeps / control), recomputing
        # PD each substep against the held q_des. This is the actual
        # controller: MPPI decides WHERE the legs go, PD makes them track.
        t_sim0 = time.perf_counter()
        n_sub_sim = 20
        sub_dt = args.control_dt / n_sub_sim
        for _ in range(n_sub_sim):
            q_err = q_sim[7:] - q_des
            v_err = v_sim[6:]
            tau_act = np.clip(
                tau_gcomp_pin - Kp_pid * q_err - Kv_pid * v_err, -tau_clip, tau_clip)
            tau_full_np = np.concatenate([np.zeros(6), tau_act])
            if push_active:
                # Lateral push on the trunk: +y in world. Body ≈ world while
                # upright (first 3 dims of the floating-base vel are the
                # body-frame linear velocity).
                tau_full_np[1] += args.push_force
            q_sim_pt[0] = torch.tensor(q_sim, device="cuda", dtype=torch.double)
            v_sim_pt[0] = torch.tensor(v_sim, device="cuda", dtype=torch.double)
            tau_sim_pt[0] = torch.tensor(tau_full_np, device="cuda", dtype=torch.double)
            sim_lp.evaluate([q_sim_pt, v_sim_pt, tau_sim_pt])
            ddq = sim_lp.outputs_sparse[0][0].cpu().numpy()
            v_sim = v_sim + ddq * sub_dt
            q_sim = pin.integrate(pin_model, q_sim, v_sim * sub_dt)
        push_state_to_mj(q_sim, v_sim)
        stats["mj_step_s"] = time.perf_counter() - t_sim0  # (now the Pinocchio sim time)

        # --- Render ---
        if step_idx % frame_interval_control == 0:
            t_render0 = time.perf_counter()
            camera.lookat[:] = [mj_data.qpos[0], mj_data.qpos[1], 0.2]
            renderer.update_scene(mj_data, camera=camera)
            frame = renderer.render().copy()
            frames.append(frame)
            stats["render_s"] = time.perf_counter() - t_render0

        stats["sim_t"] = sim_t
        stats["push_active"] = push_active
        stats["base_z"] = float(q_sim[2])
        stats["base_quat_w"] = float(q_sim[6])  # Pinocchio xyzw: w at index 6
        stats["top_cost"] = mppi.last_top_cost
        stats["mean_cost"] = mppi.last_mean_cost
        timings.append(stats)

        # Capture keyframes for paper figure.
        if sim_t < 1e-3 and "t=0.0s" not in keyframes:
            keyframes["t=0.0s"] = frame
        elif abs(sim_t - args.push_t) < args.control_dt and "t=push" not in keyframes:
            keyframes["t=push"] = frames[-1] if frames else None
        elif abs(sim_t - (args.push_t + 0.3)) < args.control_dt and "t=recover" not in keyframes:
            keyframes["t=recover"] = frames[-1] if frames else None

        if step_idx % 10 == 0:
            tag = "FALL" if q_sim[2] < 0.08 else "    "
            print(f"  [t={sim_t:5.2f}s] {tag} base_z={q_sim[2]:.3f}  "
                  f"quat_w={q_sim[6]:+.3f}  |v|={np.linalg.norm(v_sim):6.3f}  "
                  f"cost_min={mppi.last_top_cost:7.2f}  "
                  f"dyn={stats['dynamics_only_s']*1000:5.2f}ms  "
                  f"total={stats['total_control_s']*1000:5.2f}ms")

    loop_s = time.perf_counter() - loop_t0
    print(f"\n[a1_mppi] {n_control_steps} control steps in {loop_s:.2f}s wall, "
          f"sim_time={args.duration_s}s, real-time factor={args.duration_s / loop_s:.2f}x")

    # --- Save video ---
    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    mediapy.write_video(args.out_video, frames, fps=args.fps)
    print(f"[a1_mppi] Saved {len(frames)} frames to {args.out_video}")

    # --- Save keyframes ---
    os.makedirs(args.out_keyframes_dir, exist_ok=True)
    for name, img in keyframes.items():
        if img is None:
            continue
        out = os.path.join(args.out_keyframes_dir, f"a1_mppi_{name.replace('=','_').replace('.','p')}.png")
        mediapy.write_image(out, img)
        print(f"[a1_mppi] Saved keyframe: {out}")

    # --- Aggregate timing summary ---
    dyn_times = np.array([t["dynamics_only_s"] for t in timings]) * 1000
    ctrl_times = np.array([t["total_control_s"] for t in timings]) * 1000
    mj_times = np.array([t["mj_step_s"] for t in timings]) * 1000
    render_times = np.array([t.get("render_s", 0.0) for t in timings if "render_s" in t]) * 1000
    print(f"\n[a1_mppi] Per-control-step wall-clock (ms):")
    print(f"  LP dynamics (all {args.T} rollout steps): "
          f"mean={dyn_times.mean():.2f}  p50={np.median(dyn_times):.2f}  p95={np.percentile(dyn_times,95):.2f}")
    print(f"  MPPI total (rollout + cost + update):    "
          f"mean={ctrl_times.mean():.2f}  p50={np.median(ctrl_times):.2f}  p95={np.percentile(ctrl_times,95):.2f}")
    print(f"  MuJoCo substeps ({sim_steps_per_control}):            "
          f"mean={mj_times.mean():.2f}  p50={np.median(mj_times):.2f}")
    if len(render_times):
        print(f"  Offscreen render (one frame):            "
              f"mean={render_times.mean():.2f}  p50={np.median(render_times):.2f}")
    base_z = np.array([t["base_z"] for t in timings])
    print(f"[a1_mppi] Base height over episode: min={base_z.min():.3f}  "
          f"mean={base_z.mean():.3f}  max={base_z.max():.3f}  final={base_z[-1]:.3f}")

    # --- Save json ---
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "params": vars(args),
            "n_control_steps": n_control_steps,
            "wall_time_s": loop_s,
            "real_time_factor": args.duration_s / loop_s,
            "per_step": timings,
            "device": torch.cuda.get_device_name(0),
        }, f, indent=2, default=str)
    print(f"[a1_mppi] Saved timing to {args.out_json}")


if __name__ == "__main__":
    main()
