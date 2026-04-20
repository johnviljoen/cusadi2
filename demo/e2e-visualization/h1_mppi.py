"""
H1 MPPI with LP-batched compliant-contact forward dynamics.

Same structure as a1_mppi.py, adapted for H1:
 - 19 actuated joints (vs 12 on A1), nv=25
 - 2 feet, 4 corners each
 - dt_sub=0.002 (vs A1's 0.004) — stiffer dynamics need smaller substep
 - Higher PID gains (Kp=500, Kv=20) — scale with joint inertia

MPPI samples q_des (19-dim joint position targets). A PID+gcomp inner loop
runs every rollout substep AND every true-sim substep, so the planner and
the sim see identical dynamics.

Rendering is omitted here — a1_mppi.py has the MuJoCo pipeline, and the H1
MuJoCo joint mapping has not been validated in this demo. Add rendering as
a follow-up once balance is working.
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import casadi
import pinocchio as pin

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src import CUSADI_FUNCTION_DIR  # noqa: E402
from src.level_parallel.function import CusadiLevelParallelFunction  # noqa: E402

from robot_descriptions.loaders.pinocchio import load_robot_description  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- H1 home configuration ------------------------------------------
# 19 actuated joints in Pinocchio order:
#   [0] left_hip_yaw    [1] left_hip_roll   [2] left_hip_pitch
#   [3] left_knee       [4] left_ankle
#   [5] right_hip_yaw   [6] right_hip_roll  [7] right_hip_pitch
#   [8] right_knee      [9] right_ankle
#   [10] torso          [11..14] left arm   [15..18] right arm
HOME_JOINT_ANGLES_PIN = np.zeros(19)
HOME_JOINT_ANGLES_PIN[2] = -0.4   # left_hip_pitch (thighs forward)
HOME_JOINT_ANGLES_PIN[3] =  0.8   # left_knee (bent)
HOME_JOINT_ANGLES_PIN[4] = -0.4   # left_ankle (foot flat)
HOME_JOINT_ANGLES_PIN[7] = -0.4
HOME_JOINT_ANGLES_PIN[8] =  0.8
HOME_JOINT_ANGLES_PIN[9] = -0.4

# Widen stance (hip roll) for a larger support polygon — H1 is harder to
# balance than A1; the widened stance gives MPPI more margin.
HOME_JOINT_ANGLES_PIN[1] = +0.05   # left hip roll (leg splays out)
HOME_JOINT_ANGLES_PIN[6] = -0.05   # right hip roll

# At home joints with K=5000, D=150, 8 corners, mass=51.6 kg, the compliant
# equilibrium is at base_z ≈ 0.978 with ~12 mm corner penetration. Start the
# sim exactly at the predicted equilibrium so there is no drop transient
# before the balance feedback takes over.
HOME_BASE_Z = 0.978
HOME_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


# ---------- Batched SE(3) integration (identical to a1_mppi.py) ------------
def batched_quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def batched_quat_rotate_xyzw(q, v):
    qv, qw = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def batched_pin_integrate(q, v, dt):
    pos, quat, joints = q[..., :3], q[..., 3:7], q[..., 7:]
    lin, ang, jv = v[..., :3], v[..., 3:6], v[..., 6:]
    pos_new = pos + batched_quat_rotate_xyzw(quat, lin * dt)
    omega_dt = ang * dt
    angle = torch.norm(omega_dt, dim=-1, keepdim=True)
    small = angle < 1e-8
    half = angle * 0.5
    sin_over = torch.where(small, 0.5 - angle * angle / 48.0,
                           torch.sin(half) / angle.clamp(min=1e-12))
    dq_xyz = omega_dt * sin_over
    dq_w = torch.where(small, 1.0 - angle * angle / 8.0, torch.cos(half))
    quat_new = batched_quat_mul_xyzw(quat, torch.cat([dq_xyz, dq_w], dim=-1))
    quat_new = quat_new / torch.norm(quat_new, dim=-1, keepdim=True)
    return torch.cat([pos_new, quat_new, joints + jv * dt], dim=-1)


# ---------- MPPI core -------------------------------------------------------
class H1MPPI:
    """MPPI over 19-dim joint position targets with PD+gcomp+balance inner loop.

    Balance feedback maps base orientation → ankle / hip_roll torques (the
    ZMP-inspired law every real biped uses at the low level). Base pitch
    drives symmetric ankle-pitch corrections; base roll drives symmetric
    hip-roll corrections. Applied IDENTICALLY inside every rollout substep
    and every true-sim substep.
    """

    # Indices into the 19-dim actuated joint vector (Pinocchio order).
    IDX_LHIP_ROLL = 1
    IDX_LANKLE    = 4
    IDX_RHIP_ROLL = 6
    IDX_RANKLE    = 9

    def __init__(
        self,
        home_joint_angles: torch.Tensor,
        tau_gcomp_actuated: torch.Tensor,
        N: int = 128,
        T: int = 10,
        dt: float = 0.02,
        n_sub: int = 10,
        sigma: float = 0.08,
        temperature: float = 1.0,
        kp: float = 500.0,
        kv: float = 20.0,
        k_bal_p: float = 250.0,
        k_bal_d: float = 25.0,
        k_roll_p: float = 150.0,
        k_roll_d: float = 15.0,
        tau_clip: float = 200.0,
        device: str = "cuda",
        seed: int = 0,
    ):
        self.N, self.T, self.dt = N, T, dt
        self.n_sub = n_sub
        self.dt_sub = dt / n_sub
        self.sigma = sigma
        self.temperature = temperature
        self.kp, self.kv = kp, kv
        self.k_bal_p, self.k_bal_d = k_bal_p, k_bal_d
        self.k_roll_p, self.k_roll_d = k_roll_p, k_roll_d
        self.tau_clip = tau_clip
        self.device = device

        fn_path = os.path.join(CUSADI_FUNCTION_DIR, "h1_compliant_sim.casadi")
        if not os.path.exists(fn_path):
            raise FileNotFoundError(f"Run build_h1_compliant_fwd.py first — no {fn_path}")
        self.fn = casadi.Function.load(fn_path)
        assert self.fn.nnz_in(0) == 26 and self.fn.nnz_in(1) == 25 and self.fn.nnz_in(2) == 25
        assert self.fn.nnz_out(0) == 25
        self.lp = CusadiLevelParallelFunction(self.fn, N, lib_name="level_parallel_kernel")

        self.q_home_joints = home_joint_angles.to(device=device, dtype=torch.double)
        self.tau_gcomp = tau_gcomp_actuated.to(device=device, dtype=torch.double)
        self.U_nominal = self.q_home_joints.unsqueeze(0).expand(T, -1).clone()

        # Reference state.
        self.q_ref = torch.tensor(
            np.concatenate([[0.0, 0.0, HOME_BASE_Z], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN]),
            device=device, dtype=torch.double,
        )
        self.v_ref = torch.zeros(25, device=device, dtype=torch.double)

        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.last_top_cost = 0.0
        self.last_mean_cost = 0.0

    def step(self, q: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, dict]:
        times = {}
        t0 = time.perf_counter()

        noise = torch.randn(self.N, self.T, 19, device=self.device, dtype=torch.double,
                            generator=self.gen) * self.sigma
        U_samples = self.U_nominal.unsqueeze(0) + noise

        q_roll = q.unsqueeze(0).expand(self.N, -1).contiguous()
        v_roll = v.unsqueeze(0).expand(self.N, -1).contiguous()
        costs = torch.zeros(self.N, device=self.device, dtype=torch.double)
        tau_padded = torch.zeros(self.N, 25, device=self.device, dtype=torch.double)

        torch.cuda.synchronize()
        times["setup_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dyn_s = 0.0
        for t in range(self.T):
            q_des = U_samples[:, t, :]                                    # (N, 19)

            for _ in range(self.n_sub):
                q_err = q_roll[:, 7:] - q_des
                v_joints = v_roll[:, 6:]
                tau_act = self.tau_gcomp - self.kp * q_err - self.kv * v_joints

                # Base-orientation balance feedback. Small-angle:
                #   roll  ≈ 2·q_x,   pitch ≈ 2·q_y
                # ω_body ≈ v[3:6]. Ankles symmetrically absorb base pitch;
                # hip rolls symmetrically absorb base roll.
                base_roll  = 2.0 * q_roll[:, 3]     # (N,)
                base_pitch = 2.0 * q_roll[:, 4]
                omega_x = v_roll[:, 3]
                omega_y = v_roll[:, 4]
                pitch_corr = -self.k_bal_p * base_pitch - self.k_bal_d * omega_y
                roll_corr  = -self.k_roll_p * base_roll - self.k_roll_d * omega_x
                tau_act = tau_act.clone()  # avoid broadcast write on the subtracted view
                tau_act[:, self.IDX_LANKLE]    += pitch_corr
                tau_act[:, self.IDX_RANKLE]    += pitch_corr
                tau_act[:, self.IDX_LHIP_ROLL] += roll_corr
                tau_act[:, self.IDX_RHIP_ROLL] += roll_corr

                tau_act = tau_act.clamp(-self.tau_clip, self.tau_clip)
                tau_padded[:, 6:] = tau_act

                t_dyn0 = time.perf_counter()
                self.lp.evaluate([q_roll, v_roll, tau_padded])
                ddq = self.lp.outputs_sparse[0]
                torch.cuda.synchronize()
                dyn_s += time.perf_counter() - t_dyn0
                v_roll = v_roll + ddq * self.dt_sub
                q_roll = batched_pin_integrate(q_roll, v_roll, self.dt_sub)

            # Stage cost. Heavy weight on upright and base_z; lighter on joints.
            dq_pos = q_roll[:, :3] - self.q_ref[:3]
            dq_quat = q_roll[:, 3:7] - self.q_ref[3:7]
            dq_joint = q_roll[:, 7:] - self.q_ref[7:]
            dv = v_roll - self.v_ref
            ddes = q_des - self.q_home_joints
            costs += (
                50.0 * (dq_pos ** 2).sum(-1)
                + 100.0 * (dq_quat ** 2).sum(-1)    # stay upright — top priority
                + 1.0 * (dq_joint ** 2).sum(-1)
                + 0.05 * (dv ** 2).sum(-1)
                + 0.1 * (ddes ** 2).sum(-1)
            )
        times["rollout_s"] = time.perf_counter() - t0
        times["dynamics_only_s"] = dyn_s

        t0 = time.perf_counter()
        finite = torch.isfinite(costs)
        costs = torch.where(finite, costs, torch.full_like(costs, 1e12))
        min_cost = costs.min()
        weights = torch.softmax(-(costs - min_cost) / self.temperature, dim=0)
        self.U_nominal = (weights.view(-1, 1, 1) * U_samples).sum(dim=0)

        self.last_top_cost = float(costs.min().item())
        self.last_mean_cost = float(costs.mean().item())

        u0 = self.U_nominal[0].clone()
        self.U_nominal[:-1] = self.U_nominal[1:].clone()
        self.U_nominal[-1] = self.q_home_joints
        times["update_s"] = time.perf_counter() - t0
        return u0, times


# ---------- Main loop -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration_s", type=float, default=3.0)
    ap.add_argument("--control_dt", type=float, default=0.02)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--n_sub", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.08,
                    help="sample std of joint position targets, in radians")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default=os.path.join(HERE, "videos", "h1_mppi_timing.json"))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[h1_mppi] CUDA device = {torch.cuda.get_device_name(0)}")
    print(f"[h1_mppi] Params: N={args.N} T={args.T} dt={args.control_dt} "
          f"n_sub={args.n_sub} (dt_sub={args.control_dt/args.n_sub:.4f}) "
          f"sigma={args.sigma} temp={args.temperature}")

    pin_model = load_robot_description("h1_description",
                                        root_joint=pin.JointModelFreeFlyer()).model
    pin_data = pin_model.createData()
    q_home_full = np.concatenate([
        [0.0, 0.0, HOME_BASE_Z], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN
    ])
    tau_gcomp_full = pin.rnea(pin_model, pin_data, q_home_full,
                              np.zeros(25), np.zeros(25))
    tau_gcomp_pin = tau_gcomp_full[6:]
    print(f"[h1_mppi] RNEA gcomp at home: |tau|_max={np.abs(tau_gcomp_pin).max():.2f}  "
          f"mean={np.abs(tau_gcomp_pin).mean():.2f}")

    # Initialize at the predicted compliant equilibrium so contact springs
    # are already engaged — no drop transient.
    q_sim = q_home_full.astype(np.float64).copy()
    v_sim = np.zeros(25, dtype=np.float64)

    # Sim kernel (batch=1).
    fn_sim = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, "h1_compliant_sim.casadi"))
    sim_lp = CusadiLevelParallelFunction(fn_sim, 1, lib_name="level_parallel_kernel")

    Kp_pid, Kv_pid = 500.0, 40.0    # heavy joint damping
    Kp_bal, Kd_bal = 600.0, 40.0    # ankle pitch ← base pitch
    Kp_roll, Kd_roll = 400.0, 30.0  # hip roll ← base roll
    tau_clip = 400.0                # H1 actuators have ~200-400 Nm limits
    mppi = H1MPPI(
        home_joint_angles=torch.tensor(HOME_JOINT_ANGLES_PIN, dtype=torch.double),
        tau_gcomp_actuated=torch.tensor(tau_gcomp_pin, dtype=torch.double),
        N=args.N, T=args.T, dt=args.control_dt, n_sub=args.n_sub,
        sigma=args.sigma, temperature=args.temperature,
        kp=Kp_pid, kv=Kv_pid,
        k_bal_p=Kp_bal, k_bal_d=Kd_bal,
        k_roll_p=Kp_roll, k_roll_d=Kd_roll,
        tau_clip=tau_clip, seed=args.seed,
    )

    # Pre-allocate sim tensors (reused each substep).
    q_sim_pt = torch.zeros(1, 26, device="cuda", dtype=torch.double)
    v_sim_pt = torch.zeros(1, 25, device="cuda", dtype=torch.double)
    tau_sim_pt = torch.zeros(1, 25, device="cuda", dtype=torch.double)

    # ---- Pre-settle: PID on HOME + balance feedback (no MPPI), until |v| < 0.5 ----
    print(f"[h1_mppi] Settling with PD+balance on HOME for up to 3s...")
    settle_dt = 0.0005
    for step in range(int(3.0 / settle_dt)):
        q_err = q_sim[7:] - HOME_JOINT_ANGLES_PIN
        v_err = v_sim[6:]
        tau_act = tau_gcomp_pin - Kp_pid * q_err - Kv_pid * v_err
        # balance
        pitch_corr = -Kp_bal * (2.0 * q_sim[4]) - Kd_bal * v_sim[4]
        roll_corr  = -Kp_roll * (2.0 * q_sim[3]) - Kd_roll * v_sim[3]
        tau_act[H1MPPI.IDX_LANKLE]    += pitch_corr
        tau_act[H1MPPI.IDX_RANKLE]    += pitch_corr
        tau_act[H1MPPI.IDX_LHIP_ROLL] += roll_corr
        tau_act[H1MPPI.IDX_RHIP_ROLL] += roll_corr
        tau_act = np.clip(tau_act, -tau_clip, tau_clip)
        tau_full = np.concatenate([np.zeros(6), tau_act])
        q_sim_pt[0] = torch.tensor(q_sim, device="cuda", dtype=torch.double)
        v_sim_pt[0] = torch.tensor(v_sim, device="cuda", dtype=torch.double)
        tau_sim_pt[0] = torch.tensor(tau_full, device="cuda", dtype=torch.double)
        sim_lp.evaluate([q_sim_pt, v_sim_pt, tau_sim_pt])
        ddq = sim_lp.outputs_sparse[0][0].cpu().numpy()
        v_sim = v_sim + ddq * settle_dt
        q_sim = pin.integrate(pin_model, q_sim, v_sim * settle_dt)
        if step > 100 and np.linalg.norm(v_sim) < 0.5 and abs(q_sim[6] - 1.0) < 1e-3:
            break
    print(f"[h1_mppi] after settle (step {step}): base_z={q_sim[2]:.4f}  "
          f"|v|={np.linalg.norm(v_sim):.3f}  quat_w={q_sim[6]:+.4f}")

    n_control_steps = int(round(args.duration_s / args.control_dt))
    timings: list[dict] = []
    loop_t0 = time.perf_counter()

    for step_idx in range(n_control_steps):
        sim_t = step_idx * args.control_dt

        # MPPI step: get 19-dim q_des target.
        q_pt = torch.tensor(q_sim, device="cuda", dtype=torch.double)
        v_pt = torch.tensor(v_sim, device="cuda", dtype=torch.double)
        t_ctrl0 = time.perf_counter()
        q_des_t, stats = mppi.step(q_pt, v_pt)
        torch.cuda.synchronize()
        stats["total_control_s"] = time.perf_counter() - t_ctrl0
        q_des = q_des_t.cpu().numpy()

        # Advance true sim: n_sub_sim=20 substeps at dt_sub=0.001 (stiffer
        # margin in the sim than in the rollout).
        t_sim0 = time.perf_counter()
        n_sub_sim = 20
        sub_dt = args.control_dt / n_sub_sim
        for _ in range(n_sub_sim):
            q_err = q_sim[7:] - q_des
            v_err = v_sim[6:]
            tau_act = tau_gcomp_pin - Kp_pid * q_err - Kv_pid * v_err

            # Base-orientation balance feedback (same law as MPPI rollout).
            base_roll  = 2.0 * q_sim[3]   # qx
            base_pitch = 2.0 * q_sim[4]   # qy
            omega_x = v_sim[3]
            omega_y = v_sim[4]
            pitch_corr = -Kp_bal * base_pitch - Kd_bal * omega_y
            roll_corr  = -Kp_roll * base_roll - Kd_roll * omega_x
            tau_act[H1MPPI.IDX_LANKLE]    += pitch_corr
            tau_act[H1MPPI.IDX_RANKLE]    += pitch_corr
            tau_act[H1MPPI.IDX_LHIP_ROLL] += roll_corr
            tau_act[H1MPPI.IDX_RHIP_ROLL] += roll_corr

            tau_act = np.clip(tau_act, -tau_clip, tau_clip)
            tau_full = np.concatenate([np.zeros(6), tau_act])
            q_sim_pt[0] = torch.tensor(q_sim, device="cuda", dtype=torch.double)
            v_sim_pt[0] = torch.tensor(v_sim, device="cuda", dtype=torch.double)
            tau_sim_pt[0] = torch.tensor(tau_full, device="cuda", dtype=torch.double)
            sim_lp.evaluate([q_sim_pt, v_sim_pt, tau_sim_pt])
            ddq = sim_lp.outputs_sparse[0][0].cpu().numpy()
            v_sim = v_sim + ddq * sub_dt
            q_sim = pin.integrate(pin_model, q_sim, v_sim * sub_dt)
        stats["sim_s"] = time.perf_counter() - t_sim0
        stats["sim_t"] = sim_t
        stats["base_z"] = float(q_sim[2])
        stats["base_quat_w"] = float(q_sim[6])
        stats["top_cost"] = mppi.last_top_cost
        stats["mean_cost"] = mppi.last_mean_cost
        timings.append(stats)

        if step_idx % 5 == 0:
            tag = "FALL" if q_sim[2] < 0.4 else "    "
            print(f"  [t={sim_t:5.2f}s] {tag} base_z={q_sim[2]:.3f}  "
                  f"quat_w={q_sim[6]:+.3f}  |v|={np.linalg.norm(v_sim):6.2f}  "
                  f"cost_min={mppi.last_top_cost:7.1f}  "
                  f"dyn={stats['dynamics_only_s']*1000:5.1f}ms  "
                  f"total={stats['total_control_s']*1000:5.1f}ms")

    loop_s = time.perf_counter() - loop_t0
    print(f"\n[h1_mppi] {n_control_steps} control steps in {loop_s:.2f}s wall, "
          f"sim_time={args.duration_s}s, real-time factor={args.duration_s/loop_s:.2f}x")

    dyn_times = np.array([t["dynamics_only_s"] for t in timings]) * 1000
    ctrl_times = np.array([t["total_control_s"] for t in timings]) * 1000
    sim_times = np.array([t["sim_s"] for t in timings]) * 1000
    print(f"[h1_mppi] LP dynamics (all T rollout steps): "
          f"mean={dyn_times.mean():.2f}ms p95={np.percentile(dyn_times,95):.2f}ms")
    print(f"[h1_mppi] MPPI total: mean={ctrl_times.mean():.2f}ms  sim: mean={sim_times.mean():.2f}ms")
    base_z = np.array([t["base_z"] for t in timings])
    print(f"[h1_mppi] Base_z: min={base_z.min():.3f} mean={base_z.mean():.3f} "
          f"max={base_z.max():.3f} final={base_z[-1]:.3f}")

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
    print(f"[h1_mppi] Saved timing to {args.out_json}")


if __name__ == "__main__":
    main()
