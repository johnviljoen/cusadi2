"""
Phase 2 validation: integration stability + LP timing for the compliant model.

An A1 at home pose is inherently unstable — no open-loop tau keeps it upright
for more than a second. So the phase-2 tests split two distinct concerns:

 2.1a  Rollout-horizon stability: run MPPI-style open-loop (constant tau_hold)
       for T * dt = 20 * 0.02 = 0.4 s. Check |state| stays bounded — this is
       the envelope MPPI actually rolls through at each control step.

 2.1b  Closed-loop settling: 5 s with PD feedback + gcomp feedforward. This
       mirrors the "true sim" in the MPPI outer loop. Must stabilize upright.

 2.2   GPU batched reproduces CPU to FP tolerance for both regimes.

 2.3   LP eval timing at batch=128 → rollout budget check.
"""
from __future__ import annotations

import os
import sys
import time

import casadi
import numpy as np
import pinocchio as pin
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from robot_descriptions.loaders.pinocchio import load_robot_description  # noqa: E402

from cusadi import CUSADI_FUNCTION_DIR  # noqa: E402
from cusadi.level_parallel.function import CusadiLevelParallelFunction  # noqa: E402


HOME_JOINT_ANGLES_PIN = np.array([
    0.0,  0.8, -1.6,
    0.0,  0.8, -1.6,
    0.0,  0.8, -1.6,
    0.0,  0.8, -1.6,
], dtype=np.float64)
HOME_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
GROUND_TOUCH_BASE_Z = 0.279


def make_q(base_z: float) -> np.ndarray:
    return np.concatenate([[0.0, 0.0, base_z], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN])


def newton_equilibrium(f: casadi.Function, v_zero: np.ndarray, init_z: float = 0.25):
    """Least-squares Newton to find (base_z, tau_act) that drives |ddq| to its
    range-space minimum at home pose, zero velocity. Same pattern as the
    existing a1_mppi.py. The 18-dim residual can't reach zero with 13 unknowns
    — the ~2 rad/s² base roll/pitch/xy residual is null-space drift."""
    state = np.concatenate([[init_z], np.zeros(12)])
    def _r(s):
        q = np.concatenate([[0, 0, s[0]], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN])
        tau = np.concatenate([np.zeros(6), s[1:]])
        return np.array(f(q, v_zero, tau)).flatten()
    for _ in range(60):
        r = _r(state)
        if np.linalg.norm(r) < 1e-7:
            break
        J = np.empty((18, 13))
        for k in range(13):
            sp = state.copy(); sp[k] += 1e-5
            J[:, k] = (_r(sp) - r) / 1e-5
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        state = state + 0.5 * step
    return float(state[0]), state[1:].copy(), float(np.linalg.norm(_r(state)))


# Batched SE(3) integration.
def _quat_mul(q1, q2):
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def _quat_rot(q, v):
    qv, qw = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def batched_integrate(q, v, dt):
    pos, quat, joints = q[..., :3], q[..., 3:7], q[..., 7:]
    lin, ang, jv = v[..., :3], v[..., 3:6], v[..., 6:]
    pos_new = pos + _quat_rot(quat, lin * dt)
    omega_dt = ang * dt
    angle = torch.norm(omega_dt, dim=-1, keepdim=True)
    small = angle < 1e-8
    half = angle * 0.5
    sin_over = torch.where(small, 0.5 - angle * angle / 48.0,
                           torch.sin(half) / angle.clamp(min=1e-12))
    dq_xyz = omega_dt * sin_over
    dq_w = torch.where(small, 1.0 - angle * angle / 8.0, torch.cos(half))
    quat_new = _quat_mul(quat, torch.cat([dq_xyz, dq_w], dim=-1))
    quat_new = quat_new / torch.norm(quat_new, dim=-1, keepdim=True)
    return torch.cat([pos_new, quat_new, joints + jv * dt], dim=-1)


def cpu_rollout_open(f, model, tau_hold, dt_sub, n_steps, initial_drop=0.02, init_z=0.2898):
    """Open-loop rollout (constant tau = tau_hold, no feedback) — MPPI-style."""
    q = make_q(init_z + initial_drop)
    v = np.zeros(18)
    tau = np.concatenate([np.zeros(6), tau_hold])
    zs, quats, vns = [], [], []
    nan_at = -1
    for step in range(n_steps):
        ddq = np.array(f(q, v, tau)).flatten()
        if not np.all(np.isfinite(ddq)):
            nan_at = step; break
        v = v + ddq * dt_sub
        q = pin.integrate(model, q, v * dt_sub)
        zs.append(q[2]); quats.append(q[6]); vns.append(float(np.linalg.norm(v)))
    return {"zs": np.array(zs), "quats": np.array(quats), "vns": np.array(vns),
            "nan_at": nan_at, "q": q, "v": v}


def cpu_rollout_closed(f, model, tau_hold, dt_sub, n_steps,
                       kp=300.0, kv=15.0, tau_clip=33.5,
                       initial_drop=0.02, init_z=0.2898):
    """Closed-loop rollout with joint-PD + gcomp feedforward."""
    q = make_q(init_z + initial_drop)
    v = np.zeros(18)
    zs, quats, vns = [], [], []
    nan_at = -1
    for step in range(n_steps):
        q_err = q[7:] - HOME_JOINT_ANGLES_PIN
        v_err = v[6:]
        tau_act = np.clip(tau_hold - kp * q_err - kv * v_err, -tau_clip, tau_clip)
        tau = np.concatenate([np.zeros(6), tau_act])
        ddq = np.array(f(q, v, tau)).flatten()
        if not np.all(np.isfinite(ddq)):
            nan_at = step; break
        v = v + ddq * dt_sub
        q = pin.integrate(model, q, v * dt_sub)
        zs.append(q[2]); quats.append(q[6]); vns.append(float(np.linalg.norm(v)))
    return {"zs": np.array(zs), "quats": np.array(quats), "vns": np.array(vns),
            "nan_at": nan_at, "q": q, "v": v}


def main():
    fn_path = os.path.join(CUSADI_FUNCTION_DIR, "a1_compliant_sim.casadi")
    f_cas = casadi.Function.load(fn_path)
    rp = load_robot_description("a1_description", root_joint=pin.JointModelFreeFlyer())
    model = rp.model

    # Newton-solve equilibrium torques.
    v_zero = np.zeros(18)
    eq_z, tau_hold, r_norm = newton_equilibrium(f_cas, v_zero)
    print(f"[newton] eq_base_z={eq_z:.4f}  |r|={r_norm:.3f}  "
          f"tau_hold mag={np.abs(tau_hold).max():.2f} Nm")

    # ---------------------------------------------------------------
    # Phase 2.1a  Rollout-horizon open-loop stability.
    # ---------------------------------------------------------------
    print("\n=== Phase 2.1a: Open-loop rollout stability (T*dt=0.4s) ===")
    dt_ctrl = 0.02
    T = 20
    for dt_sub in [0.001, 0.002, 0.004, 0.005]:
        n_sub = max(1, int(round(dt_ctrl / dt_sub)))
        n_steps = T * n_sub
        out = cpu_rollout_open(f_cas, model, tau_hold, dt_sub, n_steps, initial_drop=0.0, init_z=eq_z)
        ok = (out["nan_at"] < 0
              and out["zs"].min() > 0.05
              and out["vns"].max() < 100.0)
        print(f"  dt_sub={dt_sub:.4f}  n_sub={n_sub:2d}  n_steps={n_steps:4d}  "
              f"z[min,max]=[{out['zs'].min():.3f},{out['zs'].max():.3f}]  "
              f"|v|_max={out['vns'].max():7.2f}  quatw[min]={out['quats'].min():+.3f}  "
              f"{'OK' if ok else 'BAD'}")

    # ---------------------------------------------------------------
    # Phase 2.1b  Closed-loop 5s stability (PD holding home).
    # ---------------------------------------------------------------
    print("\n=== Phase 2.1b: Closed-loop 5s with PD+gcomp ===")
    for dt_sub in [0.0005, 0.001]:
        n_steps = int(5.0 / dt_sub)
        out = cpu_rollout_closed(f_cas, model, tau_hold, dt_sub, n_steps,
                                  initial_drop=0.02, init_z=eq_z)
        ok = (out["nan_at"] < 0
              and out["zs"][-200:].min() > 0.2
              and out["quats"][-200:].min() > 0.9
              and out["vns"][-200:].mean() < 5.0)
        print(f"  dt_sub={dt_sub:.4f}  n_steps={n_steps:5d}  "
              f"final_z={out['q'][2]:.4f}  final_quatw={out['q'][6]:+.3f}  "
              f"final_|v|={np.linalg.norm(out['v']):.3f}  "
              f"{'OK' if ok else 'BAD'}")

    # ---------------------------------------------------------------
    # Phase 2.2  GPU batched open rollout matches CPU.
    # ---------------------------------------------------------------
    print("\n=== Phase 2.2: GPU batched open rollout matches CPU (dt_sub=0.002, 0.4s) ===")
    N = 128
    lp = CusadiLevelParallelFunction(f_cas, N, lib_name="level_parallel_kernel")
    dt_sub = 0.002
    n_steps = int(0.4 / dt_sub)

    q0 = torch.tensor(make_q(eq_z), device="cuda", dtype=torch.double)
    v0 = torch.zeros(18, device="cuda", dtype=torch.double)
    q_b = q0.unsqueeze(0).expand(N, -1).contiguous().clone()
    v_b = v0.unsqueeze(0).expand(N, -1).contiguous().clone()
    tau_b = torch.zeros(N, 18, device="cuda", dtype=torch.double)
    tau_b[:, 6:] = torch.tensor(tau_hold, device="cuda", dtype=torch.double)

    for step in range(n_steps):
        lp.evaluate([q_b, v_b, tau_b])
        ddq_b = lp.outputs_sparse[0]
        v_b = v_b + ddq_b * dt_sub
        q_b = batched_integrate(q_b, v_b, dt_sub)

    gpu_q_final = q_b[0].cpu().numpy()
    gpu_v_final = v_b[0].cpu().numpy()
    z_spread = q_b[:, 2].max().item() - q_b[:, 2].min().item()
    print(f"  All 128 rows identical: z spread={z_spread:.3e}")
    # Compare row 0 with CPU.
    cpu = cpu_rollout_open(f_cas, model, tau_hold, dt_sub, n_steps, initial_drop=0.0, init_z=eq_z)
    z_err = abs(gpu_q_final[2] - cpu["q"][2])
    v_err = np.linalg.norm(gpu_v_final - cpu["v"])
    ok_22 = z_spread < 1e-10 and z_err < 1e-6 and v_err < 1e-4
    print(f"  CPU final_z={cpu['q'][2]:.6f}  GPU final_z={gpu_q_final[2]:.6f}  "
          f"err_z={z_err:.3e}  err_v={v_err:.3e}")
    print(f"Phase 2.2 gate: {'PASS' if ok_22 else 'FAIL'}")

    # ---------------------------------------------------------------
    # Phase 2.3  LP eval timing at batch=128.
    # ---------------------------------------------------------------
    print("\n=== Phase 2.3: LP eval timing at batch=128 ===")
    # Warm.
    for _ in range(20):
        lp.evaluate([q_b, v_b, tau_b])
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lp.evaluate([q_b, v_b, tau_b])
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1e3)
    times_ms = np.array(times_ms)
    med = float(np.median(times_ms))
    print(f"  median={med:.3f} ms  p95={np.percentile(times_ms,95):.3f} ms  "
          f"mean={times_ms.mean():.3f} ms")

    for T_, nsub_ in [(20, 10), (20, 5), (10, 10), (10, 5), (20, 4)]:
        b = T_ * nsub_ * med
        print(f"  T={T_:2d}, n_sub={nsub_:2d}, dt_sub={dt_ctrl/nsub_:.4f}s → "
              f"{T_*nsub_:3d} evals = {b:.2f} ms / control step")

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
