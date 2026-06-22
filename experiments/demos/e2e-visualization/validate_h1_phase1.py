"""
Phase 1 validation for the H1 compliant-contact forward dynamics.

Structural parallel to validate_compliant_phase1.py for A1.
"""
from __future__ import annotations

import os
import sys

import casadi
import numpy as np
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cusadi import CUSADI_FUNCTION_DIR  # noqa: E402
from cusadi.pytorch.lp_function import CusadiLevelParallelFunction  # noqa: E402


# Home joint angles for H1 (19-dim, pinocchio order after floating base).
# Bent-knee standing stance (hip_pitch=-0.4, knee=+0.8, ankle=-0.4 keeps foot
# flat). Torso & arms at 0 — arms relaxed-down in the default URDF pose.
HOME_JOINT_ANGLES_PIN = np.zeros(19)
HOME_JOINT_ANGLES_PIN[2] = -0.4   # left_hip_pitch
HOME_JOINT_ANGLES_PIN[3] =  0.8   # left_knee
HOME_JOINT_ANGLES_PIN[4] = -0.4   # left_ankle
HOME_JOINT_ANGLES_PIN[7] = -0.4   # right_hip_pitch
HOME_JOINT_ANGLES_PIN[8] =  0.8   # right_knee
HOME_JOINT_ANGLES_PIN[9] = -0.4   # right_ankle

HOME_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])

# From FK scan: at base_z=1.0, ankle_link_z ≈ 0.089. Contact corner z_local
# = -0.08, so corner_world_z ≈ 0.009 at base_z=1.0. The just-touching base
# height is ≈ 0.991.
GROUND_TOUCH_BASE_Z = 0.991


def make_q(base_z: float) -> np.ndarray:
    return np.concatenate([[0.0, 0.0, base_z], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN])


def main():
    fn_path = os.path.join(CUSADI_FUNCTION_DIR, "h1_compliant_sim.casadi")
    f = casadi.Function.load(fn_path)
    print(f"Loaded {f.name()}: nnz_in={[f.nnz_in(i) for i in range(f.n_in())]}, "
          f"nnz_out={[f.nnz_out(i) for i in range(f.n_out())]}, n_instr={f.n_instructions()}")

    rp = load_robot_description("h1_description", root_joint=pin.JointModelFreeFlyer())
    model = rp.model
    v_zero = np.zeros(model.nv)
    tau_zero = np.zeros(model.nv)

    # --- 1.2.1 Freefall at +5cm above ground ---
    print("\n=== Phase 1.2: CPU physics ===")
    q_high = make_q(GROUND_TOUCH_BASE_Z + 0.05)
    ddq_high = np.array(f(q_high, v_zero, tau_zero)).flatten()
    print(f"[1.2.1] +5cm, tau=0:  ddq_z = {ddq_high[2]:+.4f} (expect ≈ -9.81)")
    ok1 = abs(ddq_high[2] + 9.81) < 0.1
    print(f"        ok (freefall): {ok1}")

    # --- 1.2.2 Monotone sweep across the ground ---
    sweep = []
    for dz in [+0.05, +0.02, +0.005, 0.0, -0.005, -0.01, -0.02]:
        q = make_q(GROUND_TOUCH_BASE_Z + dz)
        ddq = np.array(f(q, v_zero, tau_zero)).flatten()
        sweep.append((dz, float(ddq[2])))
    print(f"[1.2.2] sweep base_z offset → ddq_z (tau=0):")
    for dz, dz_acc in sweep:
        print(f"           dz={dz:+.3f} m  →  ddq_z = {dz_acc:+7.4f}")
    ddqs = np.array([s[1] for s in sweep])
    ok2 = np.all(np.diff(ddqs) > 0) and np.all(np.isfinite(ddqs))
    print(f"        ok (monotone + finite): {ok2}")

    # --- 1.2.3 Newton equilibrium (19 tau_act + base_z = 20 unknowns,
    #         25 residual components — under-determined on base roll/pitch).
    def _resid(state):
        q = np.concatenate([[0, 0, state[0]], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN])
        tau_full = np.concatenate([np.zeros(6), state[1:]])
        return np.array(f(q, v_zero, tau_full)).flatten()

    nu = 1 + 19  # base_z + 19 actuated torques
    state = np.concatenate([[0.98], np.zeros(19)])
    r0 = np.linalg.norm(_resid(state))
    for _ in range(60):
        r = _resid(state)
        if np.linalg.norm(r) < 1e-6:
            break
        J = np.empty((model.nv, nu))
        for k in range(nu):
            sp = state.copy(); sp[k] += 1e-5
            J[:, k] = (_resid(sp) - r) / 1e-5
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        state = state + 0.5 * step
    r_final = np.linalg.norm(_resid(state))
    print(f"[1.2.3] Newton equilibrium: |r0|={r0:7.2f} → |r|={r_final:7.4f}  "
          f"eq_base_z={state[0]:.4f}")
    ok3 = r_final < 10.0 and 0.90 < state[0] < 1.05
    print(f"        ok (residual small & equilibrium plausible): {ok3}")

    phase12_ok = ok1 and ok2 and ok3
    print(f"\nPhase 1.2 gate: {'PASS' if phase12_ok else 'FAIL'}")
    if not phase12_ok:
        sys.exit(1)

    # --- 1.3 GPU vs CPU at 10 random near-home configs ---
    print("\n=== Phase 1.3: LP GPU vs CPU (10 random configs) ===")
    import torch
    N = 16
    lp = CusadiLevelParallelFunction(f, N, lib_name="level_parallel_kernel")
    rng = np.random.default_rng(42)
    q_home = make_q(GROUND_TOUCH_BASE_Z)
    max_err = 0.0
    all_ok = True
    for i in range(10):
        q_test = q_home.copy()
        q_test[:3] += rng.normal(scale=0.02, size=3)
        theta = rng.normal(scale=0.1, size=3)
        half = theta * 0.5
        a = np.linalg.norm(half)
        if a < 1e-8:
            dq = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            dq = np.concatenate([np.sin(a) * half / a, [np.cos(a)]])
        x1, y1, z1, w1 = q_test[3:7]
        x2, y2, z2, w2 = dq
        q_test[3] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        q_test[4] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        q_test[5] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        q_test[6] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        q_test[3:7] /= np.linalg.norm(q_test[3:7])
        q_test[7:] += rng.normal(scale=0.15, size=19)

        v_test = rng.normal(scale=0.3, size=model.nv)
        tau_test = np.concatenate([np.zeros(6), rng.normal(scale=5.0, size=19)])

        cpu_ddq = np.array(f(q_test, v_test, tau_test)).flatten()

        q_b = torch.tensor(q_test, device="cuda", dtype=torch.double).unsqueeze(0).expand(N, -1).contiguous()
        v_b = torch.tensor(v_test, device="cuda", dtype=torch.double).unsqueeze(0).expand(N, -1).contiguous()
        tau_b = torch.tensor(tau_test, device="cuda", dtype=torch.double).unsqueeze(0).expand(N, -1).contiguous()
        lp.evaluate([q_b, v_b, tau_b])
        gpu_ddq = lp.outputs_sparse[0][0].cpu().numpy().flatten()

        err = np.max(np.abs(cpu_ddq - gpu_ddq))
        max_err = max(max_err, err)
        mark = "OK" if err < 1e-6 else "FAIL"
        if err >= 1e-6:
            all_ok = False
        print(f"  test {i:2d}: max_err = {err:.3e}  |cpu|={np.linalg.norm(cpu_ddq):7.4f}  [{mark}]")

    print(f"\n  Max error across 10 tests: {max_err:.3e}")
    print(f"Phase 1.3 gate: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        sys.exit(2)
    print("\nPhase 1 complete. Proceed to Phase 2.")


if __name__ == "__main__":
    main()
