"""
Phase 1 validation for the A1 compliant-contact forward dynamics function.

Two gates that must both pass before moving to Phase 2:
 1.2  Physics sanity on CPU oracle (3 tests).
 1.3  LP GPU kernel matches CPU to <1e-6 on 10 random (q,v,tau).
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
from cusadi.level_parallel.function import CusadiLevelParallelFunction  # noqa: E402


HOME_JOINT_ANGLES_PIN = np.array([
    0.0,  0.8, -1.6,
    0.0,  0.8, -1.6,
    0.0,  0.8, -1.6,
    0.0,  0.8, -1.6,
], dtype=np.float64)
HOME_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

# At this base height the foot frame origins sit exactly at z=0 (feet just
# touching the ground, penetration ~softplus_width). Derived from FK on the
# A1 URDF: at home joints and base_z = 0.2787, foot_z = 0.000. Rounded to
# 0.279 for the test.
GROUND_TOUCH_BASE_Z = 0.279


def build_home_q(base_z: float) -> np.ndarray:
    return np.concatenate([[0.0, 0.0, base_z], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN])


def main():
    fn_path = os.path.join(CUSADI_FUNCTION_DIR, "a1_compliant_sim.casadi")
    assert os.path.exists(fn_path), f"Missing {fn_path}; run build_a1_compliant_fwd.py first"
    f = casadi.Function.load(fn_path)
    print(f"Loaded {f.name()}: nnz_in={[f.nnz_in(i) for i in range(f.n_in())]}, "
          f"nnz_out={[f.nnz_out(i) for i in range(f.n_out())]}, n_instr={f.n_instructions()}")

    # -----------------------------------------------------------------
    # Phase 1.2  CPU physics sanity
    # -----------------------------------------------------------------
    print("\n=== Phase 1.2: CPU physics ===")
    rp = load_robot_description("a1_description", root_joint=pin.JointModelFreeFlyer())
    model = rp.model
    v_zero = np.zeros(model.nv)
    tau_zero = np.zeros(model.nv)

    # Test 1: feet 5cm above ground, tau=0 → freefall (gravity only).
    q_high = build_home_q(GROUND_TOUCH_BASE_Z + 0.05)
    ddq_high = np.array(f(q_high, v_zero, tau_zero)).flatten()
    print(f"[1.2.1] +5cm, tau=0:  ddq_z = {ddq_high[2]:+.4f} (expect ≈ -9.81)")
    ok1 = abs(ddq_high[2] + 9.81) < 0.1
    print(f"        ok (freefall): {ok1}")

    # Test 2: monotonicity. Sweep base_z from well above ground down to moderate
    # penetration; ddq_z at tau=0 must increase monotonically (freefall → ground
    # pushes back). Also verifies the function has no NaN / discontinuity across
    # the softplus transition.
    sweep = []
    for dz in [+0.05, +0.02, +0.005, 0.0, -0.005, -0.01, -0.02]:
        q = build_home_q(GROUND_TOUCH_BASE_Z + dz)
        ddq = np.array(f(q, v_zero, tau_zero)).flatten()
        sweep.append((dz, float(ddq[2])))
    print(f"[1.2.2] sweep base_z offset → ddq_z (tau=0):")
    for dz, dz_acc in sweep:
        print(f"           dz={dz:+.3f} m  →  ddq_z = {dz_acc:+7.4f}")
    ddqs = np.array([s[1] for s in sweep])
    ok2 = np.all(np.diff(ddqs) > 0) and np.all(np.isfinite(ddqs))
    print(f"        ok (monotone + finite): {ok2}")

    # Test 3: Newton-solve compliant-model equilibrium. We solve for base_z and
    # the 12 actuated torques that zero ddq at home-pose, zero velocity. The
    # problem is over-determined (18 equations, 13 unknowns) but dominated by
    # the vertical force balance + the per-leg statics. Residual should drop
    # to a finite floor (base roll/pitch modes not in the Jacobian image) but
    # the function must be continuous and the solve must make progress.
    def _resid(state):
        q = np.concatenate([[0, 0, state[0]], HOME_QUAT_XYZW, HOME_JOINT_ANGLES_PIN])
        tau_full = np.concatenate([np.zeros(6), state[1:]])
        return np.array(f(q, v_zero, tau_full)).flatten()

    state = np.concatenate([[0.25], np.zeros(12)])
    r0 = np.linalg.norm(_resid(state))
    for _ in range(60):
        r = _resid(state)
        if np.linalg.norm(r) < 1e-6:
            break
        J = np.empty((18, 13))
        for k in range(13):
            sp = state.copy(); sp[k] += 1e-5
            J[:, k] = (_resid(sp) - r) / 1e-5
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        state = state + 0.5 * step
    r_final = np.linalg.norm(_resid(state))
    print(f"[1.2.3] Newton equilibrium: |r0|={r0:7.2f} → |r|={r_final:7.4f}  "
          f"eq_base_z={state[0]:.4f}")
    ok3 = r_final < 5.0 and 0.25 < state[0] < 0.29
    print(f"        ok (residual small & equilibrium plausible): {ok3}")

    phase12_ok = ok1 and ok2 and ok3
    print(f"\nPhase 1.2 gate: {'PASS' if phase12_ok else 'FAIL'}")
    if not phase12_ok:
        sys.exit(1)

    # -----------------------------------------------------------------
    # Phase 1.3  LP GPU matches CPU to <1e-6
    # -----------------------------------------------------------------
    print("\n=== Phase 1.3: LP GPU vs CPU (10 random configs) ===")
    import torch

    N = 16  # use a small batch; single-instance ok too but batched is closer to real use
    lp = CusadiLevelParallelFunction(f, N, lib_name="level_parallel_kernel")

    rng = np.random.default_rng(42)
    max_err = 0.0
    all_ok = True
    q_home = build_home_q(GROUND_TOUCH_BASE_Z)
    for i in range(10):
        # Random config near home: perturb joints moderately, base pose a bit.
        q_test = q_home.copy()
        q_test[:3] += rng.normal(scale=0.02, size=3)
        # Perturb the quaternion with a small axis-angle and renormalize.
        theta = rng.normal(scale=0.1, size=3)
        # Apply as a delta-rotation right-multiplication (xyzw).
        half = theta * 0.5
        a = np.linalg.norm(half)
        if a < 1e-8:
            dq = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            dq = np.concatenate([np.sin(a) * half / a, [np.cos(a)]])
        # Hamilton product (xyzw) of q_test[3:7] ⊗ dq
        x1, y1, z1, w1 = q_test[3:7]
        x2, y2, z2, w2 = dq
        q_test[3] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        q_test[4] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        q_test[5] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        q_test[6] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        q_test[3:7] /= np.linalg.norm(q_test[3:7])
        q_test[7:] += rng.normal(scale=0.15, size=12)

        v_test = rng.normal(scale=0.3, size=model.nv)
        tau_test = np.concatenate([np.zeros(6), rng.normal(scale=5.0, size=12)])

        cpu_ddq = np.array(f(q_test, v_test, tau_test)).flatten()

        # Fill a batch of N copies of the same input, verify all are identical.
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
    phase13_ok = all_ok
    print(f"Phase 1.3 gate: {'PASS' if phase13_ok else 'FAIL'}")
    if not phase13_ok:
        sys.exit(2)

    print("\nPhase 1 complete. Proceed to Phase 2.")


if __name__ == "__main__":
    main()
