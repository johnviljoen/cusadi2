"""Contact gate: the fused contact step (libg1step, variant 1) against MuJoCo C with contacts on, with
mujoco-warp through the same comparisons.

    python gate_contact.py [--dtype f64|f32] [--states 200]

1. Per-state: `states` random states (half around the knees_bent keyframe at ground level, half dropped
   and rotated so that hands, knees, torso and head penetrate the floor), zero warm start everywhere, one
   forward each. Checked: the row counts (nefc, ncon) match C exactly; on the states where both C and we
   converge within the model's 5 Newton iterations, qacc and qfrc_constraint match C to the tolerance;
   over all states the residual against C is reported for us and for warp (the model's iteration cap is
   reached on most contact-rich states, where C's and warp's Newton paths differ: see the README), and
   ours against warp's.
2. Trajectories: the first 8 instances of build/inputs for both scenarios, 200 (standing) and 300 (drop)
   chained steps, final qpos / qvel of ours and warp against mj_step, and ours against warp.
Exits 1 if a per-state check fails.
"""
import argparse
import os
import sys

import mujoco
import numpy as np

import g1
from bench import BUILD, CusadiStep

TOL = {"f64": dict(qacc=1e-6, qfc=1e-6), "f32": dict(qacc=1e-3, qfc=1e-3)}      # converged states only


def sample_states(m, rng, n):
    key = [i for i in range(m.nkey) if m.key(i).name == "knees_bent"][0]
    q0, u0 = np.asarray(m.key_qpos[key]), np.asarray(m.key_ctrl[key])
    Q, V, U = np.zeros((n, m.nq)), np.zeros((n, m.nv)), np.zeros((n, m.nu))
    for i in range(n):
        q = q0.copy()
        if i % 2 == 0:                       # on the feet, slightly sunk, small perturbations
            q[7:] += rng.uniform(-0.1, 0.1, m.nq - 7)
            q[2] += rng.uniform(-0.02, 0.01)
            v = rng.normal(0, 0.3, m.nv)
        else:                                # lying / falling: random orientation, low, penetrating somewhere
            quat = rng.normal(size=4); q[3:7] = quat / np.linalg.norm(quat)
            q[2] = rng.uniform(0.1, 0.5)
            q[7:] += rng.uniform(-0.5, 0.5, m.nq - 7)
            v = rng.normal(0, 1.0, m.nv)
        Q[i], V[i], U[i] = q, v, u0 + rng.uniform(-0.2, 0.2, m.nu)
    return Q, V, U


def c_forward(m, q, v, u):
    d = mujoco.MjData(m)
    out = []
    for i in range(q.shape[0]):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q[i], v[i], u[i]
        d.qacc_warmstart[:] = 0
        mujoco.mj_forward(m, d)
        out.append((d.qacc.copy(), d.qfrc_constraint.copy(), int(d.nefc), int(d.ncon), int(d.solver_niter[0])))
    return out


def c_rollout(m, q, v, u, n_steps):
    d = mujoco.MjData(m)
    Q, V = np.zeros_like(q), np.zeros_like(v)
    for i in range(q.shape[0]):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], d.ctrl[:] = q[i], v[i], u[i]
        for _ in range(n_steps):
            mujoco.mj_step(m, d)
        Q[i], V[i] = d.qpos, d.qvel
    return Q, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="f64")
    ap.add_argument("--states", type=int, default=200)
    a = ap.parse_args()
    tol = TOL[a.dtype]
    m = mujoco.MjModel.from_xml_path(g1.XML)
    rng = np.random.default_rng(1)
    ok = True
    # ---- 1. per-state
    Q, V, U = sample_states(m, rng, a.states)
    ref = c_forward(m, Q, V, U)
    step = CusadiStep(a.dtype, 1, a.states, 1, Q, V, U)
    qacc = np.zeros((a.states, m.nv)); qfc = np.zeros((a.states, m.nv)); stats = np.zeros((a.states, 4), dtype=np.int32)
    import ctypes
    step.lib.g1_solver.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    step.lib.g1_solver(step.h, qacc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), qfc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    step.close()
    import warp as wp
    import mujoco_warp as mjw
    wp.init()
    mw = mjw.put_model(m)
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=a.states, njmax=192, nconmax=48)
    dw.qpos = wp.array(Q.astype(np.float32)); dw.qvel = wp.array(V.astype(np.float32)); dw.ctrl = wp.array(U.astype(np.float32))
    dw.qacc_warmstart.zero_()
    mjw.forward(mw, dw); wp.synchronize()
    qacc_w, nefc_w, niter_w = dw.qacc.numpy().astype(np.float64), dw.nefc.numpy(), dw.solver_niter.numpy()
    rel = lambda x, y: np.abs(x - y).max() / max(1.0, np.abs(y).max())
    e_conv = e_conv_f = 0.0; n_conv = 0; e_all = []; e_warp = []; e_ours_warp = []; n_rows_bad = 0; n_rows_bad_w = 0
    overflow = [i for i in range(a.states) if stats[i, 3] & 1]       # rows beyond the built cap: dropped, excluded below
    for i, (qa, qf, nefc, ncon, niter) in enumerate(ref):
        if i in overflow:
            continue
        n_rows_bad += int(stats[i, 0] != nefc or stats[i, 1] != ncon)
        n_rows_bad_w += int(nefc_w[i] != nefc)
        e_all.append(rel(qacc[i], qa)); e_warp.append(rel(qacc_w[i], qa)); e_ours_warp.append(rel(qacc[i], qacc_w[i]))
        if niter < m.opt.iterations and stats[i, 2] < m.opt.iterations:
            n_conv += 1
            e_conv = max(e_conv, rel(qacc[i], qa)); e_conv_f = max(e_conv_f, rel(qfc[i], qf))
    e_all, e_warp, e_ours_warp = np.array(e_all), np.array(e_warp), np.array(e_ours_warp)
    print(f"per-state ({a.states}, {a.dtype}): row counts (nefc, ncon) vs C: {n_rows_bad} mismatches (warp nefc: {n_rows_bad_w}); "
          f"C nefc range {min(r[2] for r in ref)}-{max(r[2] for r in ref)}; {len(overflow)} instances over the built row cap excluded "
          f"(C nefc {[ref[i][2] for i in overflow]})")
    print(f"  converged states ({n_conv} of {a.states}: C and ours < {m.opt.iterations} iterations): max rel |qacc err| {e_conv:.3e}, "
          f"max rel |qfrc_constraint err| {e_conv_f:.3e}  (tol {tol['qacc']:.0e}) {'ok' if e_conv <= tol['qacc'] and e_conv_f <= tol['qfc'] else 'FAIL'}")
    print(f"  all states, rel |qacc err| vs C: ours median {np.median(e_all):.2e} max {e_all.max():.2e}, warp median {np.median(e_warp):.2e} max {e_warp.max():.2e}; "
          f"ours vs warp median {np.median(e_ours_warp):.2e} max {e_ours_warp.max():.2e}; "
          f"iterations: C mean {np.mean([r[4] for r in ref]):.2f}, ours {stats[:, 2].mean():.2f}, warp {niter_w.mean():.2f}")
    ok &= e_conv <= tol["qacc"] and e_conv_f <= tol["qfc"] and n_rows_bad == 0
    # ---- 2. trajectories
    meta_nq, meta_nv, meta_nu = m.nq, m.nv, m.nu
    for scen, n_steps in (("standing", 200), ("drop", 300)):
        q = np.fromfile(os.path.join(BUILD, "inputs", f"{scen}_qpos.f64")).reshape(-1, meta_nq)[:8]
        v = np.fromfile(os.path.join(BUILD, "inputs", f"{scen}_qvel.f64")).reshape(-1, meta_nv)[:8]
        u = np.fromfile(os.path.join(BUILD, "inputs", f"{scen}_ctrl.f64")).reshape(-1, meta_nu)[:8]
        Qc, Vc = c_rollout(m, q, v, u, n_steps)
        step = CusadiStep(a.dtype, 1, 8, n_steps, q, v, u)
        Qg, Vg = step.state()
        step.close()
        from bench import WarpStep
        w = WarpStep(m, 8, n_steps, q, v, u)
        Qw, Vw = w.state()
        w.close()
        print(f"trajectory {scen} ({n_steps} steps): ours ({a.dtype}) vs C max |qpos err| {np.abs(Qg - Qc).max():.3e} |qvel err| {np.abs(Vg - Vc).max():.3e}; "
              f"warp vs C {np.abs(Qw - Qc).max():.3e} / {np.abs(Vw - Vc).max():.3e}; ours vs warp {np.abs(Qg - Qw).max():.3e} / {np.abs(Vg - Vw).max():.3e}")
        print(f"   z after {n_steps}: C {np.round(Qc[:, 2], 3).tolist()} ours {np.round(Qg[:, 2], 3).tolist()} warp {np.round(Qw[:, 2], 3).tolist()}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
