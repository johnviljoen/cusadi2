"""Smooth-step gate: the CasADi `smooth` and `integrate` functions against MuJoCo C with contacts and
constraints disabled (the same flags the warp / MJX baselines get in the Menagerie campaign).

    python gate_smooth.py [n_states]

Checks at random states around both keyframes (joint angles +-0.5 rad, random base orientation and
velocities): M against mj_fullM, qfrc_smooth, qacc_smooth, geom placements, and the geom Jacobians
assembled from `cdof` over the body's dof ancestry (as the contact kernel does) against mj_jacGeom, one implicitfast step against mj_step; then a 500-step trajectory from each keyframe
(falls through the floor: no contact) compared step by step. Prints the max errors; exits 1 on failure.
"""
import sys
import time

import mujoco
import numpy as np

import g1

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TOL = dict(M=1e-9, qfrc_smooth=1e-8, qacc_smooth=1e-7, geom_pos=1e-10, geom_rot=1e-10, geom_jac=1e-9,
           step_qpos=1e-10, step_qvel=1e-8, traj_qpos=1e-7, traj_qvel=1e-5)


def main():
    cm = g1.load()
    t0 = time.time()
    smooth = cm.build_smooth(geoms=True)
    integ = cm.build_integrate()
    print(f"smooth: {smooth.n_instructions()} instructions, integrate: {integ.n_instructions()} ({time.time() - t0:.1f} s)")
    m = g1.disable_contacts(mujoco.MjModel.from_xml_path(g1.XML))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    err = {k: 0.0 for k in TOL}
    ng = cm.n_geom
    gids = [g["geom"] for g in cm.geoms]
    chains = [cm.body_dofs(g["body"]) for g in cm.geoms]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for s in range(N):
        mujoco.mj_resetDataKeyframe(m, d, s % m.nkey)
        d.qpos[7:] += rng.uniform(-0.5, 0.5, m.nq - 7)
        d.qpos[3:7] = rng.normal(size=4)
        d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
        d.qvel[:] = rng.normal(0.0, 1.0, m.nv)
        d.ctrl[:] = m.key_ctrl[s % m.nkey] + rng.uniform(-0.3, 0.3, m.nu)
        q0, v0, u0 = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
        mujoco.mj_forward(m, d)
        M, qfs, gp, gr, cdof = [np.asarray(x) for x in smooth(q0, v0, u0)]
        qfs = qfs.ravel()
        Mref = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, Mref)
        err["M"] = max(err["M"], np.abs(M - Mref).max())
        err["qfrc_smooth"] = max(err["qfrc_smooth"], np.abs(qfs - d.qfrc_smooth).max())
        err["qacc_smooth"] = max(err["qacc_smooth"], np.abs(np.linalg.solve(M, qfs) - d.qacc_smooth).max())
        for k, gid in enumerate(gids):
            err["geom_pos"] = max(err["geom_pos"], np.abs(gp[:, k] - d.geom_xpos[gid]).max())
            err["geom_rot"] = max(err["geom_rot"], np.abs(gr[:, k].reshape(3, 3, order="F") - d.geom_xmat[gid].reshape(3, 3)).max())
            mujoco.mj_jacGeom(m, d, jacp, jacr, gid)
            J = np.zeros((6, m.nv))
            for dof in chains[k]:
                J[3:, dof] = cdof[3:, dof]
                J[:3, dof] = cdof[:3, dof] + np.cross(cdof[3:, dof], gp[:, k] - q0[:3])
            err["geom_jac"] = max(err["geom_jac"], np.abs(J[:3] - jacp).max(), np.abs(J[3:] - jacr).max())
        mujoco.mj_step(m, d)
        q1, v1, _ = [np.asarray(x).ravel() for x in integ(q0, v0, M, qfs, np.zeros(m.nv))]
        err["step_qpos"] = max(err["step_qpos"], np.abs(q1 - d.qpos).max())
        err["step_qvel"] = max(err["step_qvel"], np.abs(v1 - d.qvel).max())
    # trajectories
    for key in range(m.nkey):
        mujoco.mj_resetDataKeyframe(m, d, key)
        d.ctrl[:] = m.key_ctrl[key]
        q, v, u = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()
        for t in range(500):
            mujoco.mj_step(m, d)
            M, qfs = [np.asarray(x) for x in smooth(q, v, u)[:2]]
            q, v, _ = [np.asarray(x).ravel() for x in integ(q, v, M, qfs, np.zeros(m.nv))]
            err["traj_qpos"] = max(err["traj_qpos"], np.abs(q - d.qpos).max())
            err["traj_qvel"] = max(err["traj_qvel"], np.abs(v - d.qvel).max())
        print(f"trajectory {m.key(key).name}: z after 500 steps {d.qpos[2]:.3f} (C) {q[2]:.3f} (cusadi2)")
    ok = True
    for k, v in err.items():
        flag = "ok" if v <= TOL[k] else "FAIL"
        ok &= v <= TOL[k]
        print(f"{k:12s} max |err| {v:.3e}  (tol {TOL[k]:.0e}) {flag}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
