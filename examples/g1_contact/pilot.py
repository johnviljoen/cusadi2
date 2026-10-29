"""Row-cap pilot: MuJoCo C steps the first `n` instances of each scenario and records the largest nefc and ncon
seen at any step; both arms' caps (our G1_NJMAX, warp's njmax / nconmax) are set from these with a margin.

    python pilot.py [n=512]      -> build/pilot.json
"""
import json
import os
import sys

import mujoco
import numpy as np

import g1

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
STEPS = {"standing": 200, "drop": 300}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    m = mujoco.MjModel.from_xml_path(g1.XML)
    d = mujoco.MjData(m)
    out = {}
    for scen, n_steps in STEPS.items():
        q = np.fromfile(os.path.join(BUILD, "inputs", f"{scen}_qpos.f64")).reshape(-1, m.nq)[:n]
        v = np.fromfile(os.path.join(BUILD, "inputs", f"{scen}_qvel.f64")).reshape(-1, m.nv)[:n]
        u = np.fromfile(os.path.join(BUILD, "inputs", f"{scen}_ctrl.f64")).reshape(-1, m.nu)[:n]
        max_nefc = max_ncon = 0; sum_nefc = sum_ncon = 0; sum_niter = 0; fell = 0
        for b in range(n):
            mujoco.mj_resetData(m, d); d.qpos[:], d.qvel[:], d.ctrl[:] = q[b], v[b], u[b]
            for _ in range(n_steps):
                mujoco.mj_step(m, d)
                max_nefc = max(max_nefc, int(d.nefc)); max_ncon = max(max_ncon, int(d.ncon))
                sum_nefc += int(d.nefc); sum_ncon += int(d.ncon); sum_niter += int(d.solver_niter[0])
            fell += int(d.qpos[2] < 0.5)
        out[scen] = dict(instances=n, steps=n_steps, max_nefc=max_nefc, max_ncon=max_ncon, mean_nefc=sum_nefc / (n * n_steps),
                         mean_ncon=sum_ncon / (n * n_steps), mean_niter=sum_niter / (n * n_steps), fell=fell)
        print(scen, out[scen], flush=True)
    json.dump(out, open(os.path.join(BUILD, "pilot.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
