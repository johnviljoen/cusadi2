"""Build the G1 contact example: the cusadi2 plans, kernel sources, benchmark inputs and the step library.

    python build.py [build_dir]            (default: ./build)

Writes plans/{smooth,smooth_nogeom,integrate}_{float,double}.plan, fns/*.casadi, kernel_{float,double}.cu,
g1_model.h (the model constants for cusadi2_contact.h), meta.json, inputs/ (the per-instance initial states and controls of both benchmark scenarios, shared by
the cusadi2 driver and the warp harness) and compiles libg1step_{f32,f64}.so with nvcc.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import mujoco
import numpy as np

import cusadi2
import g1

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "build")
NVCC = os.environ.get("NVCC") or shutil.which("nvcc") or os.path.join(sys.prefix, "bin", "nvcc")
NJMAX = int(os.environ.get("G1_NJMAX", 128))     # constraint-row cap per instance (mujoco-warp's njmax): pilot.py max nefc 113 (drop) with margin
B_MAX = 16384
SEED = 0


def scenario_inputs(m, rng):
    """Initial (qpos, qvel, ctrl) per instance for the two scenarios, B_MAX each:
    standing: the knees_bent keyframe, joints perturbed by +-0.05 rad, ctrl = keyframe ctrl + the same
              perturbation (so the PD targets match the perturbed pose), zero velocity;
    drop:     1 m above the keyframe height, a random base orientation, joints uniform within +-0.5 rad,
              random base velocity (0.5 m/s, 1 rad/s), ctrl = keyframe ctrl."""
    key = [i for i in range(m.nkey) if m.key(i).name == "knees_bent"][0]
    q0, u0 = np.asarray(m.key_qpos[key]), np.asarray(m.key_ctrl[key])
    out = {}
    dq = rng.uniform(-0.05, 0.05, (B_MAX, m.nq - 7))
    q = np.tile(q0, (B_MAX, 1)); q[:, 7:] += dq
    out["standing"] = (q, np.zeros((B_MAX, m.nv)), np.tile(u0, (B_MAX, 1)) + dq)
    q = np.tile(q0, (B_MAX, 1))
    q[:, 2] += 1.0
    quat = rng.normal(size=(B_MAX, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    q[:, 3:7] = quat
    q[:, 7:] += rng.uniform(-0.5, 0.5, (B_MAX, m.nq - 7))
    v = np.zeros((B_MAX, m.nv)); v[:, :3] = rng.normal(0, 0.5, (B_MAX, 3)); v[:, 3:6] = rng.normal(0, 1.0, (B_MAX, 3))
    out["drop"] = (q, v, np.tile(u0, (B_MAX, 1)))
    return out


def main():
    for sub in ("plans", "fns", "inputs"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    cm = g1.load()
    meta = {"xml": g1.XML, "nq": cm.nq, "nv": cm.nv, "nu": cm.nu, "n_geom": cm.n_geom, "dt": cm.dt, "opt": cm.opt,
            "b_max": B_MAX, "seed": SEED, "njmax": NJMAX, "plans": {}}
    fns = [cm.build_smooth(geoms=True, name="smooth"), cm.build_smooth(geoms=False, name="smooth_nogeom"), cm.build_integrate()]
    cm.write_model_header(os.path.join(OUT, "g1_model.h"))
    for f in fns:
        t0 = time.time()
        f.save(os.path.join(OUT, "fns", f"{f.name()}.casadi"))
        for dtype in ("float", "double"):
            path = os.path.join(OUT, "plans", f"{f.name()}_{dtype}.plan")
            cusadi2.write_plan(f, path, max_live=cusadi2.device_max_live(dtype))
            p = cusadi2.read_plan(path)
            meta["plans"][f"{f.name()}_{dtype}"] = {"instructions": f.n_instructions(), "levels": len(p["level_widths"]),
                                                    "peak_live_slots": p["peak_live_slots"]}
        pf = meta["plans"][f.name() + "_float"]
        print(f"{f.name()}: {f.n_instructions()} instructions, {pf['levels']} levels, {pf['peak_live_slots']} live slots "
              f"({time.time() - t0:.1f} s)", flush=True)
    rng = np.random.default_rng(SEED)
    for name, (q, v, u) in scenario_inputs(cm.m, rng).items():
        for arr, tag in ((q, "qpos"), (v, "qvel"), (u, "ctrl")):
            np.ascontiguousarray(arr, dtype=np.float64).tofile(os.path.join(OUT, "inputs", f"{name}_{tag}.f64"))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    arch = "sm_" + subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).decode().split()[0].replace(".", "")
    for dtype, tag in (("float", "f32"), ("double", "f64")):
        kern = os.path.join(OUT, f"kernel_{dtype}.cu")
        cusadi2.write_kernel_source(kern, dtype=dtype)
        t0 = time.time()
        subprocess.run([NVCC, "-std=c++17", "-O3", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", f"-DCUSADI2_SCALAR={dtype}",
                        f"-DG1_NJMAX={NJMAX}", f"-I{cusadi2.INCLUDE_DIR}", f"-I{OUT}", os.path.join(HERE, "g1_step.cu"), kern,
                        "-o", os.path.join(OUT, f"libg1step_{tag}.so")], check=True)
        print(f"nvcc libg1step_{tag}.so: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
