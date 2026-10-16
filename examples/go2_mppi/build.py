"""Build the Go2 MPPI example: the rollout step as cusadi2 plans, the kernel source, and the drivers.

    python build.py [build_dir]            (default: ./build)

Writes plans/go2_step<n>.plan for n = 1 and SUBSTEPS fused physics steps, fns/ (the CasADi functions,
for the CPU baseline and checks), kernel_{float,double}.cu, meta.json, and compiles
mppi_rollout_{f32,f64} and mppi_bench_{f32,f64} with nvcc (NVCC overrides the compiler).
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import cusadi2
import go2_model as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "build")
DT = 0.004            # physics step
SUBSTEPS = 5          # per control step: 50 Hz control
HORIZON = int(os.environ.get("GO2_HORIZON", 20))   # control steps in a rollout (0.4 s)
KNOT_DT = HORIZON * SUBSTEPS * DT / (G.N_KNOT - 1)
NVCC = os.environ.get("NVCC") or shutil.which("nvcc") or os.path.join(sys.prefix, "bin", "nvcc")


def main():
    os.makedirs(os.path.join(OUT, "plans"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "fns"), exist_ok=True)
    meta = {"dt": DT, "substeps": SUBSTEPS, "horizon": HORIZON, "knot_dt": KNOT_DT, "n_knot": G.N_KNOT,
            "nx": G.NX, "params": [n for n, _ in G.PARAMS], "plans": {}}
    for n_sub in (1, SUBSTEPS):
        t0 = time.time()
        f = G.make(DT, KNOT_DT, n_sub, name=f"go2_step{n_sub}")
        f.save(os.path.join(OUT, "fns", f"{f.name()}.casadi"))
        for dtype in ("float", "double"):
            path = os.path.join(OUT, "plans", f"{f.name()}_{dtype}.plan")
            cusadi2.write_plan(f, path, max_live=cusadi2.device_max_live(dtype))
            p = cusadi2.read_plan(path)
            meta["plans"][f"{f.name()}_{dtype}"] = {"instructions": f.n_instructions(), "levels": len(p["level_widths"]),
                                                    "peak_live_slots": p["peak_live_slots"]}
        print(f"{f.name()}: {f.n_instructions()} instructions, {meta['plans'][f.name() + '_float']['levels']} levels, "
              f"{meta['plans'][f.name() + '_float']['peak_live_slots']} live slots ({time.time() - t0:.1f} s)")
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    with open(os.path.join(OUT, "meta.txt"), "w") as mf:                 # the same scalars, for the C++ drivers
        for k, v in (("dt", DT), ("substeps", SUBSTEPS), ("horizon", HORIZON), ("knot_dt", KNOT_DT), ("n_knot", G.N_KNOT),
                     ("nx", G.NX), ("npar", len(G.PARAMS))):
            mf.write(f"{k} {v!r}\n")
    _, pm, _, _ = G.load(DT)                                               # knot limits: joint range relative to HOME
    np.stack([pm.lowerPositionLimit[7:] - G.HOME, pm.upperPositionLimit[7:] - G.HOME]).tofile(os.path.join(OUT, "ylim.f64"))
    os.makedirs(os.path.join(OUT, "bench"), exist_ok=True)                  # benchmark inputs: standing on flat ground
    G.initial_state(25.0, 25.0, 0.0).tofile(os.path.join(OUT, "bench", "x0.f64"))
    G.PAR_DEFAULT.tofile(os.path.join(OUT, "bench", "par.f64"))
    arch = "sm_" + subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).decode().split()[0].replace(".", "")
    for dtype, tag in (("float", "f32"), ("double", "f64")):
        kern = os.path.join(OUT, f"kernel_{dtype}.cu")
        cusadi2.write_kernel_source(kern, dtype=dtype)
        for drv in ("mppi_rollout", "mppi_bench"):
            src = os.path.join(HERE, drv + ".cu")
            if not os.path.exists(src):
                continue
            t0 = time.time()
            subprocess.run([NVCC, "-std=c++17", "-O3", f"-arch={arch}", f"-DCUSADI2_SCALAR={dtype}", f"-I{cusadi2.INCLUDE_DIR}",
                            src, kern, "-o", os.path.join(OUT, f"{drv}_{tag}")], check=True)
            print(f"nvcc {drv}_{tag}: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
