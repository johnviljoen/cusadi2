"""Plans for the cusadi baseline (mpc_mono_bench.cu): the wip's six whole-horizon functions as written by
vendor_setup.py, batch-only (n_stages = 1), plus p_eval.f64 and dx_ref.f64.

usage: python build_mpc_mono.py <out_dir> [double,float]
"""
import os, sys, time, numpy as np, casadi as ca, cusadi2
REF = os.path.join(os.environ.get("CUSADI_WIP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")), "humanoid_mpc_fns")
OUT = sys.argv[1]; DTYPES = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["double", "float"]); os.makedirs(OUT, exist_ok=True)
for tag in ["KKT_matrix", "ruiz_scaling", "ldl_factorize", "KKT_vector", "ldl_solve", "ADMM_step"]:
    f = ca.Function.load(os.path.join(REF, f"{tag}_mit_humanoid_mpc.casadi")); t0 = time.perf_counter()
    cusadi2.write_plan(f, os.path.join(OUT, f.name() + ".plan"))
    p = cusadi2.read_plan(os.path.join(OUT, f.name() + ".plan"))
    print(f"plan {f.name():34s} instr={f.n_instructions():8d} sz_w={f.sz_w():6d} levels={len(p['level_widths']):6d} "
          f"peak_live={p['peak_live_slots']:6d} ({time.perf_counter() - t0:.1f}s)", flush=True)
for dtype in DTYPES: cusadi2.write_kernel_source(os.path.join(OUT, f"kernel_{dtype}.cu"), dtype=dtype)
np.load(os.path.join(REF, "p_eval.npy")).reshape(-1).astype(np.float64).tofile(os.path.join(OUT, "p_eval.f64"))
np.load(os.path.join(REF, "dx_custom.npy")).reshape(-1).astype(np.float64).tofile(os.path.join(OUT, "dx_ref.f64"))
print("BUILD MONO DONE")
