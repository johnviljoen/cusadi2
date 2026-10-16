"""Compile panda_casadi.build_oscbf_function() with cusadi2 -> profiling/build/{dtype}/
(kernel .cu, oscbf_panda.plan, plan stats json).  python profiling/build_cusadi2_plan.py [--dtype double]"""
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import common
p = argparse.ArgumentParser(); p.add_argument("--dtype", default="double", choices=["double", "float"])
p.add_argument("--max-live", type=int, default=0, help="cusadi2 live-slot budget (shared memory per block); 0 = device maximum")
p.add_argument("--no-cse", action="store_true"); args = p.parse_args()
common.configure("cpu", x64=True)
import cusadi2
from panda_casadi import build_oscbf_function
max_live_arg = args.max_live
out = HERE / "build" / args.dtype; out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter(); fn, robot, info = build_oscbf_function(cse=not args.no_cse); t_build = time.perf_counter() - t0
cusadi2.write_kernel_source(str(out / "kernel.cu"), dtype=args.dtype)
max_live = max_live_arg or cusadi2.device_max_live(args.dtype)
plan_name = f"oscbf_panda_ml{max_live_arg}.plan" if max_live_arg else "oscbf_panda.plan"
t0 = time.perf_counter()
cusadi2.write_plan(fn, str(out / plan_name), n_stages=1, max_live=max_live)
t_plan = time.perf_counter() - t0
st = cusadi2.read_plan(str(out / plan_name))
summary = {k: (v if not hasattr(v, "__len__") or len(v) < 20 else f"<{len(v)} entries>") for k, v in st.items()}
summary.update(dict(dtype=args.dtype, sx_nodes=fn.n_nodes(), n_levels=len(st["level_widths"]) if "level_widths" in st else None,
                    max_level_width=int(max(st["level_widths"])) if "level_widths" in st else None,
                    max_live=max_live, cse=not args.no_cse, plan=plan_name, casadi_build_s=t_build, plan_write_s=t_plan, **info))
(out / plan_name.replace(".plan", "_stats.json")).write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
