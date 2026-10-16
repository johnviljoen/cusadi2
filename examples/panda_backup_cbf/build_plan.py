"""Compile backup_casadi.build_backup_function(N) with cusadi2.

    python build_plan.py --N 8 [--max-live 0] [--dtype double]

Writes build/<dtype>/kernel.cu (once per dtype), build/<dtype>/backup_N{N}_ml{max_live}.plan and
the matching _stats.json (levels, peak live slots, SX nodes, build times).
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling")); sys.path.insert(0, str(HERE.parent.parent))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--dtype", default="double", choices=["double", "float"])
ap.add_argument("--max-live", type=int, default=0, help="cusadi2 live-slot budget per block; 0 = device maximum")
ap.add_argument("--no-cse", action="store_true")
args = ap.parse_args()


def plan_name(N, max_live):
    return f"backup_N{N}_ml{max_live}.plan"


def main():
    common.configure("cpu", x64=True)
    import cusadi2
    from backup_casadi import build_backup_function
    out = HERE / "build" / args.dtype; out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); fn, robot, info = build_backup_function(args.N, cse=not args.no_cse); t_build = time.perf_counter() - t0
    if not (out / "kernel.cu").exists():
        cusadi2.write_kernel_source(str(out / "kernel.cu"), dtype=args.dtype)
    max_live = args.max_live or cusadi2.device_max_live(args.dtype)
    name = plan_name(args.N, args.max_live)
    t0 = time.perf_counter(); cusadi2.write_plan(fn, str(out / name), n_stages=1, max_live=max_live); t_plan = time.perf_counter() - t0
    st = cusadi2.read_plan(str(out / name))
    summary = {k: (v if not hasattr(v, "__len__") or len(v) < 20 else f"<{len(v)} entries>") for k, v in st.items()}
    summary.update(dict(dtype=args.dtype, n_levels=len(st["level_widths"]), max_level_width=int(max(st["level_widths"])),
                        median_level_width=float(__import__("numpy").median(st["level_widths"])),
                        max_live=max_live, max_live_arg=args.max_live, cse=not args.no_cse, plan=name,
                        casadi_build_s=t_build, plan_write_s=t_plan, **info))
    (out / name.replace(".plan", "_stats.json")).write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("N", "n_rows", "sx_nodes", "n_levels", "max_level_width", "median_level_width", "peak_live_slots", "max_live", "casadi_build_s", "plan_write_s") if k in summary}, indent=1))


if __name__ == "__main__":
    main()
