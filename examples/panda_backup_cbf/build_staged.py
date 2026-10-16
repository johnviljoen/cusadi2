"""Compile the staged backup-CBF model (head + step) with cusadi2 and build the FFI library.

    python build_staged.py [--max-live-head 0] [--max-live-step 0] [--force]

Writes build/double/{backup_head_ml*.plan, backup_step_ml*.plan, *_stats.json, kernel.cu,
libbackup_ffi.so}.  The library is N-agnostic (N is passed to backup_init).
"""
import argparse, json, os, subprocess, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling")); sys.path.insert(0, str(HERE.parent.parent))
import common

DTYPE = "double"
OUT = HERE / "build" / DTYPE
SRCS = [HERE / "backup_staged.py", HERE / "backup_casadi.py", HERE.parent.parent / "profiling" / "panda_casadi.py"]


def plan_paths(ml_head=0, ml_step=0):
    return OUT / f"backup_head_ml{ml_head}.plan", OUT / f"backup_step_ml{ml_step}.plan"


def build_plans(ml_head=0, ml_step=0, force=False):
    import cusadi2
    from backup_staged import build_head_function, build_step_function
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "kernel.cu").exists():
        cusadi2.write_kernel_source(str(OUT / "kernel.cu"), dtype=DTYPE)
    newest = max(s.stat().st_mtime for s in SRCS)
    for which, ml, builder in (("head", ml_head, build_head_function), ("step", ml_step, build_step_function)):
        path = OUT / f"backup_{which}_ml{ml}.plan"
        if path.exists() and path.stat().st_mtime > newest and not force:
            continue
        t0 = time.perf_counter(); res = builder(); fn, info = res[0], res[-1]; t_b = time.perf_counter() - t0
        t0 = time.perf_counter(); cusadi2.write_plan(fn, str(path), n_stages=1, max_live=ml or cusadi2.device_max_live(DTYPE)); t_p = time.perf_counter() - t0
        st = cusadi2.read_plan(str(path))
        summary = dict(which=which, n_levels=len(st["level_widths"]), max_level_width=int(max(st["level_widths"])),
                       median_level_width=float(__import__("numpy").median(st["level_widths"])), peak_live_slots=st["peak_live_slots"],
                       max_live_arg=ml, casadi_build_s=t_b, plan_write_s=t_p, **info)
        path.with_name(path.name.replace(".plan", "_stats.json")).write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps(summary, default=str), flush=True)


def build_lib(force=False):
    import jax, cusadi2
    so = OUT / "libbackup_ffi.so"; cu = HERE / "backup_ffi.cu"
    if so.exists() and so.stat().st_mtime > max(cu.stat().st_mtime, (OUT / "kernel.cu").stat().st_mtime) and not force:
        return so
    nvcc = Path(sys.executable).parent / "nvcc"
    cc = jax.devices()[0].compute_capability.replace(".", "")
    subprocess.run([str(nvcc), "-std=c++17", "-O3", f"-arch=sm_{cc}", f"-DCUSADI2_SCALAR={DTYPE}", f"-I{cusadi2.INCLUDE_DIR}",
                    f"-I{jax.ffi.include_dir()}", str(cu), str(OUT / "kernel.cu"), "-shared", "-Xcompiler", "-fPIC", "-o", str(so)], check=True)
    return so


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-live-head", type=int, default=0); ap.add_argument("--max-live-step", type=int, default=0)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--no-lib", action="store_true")
    a = ap.parse_args()
    common.configure("gpu", x64=True)
    build_plans(a.max_live_head, a.max_live_step, a.force)
    if not a.no_lib:
        print("lib:", build_lib(a.force))
