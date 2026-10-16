"""Build the Panda safe-MPPI example.

    python build.py plans   [--T 10 15 20] [--dtypes double] [--rows all|scene]
    python build.py export  --T 20 --batches 1 256 --dtypes double [--backend frax mjx]
    python build.py lib     [--dtypes double]        (nvcc: safe_mppi.cu + kernel -> libsafe_mppi_{f32,f64}.so)
    python build.py unrolled --T 100                  (original-CusADi unrolled kernels of the same functions -> build/unrolled/*.so)
    python build.py cpu      --T 100                  (CasADi C codegen of the same functions, gcc -O3 -> build/cpu/*.so)
    python build.py timer                             (nvcc: time_step.cu -> time_step_{f32,f64})

build/plans/safe_model_T{T}_{dtype}.plan and safe_integrate_{dtype}.plan (+ .json stats), kernel_{dtype}.cu,
build/programs/{safe,mjx}_model_T{T}_B{B}_{dtype}.mlir and {safe,mjx}_integrate_B{B}_{dtype}.mlir (jax.export, StableHLO
bytecode, concrete batch; "safe" = frax + cbfpy, "mjx" = MJX + cbfpy), programs/compile_options.pb,
build/unrolled/libunrolled_{model_T{T},integrate}_{dtype}.so, build/cpu/libcpu_T{T}_{dtype}.so, meta.json.
The driver (safe_mppi.cu) loads plans, programs and libraries by name.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "profiling"))
OUT = HERE / "build"
NVCC = os.environ.get("NVCC") or str(Path(sys.executable).parent / "nvcc")


def arch():
    return "sm_" + subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).decode().split()[0].replace(".", "")


def plan_paths(T, dtype, rows="all"):
    sfx = "" if rows == "all" else "_" + rows
    return OUT / "plans" / f"safe_model_T{T}{sfx}_{dtype}.plan", OUT / "plans" / f"safe_integrate_{dtype}.plan"


def _write(fn, path, dtype, info):
    import cusadi2
    t0 = time.perf_counter(); cusadi2.write_plan(fn, str(path), max_live=cusadi2.device_max_live(dtype)); t_p = time.perf_counter() - t0
    st = cusadi2.read_plan(str(path))
    summary = dict(dtype=dtype, n_levels=len(st["level_widths"]), peak_live_slots=st["peak_live_slots"],
                   shared_bytes=st["peak_live_slots"] * (4 if dtype == "float" else 8), plan_write_s=t_p, **info)
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    print(path.name, json.dumps(summary, default=str), flush=True)
    return summary


def build_plans(Ts, dtypes, force=False, rows="all"):
    import cusadi2
    from model import build_model_function, build_integrate_function, DT, N_KNOT, NX, ND, NPAR, I_TAU, I_COST, PARAMS, knot_dt
    (OUT / "plans").mkdir(parents=True, exist_ok=True)
    for dtype in dtypes:
        k = OUT / f"kernel_{dtype}.cu"
        if not k.exists():
            cusadi2.write_kernel_source(str(k), dtype=dtype)
    meta_p = OUT / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta.update(dt=DT, n_knot=N_KNOT, nx=NX, nd=ND, npar=NPAR, i_tau=I_TAU, i_cost=I_COST, params=[n for n, _ in PARAMS])
    meta.setdefault("plans", {})
    fi = None
    for dtype in dtypes:
        _, pi = plan_paths(0, dtype)
        if not pi.exists() or force:
            if fi is None:
                fi, _, info_i = build_integrate_function()
            meta["plans"][pi.stem] = _write(fi, pi, dtype, info_i)
    for T in Ts:
        fm = None
        for dtype in dtypes:
            pm, _ = plan_paths(T, dtype, rows)
            if pm.exists() and not force:
                continue
            if fm is None:
                fm, _, info_m = build_model_function(T, rows=rows); info_m["knot_dt"] = knot_dt(T)
            meta["plans"][pm.stem] = _write(fm, pm, dtype, info_m)
    meta_p.write_text(json.dumps(meta, indent=1, default=str))


def export_programs(Ts, batches, dtypes, force=False, backends=("frax",)):
    import common
    x64 = "double" in dtypes
    if "jax" not in sys.modules:
        common.configure("gpu", x64=x64)
    import jax
    (OUT / "programs").mkdir(parents=True, exist_ok=True)
    from jax._src import compiler
    co = compiler.get_compile_options(num_replicas=1, num_partitions=1, device_assignment=jax.devices()[:1])
    (OUT / "programs" / "compile_options.pb").write_bytes(co.SerializeAsString())
    for backend in backends:
        mod = __import__("frax_model" if backend == "frax" else "mjx_model")
        for T in Ts:
            for dtype in dtypes:
                t0 = time.perf_counter()
                paths = mod.export_programs(T, dtype, OUT / "programs", batches)
                print(f"exported {backend} T={T} {dtype} batches {batches}: {len(paths)} programs ({time.perf_counter() - t0:.1f} s)", flush=True)


def build_unrolled(Ts, dtypes, force=False):
    """The original CusADi's execution model on the same CasADi functions: cusadi2.unrolled codegen (one thread per
    instance, one statement per instruction, thread-major work vector), one nvcc per function."""
    import cusadi2
    from model import build_model_function, build_integrate_function
    d = OUT / "unrolled"; d.mkdir(parents=True, exist_ok=True)
    for dtype in dtypes:
        so = d / f"libunrolled_integrate_{dtype}.so"
        if not so.exists() or force:
            fi, _, _ = build_integrate_function()
            src = cusadi2.write_unrolled_source(fi, str(d / f"safe_integrate_{dtype}.cu"), dtype=dtype, n_stages=1)
            nvcc([src], so, dtype, shared=True)
        for T in Ts:
            so = d / f"libunrolled_model_T{T}_{dtype}.so"
            if so.exists() and not force:
                continue
            fm, _, info = build_model_function(T)
            src = cusadi2.write_unrolled_source(fm, str(d / f"safe_model_T{T}_{dtype}.cu"), dtype=dtype, n_stages=1)
            nvcc([src], so, dtype, shared=True)


def build_cpu(Ts, dtypes, force=False):
    """CasADi's C code generation of the same two functions, gcc -O3 -march=native, one shared library per T."""
    import casadi as ca
    from model import build_model_function, build_integrate_function
    d = OUT / "cpu"; d.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC", "gcc")
    for dtype in dtypes:
        for T in Ts:
            so = d / f"libcpu_T{T}_{dtype}.so"
            if so.exists() and not force:
                continue
            fm, _, _ = build_model_function(T); fi, _, _ = build_integrate_function()
            stem = f"cpu_T{T}_{dtype}"
            gen = ca.CodeGenerator(stem, {"with_header": True, "casadi_real": dtype, "casadi_int": "long long int"})
            gen.add(fm); gen.add(fi); gen.generate(str(d) + os.sep)
            cmd = [cc, "-O3", "-march=native", "-fPIC", "-shared", str(d / f"{stem}.c"), "-lm", "-o", str(so)]
            t0 = time.perf_counter(); subprocess.run(cmd, check=True); print(f"{cc} {so.name}: {time.perf_counter() - t0:.1f} s", flush=True)


def nvcc(srcs, out, dtype, shared, extra=()):
    import cusadi2
    import jax
    cmd = [NVCC, "-std=c++17", "-O3", f"-arch={arch()}", f"-DCUSADI2_SCALAR={dtype}", f"-I{cusadi2.INCLUDE_DIR}",
           "-I/home/jovi/code/xla", "-Xcompiler", "-fopenmp", *map(str, srcs), "-o", str(out), "-ldl", "-lgomp", *extra]
    if shared:
        cmd += ["-shared", "-Xcompiler", "-fPIC"]
    t0 = time.perf_counter(); subprocess.run(cmd, check=True); print(f"nvcc {out.name}: {time.perf_counter() - t0:.1f} s", flush=True)


def n_cbf():
    """CBF row count of the model plans (meta.json), compiled into the driver as NCBF_ROWS."""
    ns = {v["n_cbf"] for v in json.loads((OUT / "meta.json").read_text())["plans"].values() if "n_cbf" in v}
    assert len(ns) == 1, f"plans disagree on the CBF row count: {ns}"
    return ns.pop()


def build_lib(dtypes):
    import cusadi2
    for dtype in dtypes:
        k = OUT / f"kernel_{dtype}.cu"
        if not k.exists():
            cusadi2.write_kernel_source(str(k), dtype=dtype)
        tag = "f32" if dtype == "float" else "f64"
        nvcc([HERE / "safe_mppi.cu", k], OUT / f"libsafe_mppi_{tag}.so", dtype, shared=True, extra=(f"-DNCBF_ROWS={n_cbf()}",))


def build_timer(dtypes):
    import cusadi2
    for dtype in dtypes:
        k = OUT / f"kernel_{dtype}.cu"
        if not k.exists():
            cusadi2.write_kernel_source(str(k), dtype=dtype)
        tag = "f32" if dtype == "float" else "f64"
        nvcc([HERE / "time_step.cu", k], OUT / f"time_step_{tag}", dtype, shared=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["plans", "export", "lib", "unrolled", "cpu", "timer"])
    ap.add_argument("--backend", nargs="+", default=["frax"], choices=["frax", "mjx"])
    ap.add_argument("--T", type=int, nargs="+", default=[10, 15, 20])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--dtypes", nargs="+", default=["double"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 256])
    ap.add_argument("--loops", nargs="+", default=["scan", "unrolled"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rows", default="all", choices=["all", "scene"])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.what in ("plans", "unrolled", "cpu"):
        # The CasADi graphs take their constants from frax's load_panda(); its tool offset Rz(-pi/4) is computed with JAX,
        # so without x64 the rotation is fp32-rounded (3e-8) and the CasADi evaluators would compute a slightly different
        # function from the fp64 frax / MJX exports (export_programs configures x64 itself).
        import common
        common.configure("cpu", x64=True)
    if a.what == "plans":
        build_plans(a.T, a.dtypes, a.force, a.rows)
    elif a.what == "export":
        export_programs(a.T, a.batches, a.dtypes, a.force, a.backend)
    elif a.what == "lib":
        build_lib(a.dtypes)
    elif a.what == "unrolled":
        build_unrolled(a.T, a.dtypes, a.force)
    elif a.what == "cpu":
        build_cpu(a.T, a.dtypes, a.force)
    else:
        build_timer(a.dtypes)
