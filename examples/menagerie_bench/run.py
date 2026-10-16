#!/usr/bin/env python
"""Per-robot MuJoCo Menagerie campaign of the paper_results latency benchmark.

For every robot in manifest.py (against the read-only Menagerie checkout,
default ../mujoco_menagerie) it runs the benchmark matrix of
examples/paper_results: a latency-vs-batch sweep of 4 kernel configurations
(full/staged/lp/batch) x 2 dtypes (double/float) x 3 functions (fk, fkjac,
fkhess), timed from C++ and from Python via ctypes, optionally plus the MJX and
mujoco-warp baselines. Both backends load the same Menagerie MJCF (see model.py
and README.md); a per-robot compatibility gate runs first.

The campaign is phased so the GPU sweeps never share the machine with a compile:

  1. gate       one subprocess per robot (cached in build/<robot>/gate.json)
  2. compile    one subprocess per (robot, dtype), N_JOBS of them in parallel
                (nvcc does not parallelize; the machine's cores do): CasADi
                codegen -> .plan files (windowed by cusadi2 to this GPU's shared
                memory when needed, recorded), the original-cusadi unrolled
                kernel per function (nvcc, timed, with a per-kernel time budget:
                a kernel that fails or times out is recorded and its staged /
                batch rows become nan), then the latency binary and ctypes
                library. Artifacts live in build/<robot>/<dtype>/ under a stamp
                of every generated source, so a robot whose sources are
                unchanged is never recompiled -- by a later run of the same
                campaign, or by `--runs N`, which compiles once and sweeps N times.
  3. sweep      sequential, GPU only: for run k, results/<robot>/runs/<k>/
                latency_<dtype>.csv (C++) and latency_py_<dtype>.csv (Python)
  4. cpu        the paper's CPU reference (CasADi C codegen, gcc -O3, OpenMP;
                "CPU idealized" and "CPU transfer" = GPU->CPU->GPU copies in the
                timed region), compiled in the same parallel pool (gcc, cached
                by source sha in build/<robot>/<dtype>/cpu/) and then swept
                sequentially per run: results/<robot>/runs/<k>/cpu_<dtype>.json.
                --no-cpu skips it; --cpu-only runs nothing else.
  5. baselines  MJX (fp64/fp32: fk, analytical jac, AD hess) and mujoco-warp
                (fp32: fk, analytical jac) per robot, smallest first, one
                GPU process at a time, unless --no-baselines (--baselines-only
                runs nothing else). Every shape is JIT-compiled once and timed
                --runs times (results/<robot>/runs/<k>/{mjx_<fp>,warp}.json,
                flushed after every shape). A VRAM OOM is a null with its
                reason; a process killed by earlyoom or hung on VRAM (no
                heartbeat for CUSADI2_MJX_STALL s, default 3600) is restarted
                and the shape that took it down is recorded and skipped.
                frax (fp64/fp32: fk, analytical jac, AD hess; frax_model.py)
                follows per robot behind its own gate (build/<robot>/
                frax_gate.json), into results/<robot>/runs/<k>/frax_<fp>.json.
                --frax-only runs the gates and the frax stages and nothing
                else (no plots, no summary).
  6. plot       per-robot figures (mean +- std over runs), then the summary

Every phase is resumable: gate JSON, compile stamp and sweep CSVs are skipped
when present (--force redoes them). Every failure is a machine-recorded stage
in results/<robot>/status.json, never a crash of the campaign.

    python run.py [--runs 5] [--no-baselines | --baselines-only | --frax-only] [--no-cpu | --cpu-only] [--jobs 8] [--robot R] [--force]
    python run.py --summarize          # cross-robot outputs only

Env overrides: CUSADI2_GPU (device index, default "0"), NVCC, CUSADI2_JOBS,
CUSADI2_NVCC_TIMEOUT (seconds per unrolled kernel, default 7200), CC ("gcc"),
CUSADI2_CC_TIMEOUT (seconds per CPU executable, default 7200), CUSADI2_CPU_N_TIMED
(paper_results' n_timed=2000: repetitions = max(5, n_timed // n_items)),
CUSADI2_MENAGERIE, CUSADI2_MANIFEST; smoke-test knobs CUSADI2_BATCHES,
CUSADI2_N_WARMUP, CUSADI2_N_TIMED, CUSADI2_RESULTS, CUSADI2_BUILD.
CUSADI2_FRAX_GPU_ONLY=1 adds frax's *_batched_gpu series (the marginal GPU time
per launch, as MJX's); it is off by default and can be added to finished runs.

CUSADI2_STEMS=fkjac (a comma list) re-measures the listed functions, after a
change of their definition or of the kernel: the compile phases rebuild whatever
source changed as always, the cusadi2 and CPU sweeps time only the listed
functions and merge their rows into the existing per-run files (the other
functions' measurements are kept), each run recorded against the sha of what it
measured so the re-measurement is resumable. Listing every function re-measures
the whole sweep in place (with --no-cpu when the CPU side is to be kept). The MJX / mujoco-warp results do not depend on
these functions and stay as recorded.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BUILD = os.environ.get("CUSADI2_BUILD", os.path.join(HERE, "build"))
RESULTS = os.environ.get("CUSADI2_RESULTS", os.path.join(HERE, "results"))

DEVICE = os.environ.get("CUSADI2_GPU", "0")
NVCC = os.environ.get("NVCC", "nvcc")
N_JOBS = int(os.environ.get("CUSADI2_JOBS", "6"))           # parallel compile subprocesses
HEAVY_INSTR = 60000      # a robot is heavy when its per-stage Hessian has this many instructions
                         # (nvcc on the x8-unrolled kernel then needs several GB; see results/dag)
NVCC_TIMEOUT_S = float(os.environ.get("CUSADI2_NVCC_TIMEOUT", "7200"))   # per unrolled kernel
MENAGERIE = os.path.abspath(os.environ.get("CUSADI2_MENAGERIE",
                                           os.path.join(HERE, "..", "mujoco_menagerie")))

# Stage-axis horizon (fixed) and batch sweep (the x-axis), identical to paper_results.
N_STAGES = 8
BATCHES = [int(b) for b in
           os.environ.get("CUSADI2_BATCHES", "1,4,16,64,256,1024,4096,16384").split(",")]
N_WARMUP = int(os.environ.get("CUSADI2_N_WARMUP", "5"))
N_TIMED = int(os.environ.get("CUSADI2_N_TIMED", "20"))

MJX_BATCHES = BATCHES
MJX_N_WARMUP = 8
MJX_N_EVALS = 20

DTYPES = ("double", "float")          # cusadi2 scalar types
DT2FP = {"double": "fp64", "float": "fp32"}

# CSV function stems; plans are <stem>.plan / <stem>_unr.plan in the robot's plan dir.
STEMS = ["fk", "fkjac", "fkhess"]
SWEEP_STEMS = [s for s in os.environ.get("CUSADI2_STEMS", "").split(",") if s] or STEMS   # functions the sweeps time
RESWEEP = bool(os.environ.get("CUSADI2_STEMS"))    # re-measure SWEEP_STEMS, merging into the existing run files
assert set(SWEEP_STEMS) <= set(STEMS), f"CUSADI2_STEMS: unknown function in {SWEEP_STEMS}"
LATENCY_CU = os.path.join(HERE, "..", "paper_results", "latency.cu")   # shared driver

# Subprocess timeouts (seconds); expiry is recorded as status "timeout".
STAGE_TIMEOUT_S = {"gate": 900, "frax_gate": 900, "pylib": 7200, "mjx": 7200, "warp": 7200}

# CPU reference (CasADi C codegen, gcc, OpenMP): the driver source, cuda_root()
# and the thread list come from paper_results/bench.py unchanged, so the timed
# region is byte-identical to the paper's "CPU idealized" / "CPU transfer".
BENCH_PY = os.path.join(HERE, "..", "paper_results", "bench.py")
CC = os.environ.get("CC", "gcc")
CC_TIMEOUT_S = float(os.environ.get("CUSADI2_CC_TIMEOUT", "7200"))      # per executable
CPU_N_TIMED = int(os.environ.get("CUSADI2_CPU_N_TIMED", "2000"))        # paper_results run_cpu default
CPU_MEM_FRACTION = 0.9      # skip a CPU point whose host buffers exceed this fraction of MemAvailable


# ---------------------------------------------------------------------------
# Manifest / menagerie checkout
# ---------------------------------------------------------------------------
def load_manifest(path=None):
    path = os.path.abspath(path or os.environ.get("CUSADI2_MANIFEST")
                           or os.path.join(HERE, "manifest.py"))
    if not os.path.exists(path):
        sys.exit(f"manifest not found: {path}\n"
                 "menagerie_bench is manifest-driven — provide manifest.py next to "
                 "run.py (or point CUSADI2_MANIFEST / --manifest at one).")
    spec = importlib.util.spec_from_file_location("menagerie_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.environ["CUSADI2_MANIFEST"] = path      # subprocess modes resolve the same one
    return mod


def robot_xml(manifest, robot):
    xml = manifest.ROBOTS[robot].get("xml")
    return os.path.join(MENAGERIE, xml) if xml else None


def _git(args):
    try:
        return subprocess.check_output(["git", "-C", MENAGERIE] + args,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def check_menagerie(manifest):
    """Warn loudly if the menagerie checkout is dirty or its HEAD differs from
    manifest.MENAGERIE_COMMIT. Returns the actual HEAD (or None)."""
    head = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain"])
    expected = getattr(manifest, "MENAGERIE_COMMIT", None)
    problems = []
    if head is None:
        problems.append(f"could not read git HEAD of {MENAGERIE}")
    elif expected and head != expected:
        problems.append(f"menagerie HEAD {head} != manifest.MENAGERIE_COMMIT {expected}")
    if dirty:
        problems.append(f"menagerie checkout is DIRTY:\n{dirty[:1000]}")
    if problems:
        bar = "!" * 78
        print(f"\n{bar}\n!!! MENAGERIE CHECKOUT WARNING\n" +
              "\n".join("!!! " + p for p in problems) + f"\n{bar}\n", file=sys.stderr)
    return head


def gpu_arch() -> str:
    cc = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", DEVICE]
    ).decode().splitlines()[0].strip()
    return "sm_" + cc.replace(".", "")


# ---------------------------------------------------------------------------
# compile phase (subprocess: run.py --compile --robot R --dtype D)
# ---------------------------------------------------------------------------
def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _run_timed(cmd, timeout, log):
    """nvcc with a time budget; (seconds, error or None). Output appended to log."""
    t0 = time.perf_counter()
    with open(log, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n"); f.flush()
        try:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            err = None if p.returncode == 0 else f"exit code {p.returncode}"
        except subprocess.TimeoutExpired:
            err = f"timeout after {timeout:.0f}s"
    return time.perf_counter() - t0, err


def compile_robot(manifest, robot, dtype) -> int:
    """CasADi codegen, original-cusadi unrolled kernels, latency binary + ctypes
    library for one robot x dtype, exactly as paper_results.bench.run_cusadi2.
    Generated sources are hashed into build/<robot>/<dtype>/stamp.json; when the
    stamp matches and every artifact exists, nothing is recompiled."""
    import model
    import cusadi2
    from cusadi2.dag import extract_level_plan

    xml = robot_xml(manifest, robot)
    m = model.load_mujoco(xml)
    pmodel = model.load_pinocchio(xml, m=m)   # SAME repaired load path as the gate
    names = model.body_names(m)
    corr = model.root_correction(m, pmodel, model.qpos_mj_to_pin_index(m, pmodel))
    builders = {"fk": model.build_fk, "fkjac": model.build_fkjac, "fkhess": model.build_fkhess}

    outdir = os.path.join(BUILD, robot, dtype)
    plan_dir = os.path.join(outdir, "plans")
    unr_dir = os.path.join(outdir, "unrolled")
    resdir = os.path.join(RESULTS, robot)
    for d in (plan_dir, unr_dir, resdir):
        os.makedirs(d, exist_ok=True)
    log = os.path.join(outdir, "compile.log")
    comp_json = os.path.join(resdir, f"compile_cusadi2_{dtype}.json")
    stamp_path = os.path.join(outdir, "stamp.json")
    binpath, libpath = os.path.join(outdir, "latency"), os.path.join(outdir, "latency_lib.so")

    # codegen (cheap, deterministic): plans + unrolled sources
    cusadi2.write_kernel_source(os.path.join(outdir, "kernel.cu"), dtype=dtype)
    print(f"=== [{robot}] {dtype}: codegen (N={N_STAGES}) ===", flush=True)
    codegen_s, codegen_unrolled_s, meta = {}, {}, {}
    unrolled = []                                  # (name, .cu) in table order
    bytes_per = 4 if dtype == "float" else 8
    max_live = cusadi2.device_max_live(dtype, int(DEVICE))
    for stem in STEMS:
        t0 = time.perf_counter()
        fn = builders[stem](pmodel, names, name=stem, root_corr=corr)
        cusadi2.write_plan(fn, os.path.join(plan_dir, f"{stem}.plan"), n_stages=N_STAGES, max_live=max_live)
        fu = model.unroll(fn, N_STAGES)
        cusadi2.write_plan(fu, os.path.join(plan_dir, f"{stem}_unr.plan"), max_live=max_live)
        codegen_s[stem] = time.perf_counter() - t0
        for f_, nst in ((fn, N_STAGES), (fu, 1)):     # original cusadi codegen: per function
            t0 = time.perf_counter()
            src = cusadi2.write_unrolled_source(f_, os.path.join(unr_dir, f"{f_.name()}.cu"),
                                                dtype=dtype, n_stages=nst)
            codegen_unrolled_s[f_.name()] = time.perf_counter() - t0
            unrolled.append((f_.name(), src))
        ps = cusadi2.read_plan(os.path.join(plan_dir, f"{stem}.plan"))
        pu = cusadi2.read_plan(os.path.join(plan_dir, f"{stem}_unr.plan"))
        # windowed = the plan cusadi2 wrote has more levels than the plain longest-path plan
        meta[stem] = {
            "nnz_out": int(fn.nnz_out(0)),
            "n_instr": int(fn.n_instructions()),
            "n_instr_unr": int(fu.n_instructions()),
            "sz_w": ps["sz_w"], "sz_w_unr": pu["sz_w"],
            "smem_kb": ps["peak_live_slots"] * bytes_per / 1024,
            "smem_unr_kb": pu["peak_live_slots"] * bytes_per / 1024,
            "levels": len(ps["level_widths"]), "levels_unr": len(pu["level_widths"]),
            "windowed": len(ps["level_widths"]) != extract_level_plan(fn).n_levels,
            "windowed_unr": len(pu["level_widths"]) != extract_level_plan(fu).n_levels,
        }
        print(f"  {stem}: nout={fn.nnz_out(0)} instr={fn.n_instructions()} sz_w={ps['sz_w']} "
              f"smem={meta[stem]['smem_kb']:.1f}KB{' WINDOWED' if meta[stem]['windowed'] else ''} | "
              f"unrolled instr={fu.n_instructions()} smem={meta[stem]['smem_unr_kb']:.1f}KB"
              f"{' WINDOWED' if meta[stem]['windowed_unr'] else ''}  codegen={codegen_s[stem]:.2f}s",
              flush=True)

    # stamp.json: sha of each unrolled kernel source whose .o was built (or that
    # failed), and of the driver's sources; a kernel whose source is unchanged is
    # never recompiled, the driver is only relinked when its inputs changed.
    stamp = json.load(open(stamp_path)) if os.path.exists(stamp_path) else {}
    old_comp = json.load(open(comp_json)) if os.path.exists(comp_json) else {}
    kernels_old, failed_old = stamp.get("kernels", {}), stamp.get("failed", {})
    driver_src = {os.path.relpath(p, outdir): _sha(p) for p in
                  [os.path.join(outdir, "kernel.cu"), LATENCY_CU, os.path.join(cusadi2.INCLUDE_DIR, "cusadi2.h")]}
    open(log, "a").close()

    arch = f"-arch={gpu_arch()}"
    nvcc_unrolled_s, unrolled_failed, objs, built, kernels, killed = {}, {}, [], [], {}, []
    n_cached = 0
    for name, src in unrolled:                     # one nvcc per function, timed: the original's cost
        obj = src[:-3] + ".o"
        h = _sha(src)
        if kernels_old.get(name) == h and os.path.exists(obj):          # unchanged source, built before
            kernels[name] = h
            nvcc_unrolled_s[name] = (old_comp.get("nvcc_unrolled_s") or {}).get(name)
            objs.append(obj); built.append(name); n_cached += 1
            continue
        if failed_old.get(name, {}).get("sha") == h:                    # unchanged source, gave up before
            unrolled_failed[name] = failed_old[name]["error"]
            continue
        secs, err = _run_timed([NVCC, "-std=c++17", "-O3", arch, "-Xcompiler", "-fPIC", "-c", src, "-o", obj],
                               NVCC_TIMEOUT_S, log)
        if err:
            if os.path.exists(obj):
                os.remove(obj)
            if (err in ("exit code 137", "exit code 143") or err.startswith("exit code -")) \
                    and not os.environ.get("CUSADI2_STAMP_KILLS"):
                print(f"  nvcc {name}: KILLED ({err}, e.g. earlyoom) -> not stamped, retried next time", flush=True)
                killed.append(name)
                continue
            if err in ("exit code 137", "exit code 143") or err.startswith("exit code -"):
                err += " (killed while compiling alone: exceeds this machine's memory)"
            unrolled_failed[name] = err
            print(f"  nvcc {name}: FAILED ({err}) -> staged/batch rows nan for this kernel", flush=True)
            continue
        nvcc_unrolled_s[name] = secs
        kernels[name] = h
        objs.append(obj); built.append(name)
        print(f"  nvcc {name}: {secs:.1f}s (original cusadi, per function)", flush=True)
    if n_cached:
        print(f"  {n_cached} unrolled kernel(s) unchanged since last build, nvcc skipped", flush=True)
    if killed:                      # keep what was built (stamped), report the rest for a retry
        with open(stamp_path, "w") as f:
            json.dump({"kernels": kernels, "driver": stamp.get("driver"),
                       "failed": {n: {"sha": _sha(dict(unrolled)[n]), "error": e} for n, e in unrolled_failed.items()},
                       "when": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        print(f"  compile incomplete: killed nvcc for {killed}; rerun to retry", flush=True)
        return 3
    with open(os.path.join(outdir, "unrolled_table.h"), "w") as f:
        f.write("// generated by run.py: original-cusadi kernels linked into latency.cu\n")
        for name in built:
            f.write(f'extern "C" int {name}_unrolled_sz_w();\nextern "C" int {name}_unrolled_n_stages();\n'
                    f'extern "C" int {name}_unrolled_launch(const S* const*, S* const*, S*, int, int, cudaStream_t);\n')
        f.write("static const UnrolledFn UNROLLED[] = {\n" + "".join(
            f'    {{"{n}", {n}_unrolled_sz_w, {n}_unrolled_n_stages, {n}_unrolled_launch}},\n' for n in built)
            + ("    {nullptr, nullptr, nullptr, nullptr},\n" if not built else "") + "};\n")
        f.write("static const char* FUNC_NAMES[] = {" + ", ".join(f'"{s}"' for s in STEMS) + "};\n")

    driver_inputs = dict(driver_src, built={n: kernels[n] for n in built})   # relinked when a kernel changed
    if stamp.get("driver") == driver_inputs and all(os.path.exists(p) for p in (binpath, libpath)) \
            and "nvcc_s" in old_comp:
        nvcc_s, driver_cached = old_comp["nvcc_s"], True
        print("  latency binary + library unchanged, nvcc skipped", flush=True)
    else:
        driver_cached = False
        common = [NVCC, "-std=c++17", "-O3", arch, f"-DCUSADI2_SCALAR={dtype}",
                  f"-I{cusadi2.INCLUDE_DIR}", f"-I{outdir}", LATENCY_CU, os.path.join(outdir, "kernel.cu")]
        nvcc_s, err = _run_timed(common + objs + ["-o", binpath], NVCC_TIMEOUT_S, log)
        if err:
            print(f"  nvcc latency binary FAILED: {err} (see {log})", flush=True)
            return 1
        _, err = _run_timed(common + objs + ["-shared", "-Xcompiler", "-fPIC", "-DCUSADI2_BENCH_LIB", "-o", libpath],
                            NVCC_TIMEOUT_S, log)
        if err:
            print(f"  nvcc latency library FAILED: {err} (see {log})", flush=True)
            return 1
        print(f"  nvcc: {nvcc_s:.1f}s (cusadi2 kernel + driver; function-independent)", flush=True)
    with open(comp_json, "w") as f:
        json.dump({"robot": robot, "dtype": dtype, "codegen_s": codegen_s, "nvcc_s": nvcc_s,
                   "codegen_unrolled_s": codegen_unrolled_s, "nvcc_unrolled_s": nvcc_unrolled_s,
                   "unrolled_failed": unrolled_failed, "nvcc_jobs": N_JOBS,
                   "cached": n_cached == len(unrolled) and driver_cached,
                   "nq": int(m.nq), "nv": int(m.nv), "n_bodies": len(names), "meta": meta}, f, indent=2)
    with open(stamp_path, "w") as f:
        json.dump({"kernels": kernels, "driver": driver_inputs,
                   "failed": {n: {"sha": _sha(dict(unrolled)[n]), "error": e} for n, e in unrolled_failed.items()},
                   "when": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    return 0


# ---------------------------------------------------------------------------
# sweep phase (in-process: the C++ binary and the pylib subprocess)
# ---------------------------------------------------------------------------
def sweep_robot(robot, dtype, k, force=False) -> dict:
    """One run k of the C++ and Python-hosted sweeps for robot x dtype."""
    outdir = os.path.join(BUILD, robot, dtype)
    rundir = os.path.join(RESULTS, robot, "runs", str(k))
    os.makedirs(rundir, exist_ok=True)
    csv, pycsv = os.path.join(rundir, f"latency_{dtype}.csv"), os.path.join(rundir, f"latency_py_{dtype}.csv")
    partial = RESWEEP                       # CUSADI2_STEMS: time those functions, merge into the run's files
    marker = os.path.join(outdir, f"sweep_{k}.json")
    if partial:
        shas = {s: ":".join(_sha(os.path.join(outdir, "plans", f"{s}{u}.plan")) for u in ("", "_unr")) for s in SWEEP_STEMS}
        done = json.load(open(marker)) if os.path.exists(marker) else {}
        current = all(done.get(s) == h for s, h in shas.items())
    if not force and os.path.exists(csv) and os.path.exists(pycsv) and (not partial or current):
        return {"status": "pass", "cached": True}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE, CUSADI2_FUNCS=",".join(SWEEP_STEMS))
    out_csv, out_py = (csv + ".part", pycsv + ".part") if partial else (csv, pycsv)
    t0 = time.perf_counter()
    cmd = [os.path.join(outdir, "latency"), os.path.join(outdir, "plans"), str(N_WARMUP), str(N_TIMED)] \
        + [str(b) for b in BATCHES]
    with open(out_csv, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f).returncode
    if rc != 0:
        os.remove(out_csv)
        return {"status": "fail", "reason": f"latency binary exit code {rc}", "seconds": round(time.perf_counter() - t0, 1)}
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--pylib", "--dtype", DT2FP[dtype],
                            "--lib", os.path.join(outdir, "latency_lib.so"), "--out", out_py],
                           env=env, timeout=STAGE_TIMEOUT_S["pylib"], stdout=subprocess.DEVNULL)
        rec = {"status": "pass" if p.returncode == 0 else "fail"}
        if p.returncode != 0:
            rec["reason"] = f"pylib exit code {p.returncode}"
    except subprocess.TimeoutExpired:
        rec = {"status": "timeout", "reason": f"pylib timeout after {STAGE_TIMEOUT_S['pylib']}s"}
    if partial:                             # the run's files change only once both sweeps passed
        if rec["status"] == "pass":
            _merge_sweep_csv(csv, out_csv)
            _merge_sweep_csv(pycsv, out_py)
            with open(marker, "w") as f:
                json.dump(dict(done, **shas), f, indent=2)
        for part in (out_csv, out_py):
            if os.path.exists(part):
                os.remove(part)
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def _merge_sweep_csv(path, part):
    """Replace the rows of the functions in `part` (a sweep of SWEEP_STEMS) in
    the run's CSV `path`, keeping the other functions' rows and the STEMS order."""
    new = open(part).read().splitlines()
    old = open(path).read().splitlines() if os.path.exists(path) else new[:1]
    func = lambda row: row.split(",")[1]
    rows = [r for r in old[1:] if func(r) not in SWEEP_STEMS] + new[1:]
    rows.sort(key=lambda r: STEMS.index(func(r)))          # stable: batch / config order kept
    with open(path + ".tmp", "w") as f:
        f.write("\n".join(new[:1] + rows) + "\n")
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------------------
# Python-called cusadi2 sweep (subprocess: run.py --pylib) — identical to
# paper_results.run_pylib apart from the two generic function stems.
# ---------------------------------------------------------------------------
def run_pylib(lib_path, fp, batches, n_stages, n_warmup, n_timed, out_csv):
    """Time the native cusadi2 kernel from Python via ctypes — the host-language-
    symmetric counterpart to the MJX/warp wall-clock numbers. Same sweep and CSV
    schema as the C++ binary; wall_us here includes Python + FFI dispatch. Runs
    in its own process (clean CUDA context, same CUDA_VISIBLE_DEVICES treatment
    as the MJX subprocesses)."""
    import ctypes
    lib = ctypes.CDLL(lib_path)
    lib.c2_setup.restype = ctypes.c_int
    lib.c2_setup.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.c2_eval.restype = ctypes.c_double
    lib.c2_eval.argtypes = [ctypes.c_int]
    lib.c2_teardown.argtypes = [ctypes.c_int]

    dt = {"fp32": "f32", "fp64": "f64"}[fp]
    plan_dir = os.path.join(os.path.dirname(lib_path), "plans")
    # config -> (plan file, serial kernel?, stages per item), as paper_results.bench.config_plan
    configs = [("full", "", 0, n_stages), ("staged", "", 1, n_stages), ("lp", "_unr", 0, 1), ("batch", "_unr", 1, 1)]
    rows = ["dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us"]
    for stem in SWEEP_STEMS:
        for B in batches:
            for cfg, suffix, serial, nst in configs:
                h = lib.c2_setup(os.path.join(plan_dir, f"{stem}{suffix}.plan").encode(), serial, B)
                if h < 0:                       # infeasible (smem limit / would OOM)
                    rows.append(f"{dt},{stem},{cfg},{B},{nst},{B * nst},nan,nan")
                    continue
                for _ in range(n_warmup):
                    lib.c2_eval(h)
                kus, wus = [], []
                for _ in range(n_timed):
                    t0 = time.perf_counter()
                    k = lib.c2_eval(h)          # count upload + launch + kernel + sync
                    wus.append((time.perf_counter() - t0) * 1e6)
                    kus.append(k * 1e6)
                lib.c2_teardown(h)
                kus.sort(); wus.sort()
                k_med, w_med = kus[len(kus) // 2], wus[len(wus) // 2]
                rows.append(f"{dt},{stem},{cfg},{B},{nst},{B * nst},{k_med:.3f},{w_med:.3f}")
                print(f"[pylib] {stem:9s} {cfg:6s} B={B:6d}: wall {w_med:9.1f} us "
                      f"(kernel {k_med:.1f} us)")
    with open(out_csv, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"wrote {out_csv}")


# ---------------------------------------------------------------------------
# CPU reference: CasADi C codegen + gcc (subprocess: run.py --cpu-compile), then
# the timed sweep in-process (cpu_run_robot) — paper_results.bench.run_cpu split
# into a cached compile and per-run timing.
# ---------------------------------------------------------------------------
def _bench():
    """paper_results/bench.py as a module (CPU_DRIVER, cuda_root, CPU_THREADS)."""
    spec = importlib.util.spec_from_file_location("paper_bench", BENCH_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _killed(err):
    return err in ("exit code 137", "exit code 143") or err.startswith("exit code -")


def cpu_compile_robot(manifest, robot, dtype) -> int:
    """CasADi's own C code generation of each per-stage function, compiled with
    gcc -O3 -march=native -fopenmp against paper_results' CPU driver, twice:
    <stem> (data in host memory, "CPU idealized") and <stem>_transfer
    (-DTRANSFER: inputs cudaMemcpy'd device->host and outputs host->device inside
    the timed region, "CPU transfer"). compile_s = codegen + gcc of the idealized
    build, as paper_results records it. Cached by source sha in
    build/<robot>/<dtype>/cpu/stamp.json; a gcc killed under the parallel pool is
    left unstamped for a retry alone (rc 3), one killed alone is stamped failed."""
    import casadi as ca
    import model
    bench = _bench()
    xml = robot_xml(manifest, robot)
    m = model.load_mujoco(xml)
    pmodel = model.load_pinocchio(xml, m=m)
    names = model.body_names(m)
    corr = model.root_correction(m, pmodel, model.qpos_mj_to_pin_index(m, pmodel))
    builders = {"fk": model.build_fk, "fkjac": model.build_fkjac, "fkhess": model.build_fkhess}

    outdir = os.path.join(BUILD, robot, dtype, "cpu")
    resdir = os.path.join(RESULTS, robot)
    for d in (outdir, resdir):
        os.makedirs(d, exist_ok=True)
    driver = os.path.join(outdir, "driver.c")
    with open(driver, "w") as f:
        f.write(bench.CPU_DRIVER)
    cuda_inc, cuda_lib = bench.cuda_root()
    log = os.path.join(outdir, "compile.log")
    open(log, "a").close()
    stamp_path = os.path.join(outdir, "stamp.json")
    comp_json = os.path.join(resdir, f"cpu_compile_{dtype}.json")
    stamp = json.load(open(stamp_path)) if os.path.exists(stamp_path) else {}
    old = json.load(open(comp_json)) if os.path.exists(comp_json) else {}
    built_old, failed_old = stamp.get("built", {}), stamp.get("failed", {})
    flags = [CC, "-O3", "-march=native", "-fopenmp"]

    print(f"=== [{robot}] {dtype}: CasADi C codegen + gcc (CPU reference) ===", flush=True)
    compile_s, gcc_s, failed, built, sizes, killed = {}, {}, {}, {}, {}, []
    n_cached = 0
    for stem in STEMS:
        t0 = time.perf_counter()
        fn = builders[stem](pmodel, names, name=stem, root_corr=corr)
        gen = ca.CodeGenerator(stem, {"with_header": True, "casadi_real": dtype})
        gen.add(fn)
        gen.generate(outdir + os.sep)
        codegen_s = time.perf_counter() - t0
        sizes[stem] = {"nnz_in": int(sum(fn.nnz_in(i) for i in range(fn.n_in()))),
                       "nnz_out": int(sum(fn.nnz_out(i) for i in range(fn.n_out()))),
                       "n_instr": int(fn.n_instructions())}
        src = os.path.join(outdir, f"{stem}.c")
        exe, exe_tr = os.path.join(outdir, stem), os.path.join(outdir, stem + "_transfer")
        h = _sha(src) + ":" + _sha(driver) + ":" + " ".join(flags)
        if built_old.get(stem) == h and os.path.exists(exe) and os.path.exists(exe_tr):
            built[stem] = h
            compile_s[stem] = (old.get("compile_s") or {}).get(stem)
            gcc_s[stem] = (old.get("gcc_s") or {}).get(stem)
            n_cached += 1
            continue
        if failed_old.get(stem, {}).get("sha") == h:
            failed[stem] = failed_old[stem]["error"]
            continue
        common = flags + [f"-DFN={stem}", f'-DFN_H="{stem}.h"', f"-I{outdir}"]
        secs, err = _run_timed(common + [driver, src, "-lm", "-o", exe], CC_TIMEOUT_S, log)
        if not err:
            _, err = _run_timed(common + ["-DTRANSFER", f"-I{cuda_inc}", driver, src, f"-L{cuda_lib}", "-lcudart",
                                          f"-Wl,-rpath,{cuda_lib}", "-lm", "-o", exe_tr], CC_TIMEOUT_S, log)
        if err:
            for p in (exe, exe_tr):
                if os.path.exists(p):
                    os.remove(p)
            if _killed(err) and not os.environ.get("CUSADI2_STAMP_KILLS"):
                print(f"  gcc {stem}: KILLED ({err}, e.g. earlyoom) -> not stamped, retried next time", flush=True)
                killed.append(stem)
                continue
            if _killed(err):
                err += " (killed while compiling alone: exceeds this machine's memory)"
            failed[stem] = err
            print(f"  gcc {stem}: FAILED ({err}) -> no CPU rows for this function", flush=True)
            continue
        built[stem] = h
        gcc_s[stem] = secs
        compile_s[stem] = codegen_s + secs
        print(f"  {stem}: {sizes[stem]['n_instr']} instr, codegen {codegen_s:.1f}s + gcc {secs:.1f}s", flush=True)
    if n_cached:
        print(f"  {n_cached} CPU executable pair(s) unchanged since last build, gcc skipped", flush=True)
    with open(stamp_path, "w") as f:
        json.dump({"built": built,
                   "failed": {n: {"sha": _sha(os.path.join(outdir, f"{n}.c")) + ":" + _sha(driver) + ":" + " ".join(flags),
                                  "error": e} for n, e in failed.items()},
                   "when": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    if killed:
        print(f"  CPU compile incomplete: killed gcc for {killed}; rerun to retry", flush=True)
        return 3
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    with open(comp_json, "w") as f:
        json.dump({"robot": robot, "dtype": dtype, "cpu": cpu, "threads": os.cpu_count(),
                   "cc": " ".join(flags), "compile_s": compile_s, "gcc_s": gcc_s, "failed": failed,
                   "sizes": sizes, "cached": n_cached == len(STEMS), "gcc_jobs": N_JOBS}, f, indent=2)
    return 0


def _mem_available_bytes():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            return int(l.split()[1]) * 1024
    return 0


def cpu_run_robot(robot, dtype, k, force=False) -> dict:
    """One run k of paper_results' CPU sweep for robot x dtype ->
    results/<robot>/runs/<k>/cpu_<dtype>.json (same schema as paper_results'
    cpu_<dtype>.json): latency_us (one problem), seq_us[stem][threads][B] and
    seq_transfer_us[...] (B*N problems over that many OpenMP threads), compile_s.
    A point whose host buffers (n_items x (nnz_in + nnz_out) scalars) would exceed
    CPU_MEM_FRACTION of MemAvailable is skipped as null (paper_results' A1/H1
    never hit this; the driver would be OOM-killed), as is one whose driver fails."""
    outdir = os.path.join(BUILD, robot, dtype, "cpu")
    rundir = os.path.join(RESULTS, robot, "runs", str(k))
    os.makedirs(rundir, exist_ok=True)
    out = os.path.join(rundir, f"cpu_{dtype}.json")
    partial = RESWEEP                       # CUSADI2_STEMS: time those functions, merge into the run's file
    marker = os.path.join(outdir, f"run_{k}.json")
    if partial:
        stamp = os.path.join(outdir, "stamp.json")
        built = (json.load(open(stamp)) if os.path.exists(stamp) else {}).get("built", {})
        shas = {s: built.get(s) for s in SWEEP_STEMS}
        done = json.load(open(marker)) if os.path.exists(marker) else {}
        current = all(done.get(s) == h for s, h in shas.items())
    if not force and os.path.exists(out) and (not partial or current):
        return {"status": "pass", "cached": True}
    comp_json = os.path.join(RESULTS, robot, f"cpu_compile_{dtype}.json")
    if not os.path.exists(comp_json):
        return {"status": "skip", "reason": "no cpu_compile json"}
    comp = json.load(open(comp_json))
    bench = _bench()
    bytes_per = 4 if dtype == "float" else 8
    t0 = time.perf_counter()
    lat, seq, seq_tr, skipped = {}, {}, {}, {}
    if partial and os.path.exists(out):     # keep the other functions' measurements of this run
        prev = json.load(open(out))
        keep = lambda d: {s: v for s, v in (d or {}).items() if s not in SWEEP_STEMS}
        lat, seq, seq_tr = keep(prev.get("latency_us")), keep(prev.get("seq_us")), keep(prev.get("seq_transfer_us"))
        skipped = {key: v for key, v in (prev.get("skipped") or {}).items()
                   if key.split()[0].removesuffix("_transfer") not in SWEEP_STEMS}

    def run(exe, n_items, threads, stem):      # median us for n_items evaluations over threads
        sz = comp["sizes"][stem]
        need = n_items * (sz["nnz_in"] + sz["nnz_out"]) * bytes_per
        avail = _mem_available_bytes()
        if need > CPU_MEM_FRACTION * avail:
            skipped[f"{stem} B*N={n_items}"] = f"buffers {need / 2**30:.1f} GB > {CPU_MEM_FRACTION:.0%} of available {avail / 2**30:.1f} GB"
            return None
        reps = max(5, CPU_N_TIMED // n_items)
        try:
            return float(subprocess.check_output([exe, str(max(1, reps // 10)), str(reps), str(n_items), str(threads)],
                                                 stderr=subprocess.DEVNULL))
        except (subprocess.CalledProcessError, ValueError) as e:
            skipped[f"{os.path.basename(exe)} B*N={n_items} thr={threads}"] = repr(e)[:200]
            return None

    for stem in SWEEP_STEMS:
        exe, exe_tr = os.path.join(outdir, stem), os.path.join(outdir, stem + "_transfer")
        if stem in comp.get("failed", {}) or not (os.path.exists(exe) and os.path.exists(exe_tr)):
            continue
        lat[stem] = run(exe, 1, 1, stem)
        seq[stem] = {str(t): {str(B): run(exe, B * N_STAGES, t, stem) for B in BATCHES} for t in bench.CPU_THREADS}
        seq_tr[stem] = {str(t): {str(B): run(exe_tr, B * N_STAGES, t, stem) for B in BATCHES} for t in bench.CPU_THREADS}
        top = str(BATCHES[-1])
        fmt = lambda d: "  ".join(f"{t}thr " + (f"{d[str(t)][top] / 1e3:7.1f} ms" if d[str(t)][top] is not None else "   null")
                                  for t in bench.CPU_THREADS)
        print(f"  [{robot} {dtype}] {stem}: {lat[stem] if lat[stem] is None else round(lat[stem], 2)} us/problem"
              f"   B={top}: {fmt(seq[stem])}   transfer: {fmt(seq_tr[stem])}", flush=True)
    order = lambda d: {s: d[s] for s in STEMS if s in d}
    with open(out + ".tmp", "w") as f:
        json.dump({"robot": robot, "dtype": dtype, "cpu": comp.get("cpu"), "threads": comp.get("threads"),
                   "n_stages": N_STAGES, "cpu_threads": bench.CPU_THREADS, "n_timed": CPU_N_TIMED,
                   "latency_us": order(lat), "seq_us": order(seq), "seq_transfer_us": order(seq_tr),
                   "compile_s": comp.get("compile_s", {}), "failed": comp.get("failed", {}),
                   "skipped": skipped}, f, indent=2)
    os.replace(out + ".tmp", out)
    if partial:
        with open(marker, "w") as f:
            json.dump(dict(done, **shas), f, indent=2)
    rec = {"status": "pass" if lat else "skip", "seconds": round(time.perf_counter() - t0, 2)}
    if not lat:
        rec["reason"] = "no CPU executable built"
    if skipped:
        rec["skipped"] = len(skipped)
    return rec


# ---------------------------------------------------------------------------
# MJX baseline (subprocess: run.py --mjx --robot R --dtype fpXX --runs K) —
# paper_results.run_mjx (fk / jac / hess; batched, rollout, batched_gpu) with the
# Menagerie MJCF, every shape JIT-compiled ONCE and then timed K times
# (results/<robot>/runs/<k>/mjx_<fp>.json, paper schema), flushed after every
# shape. A shape that raises (VRAM OOM) is null with its reason; a shape that
# kills or hangs the process is recorded by the next attempt (see _Progress).
# ---------------------------------------------------------------------------
MJX_STALL_S = float(os.environ.get("CUSADI2_MJX_STALL", "3600"))   # no heartbeat this long -> hung, killed
BASELINE_ATTEMPTS = 8                                              # restarts per stage (one lost shape each)


def _disable_contacts_and_constraints(m):
    """The fair-comparison model setup for both baselines: no contact detection
    and no constraint solver at all. mjDSBL_CONTACT skips MJX's collision driver
    and contact rows (ncon = 0 at put_data); mjDSBL_CONSTRAINT makes
    make_constraint produce zero rows, and mjx.forward then returns qacc_smooth
    without ever tracing solver.solve -- the friction cone, joint limits and
    equality constraints enter no code path. The rollout's mjx.step is thus
    smooth dynamics + kinematics, the same workload class as the cusadi2 side.
    contype/conaffinity are zeroed as well so no geom pair is ever a candidate."""
    import mujoco
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    m.opt.noslip_iterations = 0     # meaningless without constraints; mujoco-warp's put_model rejects it otherwise


def _put_model_or_strip(xml_path, put_model, m, body_ids):
    """put_model(m), or -- when the backend refuses the model for a feature its
    dynamics lack (actuator plugins, body-transmission actuators, ...) -- the same
    model rebuilt from its spec with every actuator and plugin removed, which
    cannot change kinematics or Jacobians (xpos/xmat verified bit-identical).
    Returns (m, backend model, reason or None); recorded as actuators_stripped."""
    import mujoco
    try:
        return m, put_model(m), None
    except NotImplementedError as e:
        spec = mujoco.MjSpec.from_file(xml_path)
        for a in list(spec.actuators):
            spec.delete(a)
        for pl in list(spec.plugins):
            spec.delete(pl)
        m2 = spec.compile()
        _disable_contacts_and_constraints(m2)
        import model as _model
        assert [int(b) for b in _model.named_body_ids(m2)] == [int(b) for b in body_ids]
        reason = f"{type(e).__name__}: {e}"[:200]
        print(f"  put_model refused ({reason}); retrying with actuators/plugins stripped", flush=True)
        return m2, put_model(m2), reason


class _Progress:
    """build/<robot>/<tag>_progress.json: a heartbeat and the shape in flight.
    The orchestrator kills the process when the heartbeat stalls (a GPU op that
    hangs on VRAM never raises); the next attempt finds the shape still marked
    current, records it as failed and moves on. Also the set of shapes that
    raised, so a restart never retries them."""
    def __init__(self, path):
        self.path = path
        try:
            self.d = json.load(open(path))
        except Exception:
            self.d = {}
        self.d.setdefault("failed", {})
        self.d.setdefault("current", None)

    def take_over(self):
        cur = self.d.get("current")
        if cur:
            self.d["failed"][cur] = "process died on this shape (earlyoom kill or hung on VRAM): skipped"
            self.d["current"] = None
            print(f"  previous attempt died on {cur}: recorded, skipped", flush=True)
        self.beat()
        return self.d["failed"]

    def start(self, shape):
        self.d["current"] = shape
        self.beat()

    def done(self):
        self.d["current"] = None
        self.beat()

    def beat(self):
        self.d["heartbeat"] = time.time()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.d, f)
        os.replace(tmp, self.path)


def baseline_paths(robot, tag, runs):
    """results/<robot>/runs/<k>/<tag>.json for k < runs (tag: mjx_fp64 / mjx_fp32 / warp)."""
    return [os.path.join(RESULTS, robot, "runs", str(k), f"{tag}.json") for k in range(runs)]


def _load_runs(paths):
    """Existing per-run files -> (results per run, compile dict, meta) or empties."""
    res, comp, meta = [], {}, {}
    for p in paths:
        try:
            j = json.load(open(p))
        except Exception:
            j = {}
        res.append(j.get("results") or {})
        comp = comp or j.get("compile") or {}
        meta = meta or {k: v for k, v in j.items() if k not in ("results", "compile", "failed", "complete")}
    return res, comp, meta


def _flush_runs(paths, res, comp, meta, failed, complete):
    for k, p in enumerate(paths):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(meta, run=k, results=res[k], compile=comp, failed=failed, complete=complete), f, indent=2)
        os.replace(tmp, p)


def run_mjx(robot, xml_path, fp, batches, n_stages, n_warmup, n_evals, runs, progress):
    """Three quantities, all with analytical derivatives: fk (kinematics + com_pos,
    no collision), jac = MuJoCo's analytical body Jacobian (mjx.jac, from the
    motion subspace cdof), hess = exact forward-mode AD (jax.jacfwd) of that
    Jacobian — never finite differences. The model has contact detection and the
    constraint solver disabled (_disable_contacts_and_constraints). Each as *_batched (vmap over B*N configs, one jit), *_rollout (vmap over B,
    lax.scan over N mjx.step()s: includes the dynamics step) and *_batched_gpu
    (repeat-subtraction marginal GPU time per launch, overhead-free) — as
    paper_results.run_mjx. compile[key][B] is the first-call time (trace + XLA
    compile + one run), paid once; results[k][key][B] is run k's median wall."""
    import jax
    if fp == "fp64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from mujoco import mjx
    from functools import partial
    import model as _model

    paths = baseline_paths(robot, f"mjx_{fp}", runs)
    dtype = jnp.float64 if fp == "fp64" else jnp.float32
    m = mujoco.MjModel.from_xml_path(xml_path)
    _disable_contacts_and_constraints(m)
    # Output frame set = NAMED bodies excluding world, MuJoCo body order — the
    # same id list as the cusadi2 side (model.body_names). MJX still computes
    # every body; only the extraction is restricted.
    body_ids = jnp.asarray(_model.named_body_ids(m))
    m, mx, stripped = _put_model_or_strip(xml_path, mjx.put_model, m, body_ids)
    dx = mjx.put_data(m, mujoco.MjData(m))
    q0 = np.asarray(mujoco.MjData(m).qpos)
    print(f"[mjx] {robot} {fp}: nbody={m.nbody} nq={m.nq} nv={m.nv} N={n_stages} "
          f"bodies(named, excl world)={body_ids.size} runs={runs}  jax {jax.__version__} {jax.devices()}", flush=True)

    def fk_of(d):
        return jnp.concatenate([d.xpos[body_ids].reshape(-1), d.xmat[body_ids].reshape(-1)])

    def jac_of(d):
        def one(bid):
            jp, jr = mjx.jac(mx, d, d.xpos[bid], bid)          # MuJoCo's analytical body Jacobian (from cdof)
            return jnp.concatenate([jp.reshape(-1), jr.reshape(-1)])
        return jax.vmap(one)(body_ids).reshape(-1)

    def kin(q):
        return mjx.com_pos(mx, mjx.kinematics(mx, dx.replace(qpos=q)))

    def hess_of(d):                        # d/dq of the stacked body Jacobians (forward mode: nq inputs)
        return jax.jacfwd(lambda q: jac_of(kin(q)))(d.qpos).reshape(-1)

    def _same_types(new, old):
        """mjx.step widens some int fields under x64 (tendon wrap addresses, int32 ->
        int64), which lax.scan rejects as a carry-type mismatch on models with tendons
        at fp64; cast the step's output back to the carry's dtypes (no-op otherwise)."""
        return jax.tree_util.tree_map(
            lambda n, o: n.astype(o.dtype) if hasattr(n, "dtype") and hasattr(o, "dtype") and n.dtype != o.dtype else n,
            new, old)

    def make_rollout(of):
        def roll(q):
            def step(d, _):
                d = _same_types(mjx.step(mx, d), d)
                return d, of(d)
            return jax.lax.scan(step, dx.replace(qpos=q), None, length=n_stages)[1]
        return jax.jit(jax.vmap(roll))

    def make_repeat(of):
        vsingle = jax.vmap(lambda q: of(kin(q)))

        @partial(jax.jit, static_argnums=1)
        def rep(qb, R):
            def body(c, _):
                return c + vsingle(qb + c * 1e-9).sum() * 0.0 + 1.0, None   # carried dep defeats CSE
            return jax.lax.scan(body, jnp.asarray(0.0, dtype), None, length=R)[0]
        return rep

    def qbatch(rng, n):
        return jnp.asarray(rng.normal(0.0, 0.3, size=(n, m.nq)) + q0, dtype=dtype)

    def median_wall(fn, warm, n):
        for _ in range(warm):
            jax.block_until_ready(fn())
            progress.beat()
        s = np.empty(n)
        for i in range(n):
            t = time.perf_counter()
            jax.block_until_ready(fn())
            s[i] = time.perf_counter() - t
            progress.beat()
        return float(np.median(s))

    res, comp, _ = _load_runs(paths)
    res = (res + [{} for _ in range(runs)])[:runs]
    meta = {"robot": robot, "dtype": fp, "n_stages": n_stages, "n_bodies": int(body_ids.size),
            "nq": int(m.nq), "nv": int(m.nv), "batches": list(batches), "runs": runs,
            "jax": jax.__version__, "mujoco": mujoco.__version__, "actuators_stripped": stripped}
    failed = progress.take_over()

    def pending(key, B):
        return not all(str(B) in r.get(key, {}) for r in res)

    def record(key, B, values, c=None):
        for k, v in enumerate(values):
            res[k].setdefault(key, {})[str(B)] = v
        if c is not None or key not in comp or str(B) not in comp.get(key, {}):
            comp.setdefault(key, {})[str(B)] = c

    def shape_failed(key, B, shape, err):
        failed[shape] = err
        record(key, B, [None] * runs, None)
        print(f"[mjx]   {key:15s} B={B:6d}: FAILED ({err[:90]})", flush=True)

    rng = np.random.default_rng(0)
    for kind, of in (("fk", fk_of), ("jac", jac_of), ("hess", hess_of)):
        for variant, fn, per_inst in (("batched", jax.jit(jax.vmap(lambda q, of=of: of(kin(q)))), n_stages),
                                      ("rollout", make_rollout(of), 1)):
            key = f"{kind}_{variant}"
            for B in batches:
                shape = f"{key}:{B}"
                if not pending(key, B):
                    continue
                if shape in failed:
                    record(key, B, [None] * runs, None)
                    continue
                progress.start(shape)
                try:
                    qb = qbatch(rng, B * per_inst)
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(qb))                      # JIT compile + run, once
                    c = time.perf_counter() - t0
                    vals = [median_wall(lambda: fn(qb), n_warmup, n_evals) for _ in range(runs)]
                    record(key, B, vals, c)
                    print(f"[mjx]   {key:15s} B={B:6d}: " + "  ".join(f"{v * 1e6:9.1f}" for v in vals)
                          + f" us   compile {c:6.2f} s", flush=True)
                except Exception as e:                                  # VRAM OOM etc.
                    shape_failed(key, B, shape, f"{type(e).__name__}: {e}"[:300])
                progress.done()
                _flush_runs(paths, res, comp, meta, failed, False)
        gkey, rep = f"{kind}_batched_gpu", make_repeat(of)
        for B in batches:
            shape = f"{gkey}:{B}"
            if not pending(gkey, B):
                continue
            if shape in failed:
                record(gkey, B, [None] * runs, None)
                continue
            progress.start(shape)
            try:
                qb = qbatch(rng, B * n_stages)
                t0 = time.perf_counter()
                jax.block_until_ready(rep(qb, 200)); jax.block_until_ready(rep(qb, 1))     # both shapes, once
                c = time.perf_counter() - t0
                vals = []
                for _ in range(runs):
                    med = lambda R: median_wall(lambda: rep(qb, R), 4, n_evals)
                    vals.append((med(200) - med(1)) / 199)
                record(gkey, B, vals, c)
                print(f"[mjx]   {gkey:15s} B={B:6d}: " + "  ".join(f"{v * 1e6:9.2f}" for v in vals)
                      + " us/launch", flush=True)
            except Exception as e:
                shape_failed(gkey, B, shape, f"{type(e).__name__}: {e}"[:300])
            progress.done()
            _flush_runs(paths, res, comp, meta, failed, False)
    _flush_runs(paths, res, comp, meta, failed, True)
    return paths


# ---------------------------------------------------------------------------
# mujoco-warp baseline (subprocess: run.py --warp --robot R --runs K) — as
# paper_results.run_warp: FK over nworld = B*N worlds, fp32 only, no autodiff;
# kernels are shape-generic so warp compiles once (cold cache, timed), every B
# then timed K times. Same per-run files / progress / failure handling as MJX.
# ---------------------------------------------------------------------------
def run_warp(robot, xml_path, batches, n_stages, n_warmup, n_evals, runs, progress):
    """Two series over nworld = B*N worlds (matching cusadi2 / MJX-batched B*N work),
    fp32 only:
      fk   mjw.kinematics (tree kinematics, no collision)
      jac  mjw.kinematics + mjw.com_pos + mjw.jac per named body: mujoco-warp's
           analytical body Jacobian (from cdof), one launch per body over all
           worlds, into (nworld, 3, nv) jacp / jacr buffers.
    No autodiff in warp, so no Hessian. Kernels are shape-generic: warp compiles
    once per series (cold cache, timed), every B is then timed K times."""
    import numpy as np
    import warp as wp
    import mujoco
    import mujoco_warp as mjw
    import model as _model

    paths = baseline_paths(robot, "warp", runs)
    wp.init()
    try:
        wp.clear_kernel_cache()          # cold cache so the first call measures compile
    except Exception:
        pass
    m = mujoco.MjModel.from_xml_path(xml_path)
    _disable_contacts_and_constraints(m)
    body_ids = [int(b) for b in _model.named_body_ids(m)]
    print(f"[mjw] {robot}: nq={m.nq} nv={m.nv} N={n_stages} bodies={len(body_ids)} runs={runs} "
          f"devices={wp.get_devices()}", flush=True)
    m, mjw_m, stripped = _put_model_or_strip(xml_path, mjw.put_model, m, body_ids)
    rng = np.random.default_rng(0)
    q0 = np.asarray(mujoco.MjData(m).qpos)

    res, _, old_meta = _load_runs(paths)
    res = (res + [{} for _ in range(runs)])[:runs]
    meta = {"robot": robot, "backend": "mujoco_warp", "dtype": "fp32", "n_stages": n_stages,
            "nq": int(m.nq), "nv": int(m.nv), "n_bodies": len(body_ids), "batches": list(batches), "runs": runs,
            "compile_once_s": old_meta.get("compile_once_s"), "compile_jac_once_s": old_meta.get("compile_jac_once_s"),
            "warp": wp.__version__, "actuators_stripped": stripped}
    failed = progress.take_over()

    def sync():
        wp.synchronize_device("cuda:0")

    def median_wall(fn, warm, n):
        for _ in range(warm):
            fn(); sync(); progress.beat()
        s = np.empty(n)
        for i in range(n):
            t = time.perf_counter()
            fn(); sync()
            s[i] = time.perf_counter() - t
            progress.beat()
        return float(np.median(s))

    for B in batches:
        nworld = B * n_stages
        for key in ("fk", "jac"):
            shape = f"{key}:{B}"
            if all(str(B) in r.get(key, {}) for r in res):
                continue
            if shape in failed:
                for r in res:
                    r.setdefault(key, {})[str(B)] = None
                continue
            progress.start(shape)
            try:
                mjw_d = mjw.put_data(m, mujoco.MjData(m), nworld=nworld)
                qpos = (rng.normal(0.0, 0.3, size=(nworld, m.nq)) + q0).astype(np.float32)
                mjw_d.qpos = wp.from_numpy(qpos, dtype=wp.float32, device="cuda:0")
                if key == "fk":
                    def fn():
                        mjw.kinematics(mjw_m, mjw_d)
                else:
                    bodies = [wp.full(nworld, bid, dtype=int, device="cuda:0") for bid in body_ids]
                    jacp = [wp.zeros((nworld, 3, m.nv), dtype=float, device="cuda:0") for _ in body_ids]
                    jacr = [wp.zeros((nworld, 3, m.nv), dtype=float, device="cuda:0") for _ in body_ids]

                    def fn():
                        mjw.kinematics(mjw_m, mjw_d)
                        mjw.com_pos(mjw_m, mjw_d)
                        for i, bid in enumerate(body_ids):
                            mjw.jac(mjw_m, mjw_d, jacp[i], jacr[i], mjw_d.xpos[:, bid], bodies[i])
                t0 = time.perf_counter()
                fn(); sync()
                ckey = "compile_once_s" if key == "fk" else "compile_jac_once_s"
                if meta[ckey] is None:
                    meta[ckey] = time.perf_counter() - t0
                vals = [median_wall(fn, n_warmup, n_evals) for _ in range(runs)]
                for k, v in enumerate(vals):
                    res[k].setdefault(key, {})[str(B)] = v
                print(f"[mjw]   {key:3s} B={B:6d} (nworld={nworld}): " + "  ".join(f"{v * 1e6:9.1f}" for v in vals)
                      + " us", flush=True)
                del mjw_d
            except Exception as e:
                failed[shape] = f"{type(e).__name__}: {e}"[:300]
                for r in res:
                    r.setdefault(key, {})[str(B)] = None
                print(f"[mjw]   {key:3s} B={B:6d}: FAILED ({failed[shape][:90]})", flush=True)
            progress.done()
            _flush_runs(paths, res, {}, meta, failed, False)
    _flush_runs(paths, res, {}, meta, failed, True)
    return paths


# ---------------------------------------------------------------------------
# frax baseline (subprocess: run.py --frax --robot R --dtype fpXX --runs K) —
# the MJX stage's batched series for frax (frax_model.py: the gated Pinocchio
# model as a URDF of one-dof joints, frax's frame transforms and Jacobians).
# Same shapes, timing, per-run files, progress and failure handling as MJX.
# ---------------------------------------------------------------------------
FRAX_GPU_ONLY = bool(os.environ.get("CUSADI2_FRAX_GPU_ONLY"))       # also time the *_batched_gpu series


def run_frax(robot, xml_path, fp, batches, n_stages, n_warmup, n_evals, runs, progress):
    """Three quantities over the campaign's body set: fk, jac = frax's analytical
    frame Jacobian, hess = exact forward-mode AD (jax.jacfwd) of that Jacobian.
    Each as *_batched (vmap over B*N configs, one jit, inputs on the device) and,
    with CUSADI2_FRAX_GPU_ONLY, *_batched_gpu (repeat-subtraction marginal GPU
    time per launch) — as run_mjx; frax has no simulation step, so no rollout.
    compile[key][B] is the first-call time (trace + XLA compile + one run), paid
    once; results[k][key][B] is run k's median wall."""
    import jax
    if fp == "fp64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import importlib.metadata
    from functools import partial
    import frax_model

    paths = baseline_paths(robot, f"frax_{fp}", runs)
    dtype = jnp.float64 if fp == "fp64" else jnp.float32
    m, fm, frobot, frames, q0, _ = frax_model.load(xml_path, os.path.join(BUILD, robot, "frax.urdf"))
    functions = dict(zip(("fk", "jac", "hess"), frax_model.build_functions(frobot, frames)))
    print(f"[frax] {robot} {fp}: joints={fm.n} (MuJoCo nq={m.nq} nv={m.nv}) N={n_stages} bodies={len(frames)} "
          f"runs={runs}  jax {jax.__version__} {jax.devices()}", flush=True)

    def make_repeat(of):
        vsingle = jax.vmap(of)

        @partial(jax.jit, static_argnums=1)
        def rep(qb, R):
            def body(c, _):
                return c + vsingle(qb + c * 1e-9).sum() * 0.0 + 1.0, None   # carried dep defeats CSE
            return jax.lax.scan(body, jnp.asarray(0.0, dtype), None, length=R)[0]
        return rep

    def qbatch(rng, n):
        return jnp.asarray(rng.normal(0.0, 0.3, size=(n, fm.n)) + q0, dtype=dtype)

    def median_wall(fn, warm, n):
        for _ in range(warm):
            jax.block_until_ready(fn())
            progress.beat()
        s = np.empty(n)
        for i in range(n):
            t = time.perf_counter()
            jax.block_until_ready(fn())
            s[i] = time.perf_counter() - t
            progress.beat()
        return float(np.median(s))

    res, comp, _ = _load_runs(paths)
    res = (res + [{} for _ in range(runs)])[:runs]
    meta = {"robot": robot, "backend": "frax", "dtype": fp, "n_stages": n_stages, "n_bodies": len(frames),
            "nq": fm.n, "nv": fm.n, "mujoco_nq": int(m.nq), "expanded": fm.expanded,
            "pure_chain": bool(frobot.is_pure_kinematic_chain), "batches": list(batches), "runs": runs,
            "jax": jax.__version__, "frax": importlib.metadata.version("frax")}
    failed = progress.take_over()

    def pending(key, B):
        return not all(str(B) in r.get(key, {}) for r in res)

    def record(key, B, values, c=None):
        for k, v in enumerate(values):
            res[k].setdefault(key, {})[str(B)] = v
        if c is not None or key not in comp or str(B) not in comp.get(key, {}):
            comp.setdefault(key, {})[str(B)] = c

    def measure(key, B, body):
        """One shape, timed by body() -> (values per run, compile seconds), with
        the stage's skip / failure / flush bookkeeping."""
        shape = f"{key}:{B}"
        if not pending(key, B):
            return
        if shape in failed:
            return record(key, B, [None] * runs, None)
        progress.start(shape)
        try:
            vals, c = body()
            record(key, B, vals, c)
            print(f"[frax]  {key:15s} B={B:6d}: " + "  ".join(f"{v * 1e6:9.1f}" for v in vals)
                  + f" us   compile {c:6.2f} s", flush=True)
        except Exception as e:                                          # VRAM OOM etc.
            failed[shape] = f"{type(e).__name__}: {e}"[:300]
            record(key, B, [None] * runs, None)
            print(f"[frax]  {key:15s} B={B:6d}: FAILED ({failed[shape][:90]})", flush=True)
        progress.done()
        _flush_runs(paths, res, comp, meta, failed, False)

    rng = np.random.default_rng(0)
    for kind, of in functions.items():
        fn = jax.jit(jax.vmap(of))
        for B in batches:
            def batched():
                qb = qbatch(rng, B * n_stages)
                t0 = time.perf_counter()
                jax.block_until_ready(fn(qb))                          # JIT compile + run, once
                c = time.perf_counter() - t0
                return [median_wall(lambda: fn(qb), n_warmup, n_evals) for _ in range(runs)], c
            measure(f"{kind}_batched", B, batched)
        rep = make_repeat(of)
        for B in batches if FRAX_GPU_ONLY else ():
            def marginal():
                qb = qbatch(rng, B * n_stages)
                t0 = time.perf_counter()
                jax.block_until_ready(rep(qb, 200)); jax.block_until_ready(rep(qb, 1))     # both shapes, once
                c = time.perf_counter() - t0
                med = lambda R: median_wall(lambda: rep(qb, R), 4, n_evals)
                return [(med(200) - med(1)) / 199 for _ in range(runs)], c
            measure(f"{kind}_batched_gpu", B, marginal)
    _flush_runs(paths, res, comp, meta, failed, True)
    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
def _run_stage(cmd, timeout, env=None, log=None):
    """Run one stage subprocess. Returns a status record dict."""
    t0 = time.perf_counter()
    out = open(log, "w") if log else None
    try:
        p = subprocess.run(cmd, env=env, timeout=timeout, stdout=out, stderr=subprocess.STDOUT if out else None)
        rec = {"status": "pass" if p.returncode == 0 else "fail", "rc": p.returncode}
        if p.returncode != 0:
            rec["reason"] = f"exit code {p.returncode}"
    except subprocess.TimeoutExpired:
        rec = {"status": "timeout", "reason": f"timeout after {timeout}s"}
    finally:
        if out:
            out.close()
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


def _run_stage_watched(cmd, progress_path, stall_s, env, log, t_start=None):
    """_run_stage for a GPU baseline process: killed (whole process group) when
    its progress heartbeat has not moved for stall_s -> status "stalled"."""
    import signal
    t0 = time.perf_counter()
    started = time.time()
    with open(log, "a") as out:
        out.write("$ " + " ".join(cmd) + "\n"); out.flush()
        p = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                rc = p.wait(timeout=15)
                rec = {"status": "pass" if rc == 0 else "fail", "rc": rc}
                if rc != 0:
                    rec["reason"] = f"exit code {rc}"
                break
            except subprocess.TimeoutExpired:
                pass
            try:
                hb = json.load(open(progress_path)).get("heartbeat", 0)
            except Exception:
                hb = 0
            if time.time() - max(hb, started) > stall_s:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                rec = {"status": "stalled", "rc": -9, "reason": f"no progress for {stall_s:.0f}s: killed (hung on VRAM?)"}
                break
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


class Status:
    """results/<robot>/status.json: gate, stages, per-run sweeps; rewritten on every change."""
    def __init__(self, manifest, robot, head):
        self.path = os.path.join(RESULTS, robot, "status.json")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            self.d = json.load(open(self.path))
        except Exception:
            self.d = {}
        self.d.update(robot=robot, xml=manifest.ROBOTS[robot].get("xml"), menagerie_commit=head,
                      manifest_commit=getattr(manifest, "MENAGERIE_COMMIT", None))
        self.d.setdefault("gate", None); self.d.setdefault("stages", {}); self.d.setdefault("runs", {})
        self.d["complete"] = False
        self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.d, f, indent=2)

    def stage(self, name, rec):
        self.d["stages"][name] = rec
        self.save()

    def gate_ok(self):
        return (self.d.get("gate") or {}).get("status") == "pass"


def gate_phase(manifest, robots, head, force):
    """Gate every robot (cached in build/<robot>/gate.json). -> {robot: Status}"""
    self_py = [sys.executable, os.path.abspath(__file__)]
    statuses = {}
    for robot in robots:
        st = statuses[robot] = Status(manifest, robot, head)
        spec = manifest.ROBOTS[robot]
        if spec.get("skip") or not spec.get("xml"):
            reason = f"manifest: {spec.get('skip') or 'no_xml'}"
            st.d["gate"] = {"status": "skip", "reason": reason}
            st.stage("gate", {"status": "skip", "reason": reason})
            continue
        gate_json = os.path.join(BUILD, robot, "gate.json")
        os.makedirs(os.path.dirname(gate_json), exist_ok=True)
        if force or not os.path.exists(gate_json):
            rec = _run_stage(self_py + ["--gate", "--robot", robot, "--out", gate_json], STAGE_TIMEOUT_S["gate"])
        else:
            rec = {"status": "pass", "cached": True, "seconds": 0}
        if rec["status"] == "pass" and os.path.exists(gate_json):
            st.d["gate"] = json.load(open(gate_json))
            if st.d["gate"]["status"] != "pass":
                rec = {"status": "skip", "reason": st.d["gate"]["reason"], "seconds": rec["seconds"]}
        else:
            st.d["gate"] = {"status": "skip", "reason": f"gate_crashed: {rec.get('reason', 'gate subprocess died')}"}
            rec = {"status": rec["status"] if rec["status"] == "timeout" else "fail",
                   "reason": st.d["gate"]["reason"], "seconds": rec["seconds"]}
        st.stage("gate", rec)
        g = st.d["gate"]
        print(f"[gate] {robot}: {g['status']}" + (f" nq={g['nq']} nv={g['nv']} bodies={g['n_bodies']}"
                                                 if g["status"] == "pass" else f" ({g.get('reason')})"), flush=True)
    return statuses


def compile_phase(manifest, statuses, jobs, force, kind="cusadi2"):
    """All (robot, dtype) compiles. Light robots run `jobs` at a time; heavy ones
    (per-stage Hessian >= HEAVY_INSTR instructions: ptxas on the x8-unrolled
    kernel takes 10-20 GB) run afterwards strictly one at a time, and a kernel
    killed then (earlyoom) is stamped as failed: it exceeds this machine.
    kind="cpu" runs the CPU reference compiles (gcc on CasADi's C code, same
    heavy/light split: gcc -O3 on a giant Hessian is the same memory hazard)."""
    self_py = [sys.executable, os.path.abspath(__file__)]
    cpu = kind == "cpu"
    flag, stage_prefix, json_name, tool = (("--cpu-compile", "cpu_compile", "cpu_compile", "gcc") if cpu
                                           else ("--compile", "compile", "compile_cusadi2", "nvcc"))
    todo = [(robot, dtype) for robot, st in statuses.items() if st.gate_ok() for dtype in DTYPES]
    todo.sort(key=lambda rd: -statuses[rd[0]].d["gate"]["nv"])          # largest first
    if force:
        for robot, dtype in todo:
            p = os.path.join(BUILD, robot, dtype, "cpu", "stamp.json") if cpu \
                else os.path.join(BUILD, robot, dtype, "stamp.json")
            if os.path.exists(p):
                os.remove(p)

    def hess_instr(robot):          # from the DAG sweep cache when present, else a dof proxy
        try:
            return json.load(open(os.path.join(RESULTS, "dag", f"{robot}.json")))["functions"]["fkhess"]["n_instr"]
        except Exception:
            return HEAVY_INSTR if statuses[robot].d["gate"]["nv"] >= 40 else 0
    heavy = [rd for rd in todo if hess_instr(rd[0]) >= HEAVY_INSTR]
    light = [rd for rd in todo if rd not in heavy]
    print(f"=== {kind} compile phase: {len(light)} light jobs {jobs} at a time, then {len(heavy)} heavy jobs "
          f"one at a time ({sorted({r for r, _ in heavy})}) ===", flush=True)

    def job(rd):
        robot, dtype = rd
        log = os.path.join(BUILD, robot, dtype, f"{stage_prefix}_stage.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        return rd, _run_stage(self_py + [flag, "--robot", robot, "--dtype", dtype], None, log=log)

    t0 = time.perf_counter()
    killed = []

    def record(robot, dtype, rec):
        cj = os.path.join(RESULTS, robot, f"{json_name}_{dtype}.json")
        if rec["status"] == "pass" and os.path.exists(cj):
            cc = json.load(open(cj))
            rec["cached"] = cc.get("cached", False)
            if cc.get("unrolled_failed" if not cpu else "failed"):
                rec["unrolled_failed"] = cc["unrolled_failed" if not cpu else "failed"]
        elif rec.get("rc") == 3:
            rec["reason"] = f"{tool} killed (earlyoom?); retried alone"
            killed.append((robot, dtype))
        statuses[robot].stage(f"{stage_prefix}_{dtype}", rec)
        print(f"[{stage_prefix}] {robot} {dtype}: {rec['status']}"
              + (" (cached)" if rec.get("cached") else f" {rec['seconds']:.0f}s")
              + (f" {tool} failed: {list(rec['unrolled_failed'])}" if rec.get("unrolled_failed") else "")
              + (f" ({rec.get('reason')})" if rec["status"] != "pass" else ""), flush=True)

    with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
        for (robot, dtype), rec in ex.map(job, light):
            record(robot, dtype, rec)
    # heavy jobs, and light jobs whose nvcc was killed under the pool, alone:
    # a kernel killed with nothing else running exceeds the machine -> stamped
    serial = heavy + [rd for rd in killed if rd not in heavy]
    killed.clear()
    if serial:
        print(f"=== {kind} compile phase: {len(serial)} job(s) one at a time ===", flush=True)
        os.environ["CUSADI2_STAMP_KILLS"] = "1"
        for rd in serial:
            record(*rd, job(rd)[1])
        del os.environ["CUSADI2_STAMP_KILLS"]
    print(f"=== {kind} compile phase done in {time.perf_counter() - t0:.0f}s ===", flush=True)


def sweep_phase(statuses, runs, force):
    """Sequential GPU sweeps: run k over every compiled (robot, dtype)."""
    for k in range(runs):
        print(f"=== sweep run {k + 1}/{runs} ===", flush=True)
        for robot, st in statuses.items():
            for dtype in DTYPES:
                if st.d["stages"].get(f"compile_{dtype}", {}).get("status") != "pass":
                    continue
                rec = sweep_robot(robot, dtype, k, force=force)
                st.d["runs"].setdefault(str(k), {})[dtype] = rec
                st.save()
                print(f"[sweep {k}] {robot} {dtype}: {rec['status']}"
                      + (" (cached)" if rec.get("cached") else f" {rec.get('seconds', 0):.0f}s")
                      + (f" ({rec.get('reason')})" if rec["status"] != "pass" else ""), flush=True)


def cpu_run_phase(statuses, runs, force):
    """Sequential CPU sweeps (all cores; the transfer build also touches the GPU):
    run k over every CPU-compiled (robot, dtype), recorded in status "cpu_runs"."""
    for k in range(runs):
        print(f"=== CPU run {k + 1}/{runs} ===", flush=True)
        for robot, st in statuses.items():
            for dtype in DTYPES:
                if st.d["stages"].get(f"cpu_compile_{dtype}", {}).get("status") != "pass":
                    continue
                rec = cpu_run_robot(robot, dtype, k, force=force)
                st.d.setdefault("cpu_runs", {}).setdefault(str(k), {})[dtype] = rec
                st.save()
                print(f"[cpu {k}] {robot} {dtype}: {rec['status']}"
                      + (" (cached)" if rec.get("cached") else f" {rec.get('seconds', 0):.0f}s")
                      + (f" ({rec['skipped']} point(s) skipped)" if rec.get("skipped") else "")
                      + (f" ({rec.get('reason')})" if rec["status"] != "pass" else ""), flush=True)


def frax_gate(manifest, robot, force):
    """frax compatibility gate for one robot (cached in build/<robot>/frax_gate.json)."""
    gate_json = os.path.join(BUILD, robot, "frax_gate.json")
    if force or not os.path.exists(gate_json):
        _run_stage([sys.executable, os.path.abspath(__file__), "--frax-gate", "--robot", robot, "--out", gate_json],
                   STAGE_TIMEOUT_S["frax_gate"], env=dict(os.environ, JAX_PLATFORMS="cpu"),
                   log=os.path.join(BUILD, robot, "frax_gate.log"))
    try:
        return json.load(open(gate_json))
    except Exception:
        return {"status": "skip", "reason": "gate_crashed: frax gate subprocess died"}


def baseline_phase(manifest, statuses, runs, force, frax_only=False):
    """MJX (per precision), mujoco-warp and frax (per precision, behind its own
    gate) per robot, smallest first, strictly one GPU process at a time:
    results/<robot>/runs/<k>/{mjx_<fp>,warp,frax_<fp>}.json; frax_only runs the
    frax stages alone. A
    process that is killed (earlyoom) or stalls (hung on VRAM) is restarted up to
    BASELINE_ATTEMPTS times; each restart skips the shape that took the previous
    one down (recorded in the JSONs' "failed" and in status.json)."""
    self_py = [sys.executable, os.path.abspath(__file__)]
    gpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    order = sorted((r for r, st in statuses.items() if st.gate_ok()), key=lambda r: statuses[r].d["gate"]["nv"])
    for robot in order:
        st = statuses[robot]
        stages = [] if frax_only else ([(f"mjx_{fp}", ["--mjx", "--dtype", fp]) for fp in DT2FP.values()]
                                       + [("warp", ["--warp"])])
        fgate = frax_gate(manifest, robot, force)
        st.d["frax_gate"] = fgate
        if fgate["status"] == "pass":
            stages += [(f"frax_{fp}", ["--frax", "--dtype", fp]) for fp in DT2FP.values()]
        else:
            for fp in DT2FP.values():
                st.stage(f"frax_{fp}", {"status": "skip", "reason": fgate.get("reason")})
            print(f"[frax_gate] {robot}: skip ({fgate.get('reason')})", flush=True)
        for stage, args in stages:
            outs = baseline_paths(robot, stage, runs)
            progress = os.path.join(BUILD, robot, f"{stage}_progress.json")
            log = os.path.join(BUILD, robot, f"{stage}.log")
            os.makedirs(os.path.dirname(progress), exist_ok=True)
            if force:
                for p in outs + [progress]:
                    if os.path.exists(p):
                        os.remove(p)

            def complete():
                try:
                    return all(json.load(open(p)).get("complete") or json.load(open(p)).get("error") for p in outs)
                except Exception:
                    return False
            if complete():
                rec = {"status": "pass", "cached": True}
            else:
                attempts = 0
                while True:
                    attempts += 1
                    rec = _run_stage_watched(self_py + args + ["--robot", robot, "--runs", str(runs),
                                                               "--progress", progress],
                                             progress, MJX_STALL_S, gpu_env, log)
                    if rec["status"] == "pass" or attempts >= BASELINE_ATTEMPTS:
                        break
                    print(f"[{stage}] {robot}: attempt {attempts} {rec['status']} ({rec.get('reason')}); "
                          f"restarting, the shape in flight is skipped", flush=True)
                rec["attempts"] = attempts
            try:
                j = json.load(open(outs[0]))
            except Exception:
                j = {}
            if rec["status"] == "pass" and j.get("error"):
                rec = {"status": "skip", "reason": j["error"][:300], "seconds": rec.get("seconds", 0)}
            failed = j.get("failed") or {}
            if failed:
                rec["failed_shapes"] = failed
            st.stage(stage, rec)
            print(f"[{stage}] {robot}: {rec['status']}" + (" (cached)" if rec.get("cached") else f" {rec.get('seconds', 0):.0f}s")
                  + (f" {len(failed)} shape(s) failed/skipped" if failed else "")
                  + (f" ({rec.get('reason')})" if rec["status"] not in ("pass",) else ""), flush=True)


def plot_phase(statuses, runs):
    import plot
    for robot, st in statuses.items():
        if not st.gate_ok():
            st.d["complete"] = True; st.save()
            continue
        t0 = time.perf_counter()
        try:
            plot.plot_robot(robot, os.path.join(RESULTS, robot), n_stages=N_STAGES, dtypes=DTYPES)
            rec = {"status": "pass", "seconds": round(time.perf_counter() - t0, 2)}
        except Exception as e:      # noqa: BLE001  (CSVs/JSONs are the primary artifacts)
            rec = {"status": "fail", "reason": repr(e)[:300], "seconds": round(time.perf_counter() - t0, 2)}
        st.stage("plot", rec)
        st.d["n_runs"] = runs
        st.d["complete"] = True
        st.d["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st.save()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="(internal) run the compatibility gate and exit")
    ap.add_argument("--compile", action="store_true", help="(internal) codegen + nvcc for one robot x dtype")
    ap.add_argument("--cpu-compile", action="store_true", help="(internal) CasADi C codegen + gcc for one robot x dtype")
    ap.add_argument("--pylib", action="store_true", help="(internal) run the Python-called cusadi2 sweep and exit")
    ap.add_argument("--mjx", action="store_true", help="(internal) run the MJX bench and exit")
    ap.add_argument("--warp", action="store_true", help="(internal) run the mujoco-warp FK bench and exit")
    ap.add_argument("--frax", action="store_true", help="(internal) run the frax bench and exit")
    ap.add_argument("--frax-gate", action="store_true", help="(internal) run the frax compatibility gate and exit")
    ap.add_argument("--summarize", action="store_true", help="rebuild the cross-robot summary outputs and exit")
    ap.add_argument("--at-batch", type=int, help="(--summarize) fixed batch size for the speedup figure/CSV")
    ap.add_argument("--robot", help="run only this manifest robot")
    ap.add_argument("--dtype", help="cusadi dtype (double/float) or MJX precision (fp32/fp64)")
    ap.add_argument("--lib", help="(--pylib) path to the shared latency library")
    ap.add_argument("--manifest", help="path to manifest.py (default: alongside run.py)")
    ap.add_argument("--runs", type=int, default=1, help="number of sweep runs (compiled once)")
    ap.add_argument("--jobs", type=int, default=N_JOBS, help="parallel compile subprocesses")
    ap.add_argument("--force", action="store_true", help="redo gate, compile and sweeps even if cached")
    ap.add_argument("--no-baselines", action="store_true", help="skip the MJX and mujoco-warp stages (cusadi2 only)")
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU reference (CasADi C codegen) compile + sweeps")
    ap.add_argument("--cpu-only", action="store_true",
                    help="only the CPU reference phases (gate, gcc compile, CPU sweeps), then plots + summary")
    ap.add_argument("--compile-only", action="store_true", help="stop after the compile phase")
    ap.add_argument("--out")
    ap.add_argument("--progress", help="(--mjx/--warp/--frax) progress/heartbeat json")
    ap.add_argument("--frax-only", action="store_true",
                    help="only the frax baseline (gates, frax stages); no plots or summary")
    ap.add_argument("--baselines-only", action="store_true",
                    help="only the MJX / mujoco-warp phase (gate, baselines), then plots + summary")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)

    # --- internal subprocess modes ------------------------------------------
    if args.gate:
        # The gate must NEVER exit without writing its JSON: run_gate catches
        # the expected failure modes itself, and anything it misses (parser
        # edge cases, bad manifest entries, ...) is recorded here as a skip
        # with a traceback — the orchestrator treats it like any other gate
        # skip instead of a dead subprocess.
        try:
            import model
            out = model.run_gate(robot_xml(manifest, args.robot))
        except Exception as e:
            import traceback
            out = {"status": "skip",
                   "reason": f"gate_error: {type(e).__name__}: {e}"[:500],
                   "traceback": traceback.format_exc()[-4000:]}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[gate] {args.robot}: {out['status']}"
              + (f" ({out.get('reason')})" if out.get("reason") else ""))
        return 0
    if args.frax_gate:
        try:
            import frax_model
            out = frax_model.run_gate(robot_xml(manifest, args.robot), os.path.join(BUILD, args.robot, "frax.urdf"))
        except Exception as e:
            out = {"status": "skip", "reason": f"gate_error: {type(e).__name__}: {e}"[:500]}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[frax_gate] {args.robot}: {out['status']}" + (f" ({out.get('reason')})" if out.get("reason") else ""))
        return 0
    if args.compile:
        return compile_robot(manifest, args.robot, args.dtype)
    if args.cpu_compile:
        return cpu_compile_robot(manifest, args.robot, args.dtype)
    if args.pylib:
        run_pylib(args.lib, args.dtype, BATCHES, N_STAGES, N_WARMUP, N_TIMED, args.out)
        return 0
    if args.mjx or args.warp or args.frax:
        xml = robot_xml(manifest, args.robot)
        tag = f"mjx_{args.dtype}" if args.mjx else f"frax_{args.dtype}" if args.frax else "warp"
        name = "mjx" if args.mjx else "frax" if args.frax else "warp"
        progress = _Progress(args.progress or os.path.join(BUILD, args.robot, f"{tag}_progress.json"))
        try:
            if args.mjx or args.frax:
                paths = (run_mjx if args.mjx else run_frax)(args.robot, xml, args.dtype, MJX_BATCHES, N_STAGES,
                                                            MJX_N_WARMUP, MJX_N_EVALS, args.runs, progress)
            else:
                paths = run_warp(args.robot, xml, MJX_BATCHES, N_STAGES,
                                 MJX_N_WARMUP, MJX_N_EVALS, args.runs, progress)
        except Exception as e:      # e.g. mjx.put_model rejects the model outright
            err = f"{name}_unavailable: {e!r}"[:1000]
            print(f"[{'mjw' if name == 'warp' else name}] {args.robot}: {err}")
            paths = baseline_paths(args.robot, tag, args.runs)
            for p in paths:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    json.dump({"robot": args.robot, "error": err}, f, indent=2)
        progress.done()
        print(f"wrote {paths[0]} .. {os.path.basename(os.path.dirname(paths[-1]))}/")
        return 0

    # --- orchestrator --------------------------------------------------------
    head = check_menagerie(manifest)
    os.makedirs(RESULTS, exist_ok=True)
    if args.summarize:
        import summarize
        summarize.summarize(RESULTS, manifest, n_stages=N_STAGES, at_batch=args.at_batch)
        return 0

    robots = [args.robot] if args.robot else list(manifest.ROBOTS)
    for r in robots:
        if r not in manifest.ROBOTS:
            sys.exit(f"unknown robot {r!r} (not in manifest)")
    print(f"menagerie: {MENAGERIE} @ {head}\nrobots: {len(robots)}   GPU {DEVICE}   N={N_STAGES}   "
          f"batches={BATCHES}   runs={args.runs}   compile jobs={args.jobs}", flush=True)
    statuses = gate_phase(manifest, robots, head, args.force)
    if args.frax_only:
        baseline_phase(manifest, statuses, args.runs, args.force, frax_only=True)
        return 0
    cusadi2_phases = not (args.cpu_only or args.baselines_only)
    cpu_phases = not (args.no_cpu or args.baselines_only)
    if cusadi2_phases:
        compile_phase(manifest, statuses, args.jobs, args.force)
    if cpu_phases:
        compile_phase(manifest, statuses, args.jobs, args.force, kind="cpu")
    if args.compile_only:
        return 0
    if cusadi2_phases:
        sweep_phase(statuses, args.runs, args.force)
    if cpu_phases:
        cpu_run_phase(statuses, args.runs, args.force)
    if args.cpu_only:
        pass                                   # baseline stages left as recorded
    elif not args.no_baselines:
        baseline_phase(manifest, statuses, args.runs, args.force)
    else:
        for st in statuses.values():
            for stage in ("mjx_fp64", "mjx_fp32", "warp", "frax_fp64", "frax_fp32"):
                st.stage(stage, {"status": "skip", "reason": "--no-baselines"})
    plot_phase(statuses, args.runs)
    import summarize
    summarize.summarize(RESULTS, manifest, n_stages=N_STAGES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
