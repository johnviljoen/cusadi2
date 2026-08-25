#!/usr/bin/env python
"""Per-robot MuJoCo Menagerie campaign of the paper_results latency benchmark.

Runs the EXACT benchmark matrix of examples/paper_results — a latency-vs-batch
sweep of 4 kernel configurations (full/staged/lp/batch) x 2 dtypes (double/float),
timed from C++ AND from Python via ctypes, plus MuJoCo MJX baselines (fp32/fp64,
batched/rollout/marginal-GPU variants) and a mujoco-warp FK baseline with
compile-time tracking — once per robot, for every robot listed in manifest.py,
against the read-only Menagerie checkout (default ../mujoco_menagerie).

Differences from paper_results (semantics otherwise preserved verbatim):

  * BOTH backends load the same Menagerie MJCF XML. The Pinocchio side uses
    pin.buildModelFromMJCF (not robot_descriptions/URDF); base handling comes
    from the MJCF itself (freejoint present -> floating, absent -> fixed).
  * Two functions per robot (fk, fkjac) instead of four hardcoded ones;
    latency.cu uses generic ROBOT_FK / ROBOT_FKJAC plan prefixes and each robot
    gets its own build dir + binary.
  * A per-robot compatibility gate runs before any benchmarking (see model.py);
    every failure is a machine-recorded skip reason in status.json, never a
    crash of the campaign.
  * Orchestration is sequential per robot; each stage runs as a subprocess with
    a generous timeout, writing results/<robot>/. A robot whose status.json says
    "complete" is skipped on restart (resumability); --force re-runs it.

Usage:

    python examples/menagerie_bench/run.py                # full campaign
    python examples/menagerie_bench/run.py --robot unitree_a1
    python examples/menagerie_bench/run.py --robot unitree_a1 --force
    python examples/menagerie_bench/run.py --summarize    # cross-robot outputs only

Env overrides: CUSADI2_GPU (device index, default "5"), NVCC, CUSADI2_MENAGERIE
(menagerie root, default ../mujoco_menagerie relative to this dir),
CUSADI2_MANIFEST (manifest path). Smoke-test knobs (propagate to subprocesses):
CUSADI2_BATCHES (comma list), CUSADI2_N_WARMUP, CUSADI2_N_TIMED, CUSADI2_RESULTS,
CUSADI2_BUILD.
"""
import argparse
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

DEVICE = os.environ.get("CUSADI2_GPU", "5")
NVCC = os.environ.get("NVCC", "nvcc")
MENAGERIE = os.path.abspath(os.environ.get("CUSADI2_MENAGERIE",
                                           os.path.join(HERE, "..", "mujoco_menagerie")))

# Stage-axis horizon (fixed) and batch sweep (the x-axis) — identical to
# paper_results. N=8 keeps every unrolled plan within the per-block shared-memory
# limit for a1/h1; robots whose unrolled plans overflow at fp64 (e.g. pal_talos,
# robot_soccer_kit, flybody) get nan lp/batch rows via latency.cu's guards.
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

# (CSV/header stem, C plan prefix). Must match latency.cu.
FUNC_STEMS = [("fk", "ROBOT_FK"), ("fkjac", "ROBOT_FKJAC")]

# Per-stage subprocess timeouts (seconds); expiry is recorded as status "timeout".
STAGE_TIMEOUT_S = {"gate": 900, "cusadi": 7200, "pylib": 7200,
                   "mjx": 7200, "warp": 7200}


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


# ---------------------------------------------------------------------------
# cusadi2 stage (subprocess: run.py --cusadi --robot R --dtype D)
# codegen + nvcc + C++-hosted sweep for one robot x dtype.
# ---------------------------------------------------------------------------
def gpu_arch() -> str:
    cc = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", DEVICE]
    ).decode().splitlines()[0].strip()
    return "sm_" + cc.replace(".", "")


def run_cusadi(manifest, robot, dtype) -> int:
    import model
    from cusadi2.kernel_template import write_kernel_source
    from cusadi2.generate import generate_native_plan
    from cusadi2.dag import extract_level_plan
    import cusadi2
    pkg = cusadi2.__path__[0]                 # installed package dir (holds cusadi2.h)

    xml = robot_xml(manifest, robot)
    m = model.load_mujoco(xml)
    pmodel = model.load_pinocchio(xml, m=m)   # SAME repaired load path as the gate
    names = model.body_names(m)
    # constant world-frame root correction (see model.root_correction); the gate
    # has already verified the corrected functions match MuJoCo numerically
    corr = model.root_correction(m, pmodel, model.qpos_mj_to_pin_index(m, pmodel))
    builders = {"fk": model.build_fk, "fkjac": model.build_fkjac}

    outdir = os.path.join(BUILD, robot, dtype)
    resdir = os.path.join(RESULTS, robot)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(resdir, exist_ok=True)

    write_kernel_source(os.path.join(outdir, "kernel.cu"), dtype=dtype, tiled=False)
    print(f"=== [{robot}] building {dtype} plans (N={N_STAGES}) ===")
    # cusadi2 compile = per-function codegen (extract plan + emit header) plus
    # the one-time, function-independent nvcc kernel compile (shared). Both are
    # batch-independent — the whole point vs MJX's per-shape JIT.
    codegen_s, meta = {}, {}
    bytes_per = 4 if dtype == "float" else 8
    for stem, prefix in FUNC_STEMS:
        t0 = time.perf_counter()
        fn = builders[stem](pmodel, names, name=stem, root_corr=corr)
        generate_native_plan(fn, os.path.join(outdir, f"{stem}_plan.h"),
                             prefix=prefix, dtype=dtype)
        unr = model.unroll(fn, N_STAGES)
        generate_native_plan(unr, os.path.join(outdir, f"{stem}_unr_plan.h"),
                             prefix=prefix + "_UNR", dtype=dtype)
        codegen_s[stem] = time.perf_counter() - t0
        ps = extract_level_plan(fn)
        pu = extract_level_plan(unr)
        meta[stem] = {
            "nnz_out": int(fn.nnz_out(0)),
            "n_instr": int(fn.n_instructions()),
            "n_instr_unr": int(unr.n_instructions()),
            "smem_kb": ps.peak_live_slots * bytes_per / 1024,
            "smem_unr_kb": pu.peak_live_slots * bytes_per / 1024,
        }
        print(f"  {stem}: nout={fn.nnz_out(0)} per-stage instr={fn.n_instructions()} "
              f"smem={meta[stem]['smem_kb']:.1f}KB | "
              f"unrolled instr={unr.n_instructions()} "
              f"smem={meta[stem]['smem_unr_kb']:.1f}KB  codegen={codegen_s[stem]:.2f}s")

    arch = gpu_arch()
    binpath = os.path.join(outdir, "latency")
    cmd = [NVCC, "-std=c++14", "-O3", f"-arch={arch}",
           f"-DCUSADI2_SCALAR={dtype}", f"-I{pkg}", f"-I{outdir}",
           os.path.join(HERE, "latency.cu"), os.path.join(outdir, "kernel.cu"),
           "-o", binpath]
    print("nvcc:", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    nvcc_s = time.perf_counter() - t0
    with open(os.path.join(resdir, f"compile_cusadi2_{dtype}.json"), "w") as f:
        json.dump({"robot": robot, "dtype": dtype, "codegen_s": codegen_s,
                   "nvcc_s": nvcc_s, "nq": int(m.nq), "nv": int(m.nv),
                   "n_bodies": len(names), "meta": meta}, f, indent=2)
    print(f"  nvcc compile: {nvcc_s:.1f}s (function-independent, shared across all funcs/batches)")

    csv = os.path.join(resdir, f"latency_{dtype}.csv")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE)
    cmd = [binpath, str(N_STAGES), str(N_WARMUP), str(N_TIMED)] + [str(b) for b in BATCHES]
    print(f"=== [{robot}] running {dtype} (N={N_STAGES}, batches={BATCHES}) ===")
    with open(csv, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f).returncode
    print(f"  wrote {csv}")

    # Shared library build for the Python-called (ctypes) pass — the pylib stage
    # subprocess loads it (-DCUSADI2_BENCH_LIB drops main).
    libpath = os.path.join(outdir, "latency_lib.so")
    cmd = [NVCC, "-std=c++14", "-O3", f"-arch={arch}", "-shared",
           "-Xcompiler", "-fPIC", "-DCUSADI2_BENCH_LIB",
           f"-DCUSADI2_SCALAR={dtype}", f"-I{pkg}", f"-I{outdir}",
           os.path.join(HERE, "latency.cu"), os.path.join(outdir, "kernel.cu"),
           "-o", libpath]
    subprocess.run(cmd, check=True)
    return rc


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
    lib.c2_setup.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.c2_eval.restype = ctypes.c_double
    lib.c2_eval.argtypes = [ctypes.c_int]
    lib.c2_teardown.argtypes = [ctypes.c_int]

    dt = {"fp32": "f32", "fp64": "f64"}[fp]
    configs = [("full", n_stages), ("staged", n_stages), ("lp", 1), ("batch", 1)]
    rows = ["dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us"]
    for stem, _ in FUNC_STEMS:
        for B in batches:
            for cfg, nst in configs:
                h = lib.c2_setup(stem.encode(), cfg.encode(), B, n_stages)
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
# MJX baseline (subprocess: run.py --mjx --robot R --dtype fpXX) — identical to
# paper_results.run_mjx apart from loading the Menagerie MJCF directly.
# ---------------------------------------------------------------------------
def run_mjx(robot, xml_path, dtype_str, batches, n_stages, n_warmup, n_evals):
    """Two MJX curves per quantity, both doing B*N total FK/Jacobian evaluations
    at plotted batch B (matching cusadi2's stage axis):

      *_batched  — jax.vmap over B*N independent per-knot configs, one jit launch.
                   The horizon evaluated fully in parallel (cusadi2's stage axis).
      *_rollout  — jax.vmap over B instances, jax.lax.scan over N sequential
                   mjx.step()s, computing the quantity at each knot. The realistic
                   simulate-a-trajectory usage: sequential in time, so ~N launches'
                   worth of dispatch. (Includes the dynamics step that evolves state
                   — and, on models with equality constraints/tendons, the
                   constraint-solver work; contacts off.)
    """
    import jax
    if dtype_str == "fp64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from mujoco import mjx
    import mujoco.mjx._src.support as support

    dtype = jnp.float64 if dtype_str == "fp64" else jnp.float32
    m = mujoco.MjModel.from_xml_path(xml_path)
    m.geom_contype[:] = 0           # contacts OFF
    m.geom_conaffinity[:] = 0

    # Common OUTPUT frame set = NAMED bodies excluding world, MuJoCo body order
    # — the exact same id list (hence output ordering) as the cusadi2 side's
    # model.body_names frame list. MJX still computes ALL bodies internally;
    # only the extraction below is restricted (load-bearing for barkour_vb /
    # umi_gripper, which carry unnamed bodies).
    import model as _model
    body_ids = jnp.asarray(_model.named_body_ids(m))
    print(f"[mjx] {robot} {dtype_str}: nbody={m.nbody} nq={m.nq} nv={m.nv} N={n_stages} "
          f"bodies(named, excl world)={body_ids.size}  jax {jax.__version__} {jax.devices()}")

    mx = mjx.put_model(m)
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)
    q0 = np.asarray(d.qpos)

    # quantity extracted from an already-forwarded Data (xpos/xmat/cdof populated)
    def fk_of(data):                           # world placement of every named body
        return jnp.concatenate([data.xpos[body_ids].reshape(-1),
                                data.xmat[body_ids].reshape(-1)])

    def jac_of(data):                          # 6xnv world-aligned Jacobian / body
        def one(bid):
            jp, jr = support.jac(mx, data, data.xpos[bid], bid)
            return jnp.concatenate([jp.reshape(-1), jr.reshape(-1)])
        return jax.vmap(one)(body_ids).reshape(-1)

    QTY = {"fk": fk_of, "jac": jac_of}

    # batched: evaluate one given config (parallel over the B*N axis by vmap)
    def make_batched(of):
        def single(q):
            data = mjx.com_pos(mx, mjx.kinematics(mx, dx.replace(qpos=q)))
            return of(data)
        return jax.jit(jax.vmap(single))

    # rollout: per instance, N sequential mjx.step()s, quantity at each knot
    def make_rollout(of):
        def roll(q):
            def step(data, _):
                data = mjx.step(mx, data)
                return data, of(data)
            _, outs = jax.lax.scan(step, dx.replace(qpos=q), None, length=n_stages)
            return outs
        return jax.jit(jax.vmap(roll))

    def time_fn(fn, qb):
        """Returns (median_latency_s, compile_s). compile_s is the first-call
        time (trace + XLA compile + one run) — the standard MJX compile proxy."""
        t0 = time.perf_counter()
        jax.block_until_ready(fn(qb))                         # JIT compile + run
        compile_s = time.perf_counter() - t0
        for _ in range(n_warmup):
            jax.block_until_ready(fn(qb))
        s = np.empty(n_evals)
        for i in range(n_evals):
            t1 = time.perf_counter()
            jax.block_until_ready(fn(qb))
            s[i] = time.perf_counter() - t1
        return float(np.median(s)), float(compile_s)

    # Overhead-free marginal GPU cost via repeat-subtraction: run R copies of the
    # batched op fused in ONE jit (a lax.scan with a tiny loop-carried dependency so
    # XLA can't CSE/DCE the repeats), so the fixed Python-dispatch + XLA program-launch
    # overhead is paid once. marginal = (T(R) - T(1)) / (R-1) is the pure GPU time per
    # launch — the throughput number, stripped of the launch overhead that dominates
    # the wall-clock curves at these batch sizes.
    from functools import partial

    def make_repeat(of):
        vsingle = jax.vmap(lambda q: of(mjx.com_pos(mx, mjx.kinematics(mx, dx.replace(qpos=q)))))

        @partial(jax.jit, static_argnums=1)
        def rep(qb, R):
            def body(c, _):
                out = vsingle(qb + c * 1e-9)          # carried dep defeats CSE/DCE
                return c + out.sum() * 0.0 + 1.0, None
            c, _ = jax.lax.scan(body, jnp.asarray(0.0, dtype), None, length=R)
            return c
        return rep

    def marginal_s(rep, qb, R=200):
        def med(Rr):
            jax.block_until_ready(rep(qb, Rr))
            for _ in range(3):
                jax.block_until_ready(rep(qb, Rr))
            s = np.empty(n_evals)
            for i in range(n_evals):
                t = time.perf_counter()
                jax.block_until_ready(rep(qb, Rr))
                s[i] = time.perf_counter() - t
            return float(np.median(s))
        return (med(R) - med(1)) / (R - 1)

    rng = np.random.default_rng(0)
    results, compile_t = {}, {}
    for kind, of in QTY.items():
        fb, fr = make_batched(of), make_rollout(of)
        for variant, fn, per_inst in [("batched", fb, n_stages), ("rollout", fr, 1)]:
            key = f"{kind}_{variant}"
            results[key], compile_t[key] = {}, {}
            for B in batches:
                qb = jnp.asarray(rng.normal(0.0, 0.3, size=(B * per_inst, m.nq)) + q0,
                                 dtype=dtype)
                try:
                    med, comp = time_fn(fn, qb)
                    results[key][str(B)], compile_t[key][str(B)] = med, comp
                    print(f"[mjx]   {key:11s} B={B:6d}: {med*1e6:10.1f} us   "
                          f"compile {comp:6.2f} s")
                except Exception as e:  # OOM etc. — record null, keep going
                    results[key][str(B)] = compile_t[key][str(B)] = None
                    print(f"[mjx]   {key:11s} B={B:6d}: FAILED ({repr(e)[:70]})")

        # overhead-free marginal GPU throughput for the batched op (B*N work, matching
        # the *_batched wall curve) — the like-for-like throughput vs cusadi2 kernel-only
        gkey = f"{kind}_batched_gpu"
        rp = make_repeat(of)
        results[gkey] = {}
        for B in batches:
            qb = jnp.asarray(rng.normal(0.0, 0.3, size=(B * n_stages, m.nq)) + q0, dtype=dtype)
            try:
                g = marginal_s(rp, qb)
                results[gkey][str(B)] = g
                print(f"[mjx]   {gkey:15s} B={B:6d}: {g*1e6:10.2f} us/launch "
                      f"({g/(B*n_stages)*1e9:6.2f} ns/eval)")
            except Exception as e:  # OOM etc. — record null, keep going
                results[gkey][str(B)] = None
                print(f"[mjx]   {gkey:15s} B={B:6d}: FAILED ({repr(e)[:60]})")

    return {"robot": robot, "dtype": dtype_str, "n_stages": n_stages,
            "n_bodies": int(body_ids.size), "nq": int(m.nq), "nv": int(m.nv),
            "batches": list(batches), "results": results, "compile": compile_t}


# ---------------------------------------------------------------------------
# mujoco-warp baseline (subprocess: run.py --warp --robot R) — identical to
# paper_results.run_warp apart from loading the Menagerie MJCF directly.
# ---------------------------------------------------------------------------
def run_warp(robot, xml_path, batches, n_stages, n_warmup, n_evals):
    """mujoco-warp forward kinematics over nworld = B*N worlds (matches cusadi2 /
    MJX-batched B*N work). fp32 only, and no autodiff — so FK only, no Jacobian."""
    import numpy as np
    import warp as wp
    import mujoco
    import mujoco_warp as mjw

    wp.init()
    try:                            # cold cache so the first call measures compile
        wp.clear_kernel_cache()
    except Exception:
        pass
    m = mujoco.MjModel.from_xml_path(xml_path)
    m.geom_contype[:] = 0           # contacts OFF
    m.geom_conaffinity[:] = 0
    print(f"[mjw] {robot}: nq={m.nq} nv={m.nv} N={n_stages} devices={wp.get_devices()}")
    mjw_m = mjw.put_model(m)
    rng = np.random.default_rng(0)
    q0 = np.asarray(mujoco.MjData(m).qpos)
    results = {"fk": {}}
    compile_once = None             # warp compiles once (kernels are shape-generic)
    for B in batches:
        nworld = B * n_stages
        try:
            mjw_d = mjw.put_data(m, mujoco.MjData(m), nworld=nworld)
            qpos = (rng.normal(0.0, 0.3, size=(nworld, m.nq)) + q0).astype(np.float32)
            mjw_d.qpos = wp.from_numpy(qpos, dtype=wp.float32, device="cuda:0")
            t0 = time.perf_counter()               # first call -> compile (cold cache)
            mjw.kinematics(mjw_m, mjw_d)
            wp.synchronize_device("cuda:0")
            first = time.perf_counter() - t0
            if compile_once is None:
                compile_once = float(first)
            for _ in range(n_warmup):
                mjw.kinematics(mjw_m, mjw_d)
            wp.synchronize_device("cuda:0")
            s = np.empty(n_evals)
            for i in range(n_evals):
                t1 = time.perf_counter()
                mjw.kinematics(mjw_m, mjw_d)
                wp.synchronize_device("cuda:0")
                s[i] = time.perf_counter() - t1
            results["fk"][str(B)] = float(np.median(s))
            print(f"[mjw]   fk B={B:6d} (nworld={nworld}): {np.median(s)*1e6:9.1f} us")
        except Exception as e:
            results["fk"][str(B)] = None
            print(f"[mjw]   fk B={B:6d}: FAILED ({repr(e)[:70]})")
    return {"robot": robot, "backend": "mujoco_warp", "dtype": "fp32",
            "n_stages": n_stages, "nq": int(m.nq), "nv": int(m.nv),
            "batches": list(batches), "results": results,
            "compile_once_s": compile_once}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _run_stage(cmd, timeout, env=None):
    """Run one stage subprocess. Returns a status record dict."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, env=env, timeout=timeout)
        rec = {"status": "pass" if p.returncode == 0 else "fail",
               "rc": p.returncode}
        if p.returncode != 0:
            rec["reason"] = f"exit code {p.returncode}"
    except subprocess.TimeoutExpired:
        rec = {"status": "timeout", "reason": f"timeout after {timeout}s"}
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


def process_robot(manifest, robot, head, force=False) -> None:
    resdir = os.path.join(RESULTS, robot)
    status_path = os.path.join(resdir, "status.json")
    if not force and os.path.exists(status_path):
        try:
            if json.load(open(status_path)).get("complete"):
                print(f"--- {robot}: already complete, skipping (use --force to re-run)")
                return
        except Exception:
            pass

    spec = manifest.ROBOTS[robot]
    os.makedirs(resdir, exist_ok=True)
    entry = {"robot": robot, "xml": spec.get("xml"),
             "menagerie_commit": head,
             "manifest_commit": getattr(manifest, "MENAGERIE_COMMIT", None),
             "gate": None, "stages": {}, "complete": False,
             "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save(complete=False):
        entry["complete"] = complete
        if complete:
            entry["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(status_path, "w") as f:
            json.dump(entry, f, indent=2)

    def finish():
        save(complete=True)
        print(f"--- {robot}: done — " + " ".join(
            f"{k}={v['status']}" for k, v in entry["stages"].items()))

    print(f"\n===== {robot} =====")

    # manifest-level skip (or no xml)
    if spec.get("skip") or not spec.get("xml"):
        reason = spec.get("skip") or "no_xml"
        entry["gate"] = {"status": "skip", "reason": f"manifest: {reason}"}
        entry["stages"]["gate"] = {"status": "skip", "reason": f"manifest: {reason}"}
        finish()
        return

    self_py = [sys.executable, os.path.abspath(__file__)]
    gpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE,
                   XLA_PYTHON_CLIENT_PREALLOCATE="false")

    # 1. compatibility gate (its own subprocess: parser crashes must not kill us)
    gate_json = os.path.join(BUILD, robot, "gate.json")
    os.makedirs(os.path.dirname(gate_json), exist_ok=True)
    rec = _run_stage(self_py + ["--gate", "--robot", robot, "--out", gate_json],
                     STAGE_TIMEOUT_S["gate"])
    if rec["status"] == "pass" and os.path.exists(gate_json):
        entry["gate"] = json.load(open(gate_json))
        if entry["gate"]["status"] != "pass":
            rec = {"status": "skip", "reason": entry["gate"]["reason"],
                   "seconds": rec["seconds"]}
    else:
        reason = rec.get("reason", "gate subprocess died")
        entry["gate"] = {"status": "skip", "reason": f"gate_crashed: {reason}"}
        rec = {"status": rec["status"] if rec["status"] == "timeout" else "fail",
               "reason": entry["gate"]["reason"], "seconds": rec["seconds"]}
    entry["stages"]["gate"] = rec
    save()
    if entry["gate"]["status"] != "pass":
        print(f"--- {robot}: gate {rec['status']} ({rec.get('reason')})")
        finish()
        return
    print(f"--- {robot}: gate pass (nq={entry['gate']['nq']} nv={entry['gate']['nv']} "
          f"bodies={entry['gate']['n_bodies']} floating={entry['gate']['floating']} "
          f"fk_err={entry['gate']['fk_max_err']:.2e})")

    # 2. cusadi2 sweeps: codegen + nvcc + C++ sweep, then the Python-called twin
    for dtype in DTYPES:
        rec = _run_stage(self_py + ["--cusadi", "--robot", robot, "--dtype", dtype],
                         STAGE_TIMEOUT_S["cusadi"])
        entry["stages"][f"cusadi_{dtype}"] = rec
        save()
        pykey = f"pylib_{dtype}"
        if rec["status"] != "pass":
            entry["stages"][pykey] = {"status": "skip",
                                      "reason": f"cusadi_{dtype} {rec['status']}"}
            save()
            continue
        libpath = os.path.join(BUILD, robot, dtype, "latency_lib.so")
        pycsv = os.path.join(resdir, f"latency_py_{dtype}.csv")
        rec = _run_stage(self_py + ["--pylib", "--dtype", DT2FP[dtype],
                                    "--lib", libpath, "--out", pycsv],
                         STAGE_TIMEOUT_S["pylib"], env=gpu_env)
        entry["stages"][pykey] = rec
        save()

    # 3. MJX baseline, one subprocess per dtype. put_model failures are recorded
    #    by the subprocess as {"error": "mjx_unavailable: ..."} -> stage "skip".
    for fp in DT2FP.values():
        out = os.path.join(resdir, f"mjx_{fp}.json")
        rec = _run_stage(self_py + ["--mjx", "--robot", robot, "--dtype", fp,
                                    "--out", out],
                         STAGE_TIMEOUT_S["mjx"], env=gpu_env)
        if rec["status"] == "pass" and os.path.exists(out):
            err = json.load(open(out)).get("error")
            if err:
                rec = {"status": "skip", "reason": err[:300], "seconds": rec["seconds"]}
        entry["stages"][f"mjx_{fp}"] = rec
        save()

    # 4. mujoco-warp baseline (FK only, fp32; no autodiff/Jacobian)
    out = os.path.join(resdir, "warp.json")
    rec = _run_stage(self_py + ["--warp", "--robot", robot, "--out", out],
                     STAGE_TIMEOUT_S["warp"], env=gpu_env)
    if rec["status"] == "pass" and os.path.exists(out):
        err = json.load(open(out)).get("error")
        if err:
            rec = {"status": "skip", "reason": err[:300], "seconds": rec["seconds"]}
    entry["stages"]["warp"] = rec
    save()

    # 5. per-robot figures (best-effort; CSVs/JSONs are the primary artifacts)
    t0 = time.perf_counter()
    try:
        import plot
        plot.plot_robot(robot, resdir, n_stages=N_STAGES, dtypes=DTYPES)
        entry["stages"]["plot"] = {"status": "pass",
                                   "seconds": round(time.perf_counter() - t0, 2)}
    except Exception as e:
        entry["stages"]["plot"] = {"status": "fail", "reason": repr(e)[:300],
                                   "seconds": round(time.perf_counter() - t0, 2)}
    finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="(internal) run the compatibility gate and exit")
    ap.add_argument("--cusadi", action="store_true", help="(internal) codegen+compile+run the C++ sweep for one robot x dtype")
    ap.add_argument("--mjx", action="store_true", help="(internal) run the MJX bench and exit")
    ap.add_argument("--warp", action="store_true", help="(internal) run the mujoco-warp FK bench and exit")
    ap.add_argument("--pylib", action="store_true", help="(internal) run the Python-called cusadi2 bench and exit")
    ap.add_argument("--summarize", action="store_true", help="rebuild the cross-robot summary outputs and exit")
    ap.add_argument("--at-batch", type=int, help="(--summarize) fixed batch size for the speedup figure/CSV (default: per-robot saturation batch); writes summary*_B<N> variants")
    ap.add_argument("--robot", help="run only this manifest robot")
    ap.add_argument("--dtype", help="cusadi dtype (double/float) or MJX precision (fp32/fp64)")
    ap.add_argument("--lib", help="(--pylib) path to the shared latency library")
    ap.add_argument("--manifest", help="path to manifest.py (default: alongside run.py)")
    ap.add_argument("--force", action="store_true", help="re-run robots even if their status.json is complete")
    ap.add_argument("--out")
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
    if args.cusadi:
        return run_cusadi(manifest, args.robot, args.dtype)
    if args.pylib:
        run_pylib(args.lib, args.dtype, BATCHES, N_STAGES, N_WARMUP, N_TIMED, args.out)
        return 0
    if args.mjx or args.warp:
        xml = robot_xml(manifest, args.robot)
        try:
            if args.mjx:
                out = run_mjx(args.robot, xml, args.dtype, MJX_BATCHES, N_STAGES,
                              MJX_N_WARMUP, MJX_N_EVALS)
            else:
                out = run_warp(args.robot, xml, MJX_BATCHES, N_STAGES,
                               MJX_N_WARMUP, MJX_N_EVALS)
        except Exception as e:      # e.g. mjx.put_model rejects the model outright
            tag = "mjx_unavailable" if args.mjx else "warp_unavailable"
            out = {"robot": args.robot, "error": f"{tag}: {e!r}"[:1000]}
            print(f"[{'mjx' if args.mjx else 'mjw'}] {args.robot}: {out['error']}")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
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
    print(f"menagerie: {MENAGERIE} @ {head}\n"
          f"robots: {len(robots)}   GPU {DEVICE}   N={N_STAGES}   batches={BATCHES}")
    for robot in robots:
        try:
            process_robot(manifest, robot, head, force=args.force)
        except Exception as e:      # never let one robot kill the campaign
            print(f"!!! {robot}: orchestrator error {e!r}", file=sys.stderr)

    import summarize
    summarize.summarize(RESULTS, manifest, n_stages=N_STAGES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
