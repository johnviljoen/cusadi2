#!/usr/bin/env python
"""Latency-vs-batch saturation benchmark for cusadi2's parallelism axes, with a
MuJoCo MJX baseline overlaid.

Replicates the level-parallel ablation graphs from the old cusadi codebase, but
entirely within cusadi2. For A1 and H1 forward kinematics and their frame
Jacobians (free-floating base, no contacts), it sweeps batch size and times four
configurations that toggle the stage and level-parallel axes on/off:

    full      staged + level-parallel + batch    grid=(B,N),  block=pick_block_size
    staged    staged + batch (no LP)             grid=(B,N),  block=1
    lp        level-parallel + batch (no stages) grid=(B,1),  block=pick_block_size
    batch     batch only (~ original cusadi)     grid=(B,1),  block=1

The full/staged configs use the *per-stage* plan and spread the horizon across
the grid's stage axis -> B*N small blocks, each evaluating one stage. The lp/batch
configs use the *unrolled* plan (the per-stage function stamped N times into one
DAG) and run one block per batch instance -> B big blocks, each evaluating the
whole horizon. That is exactly the difference the stage axis buys: keeping blocks
small and numerous (B*N) instead of few and huge (B), so the GPU fills at a far
lower batch and the per-block shared-memory footprint stays ~N x smaller. All four
do the same total work (B*N evaluations); the x-axis is batch B at fixed horizon N.

The unrolled horizon's shared memory grows ~N x the per-stage peak, so N is kept
at 8 — the largest horizon whose unrolled plans all fit the device's per-block
limit (cusadi2 has no tiling scheduler yet).

Host-language symmetry: the MJX/warp baselines are timed from Python, so the
cusadi2 sweep is ALSO run a second time through a ctypes-loaded shared build of
latency.cu (--pylib mode, -> ./results/latency_py_<dtype>.csv), with
time.perf_counter() around each evaluate() — the Python-called wall-clock,
including the interpreter + FFI dispatch a real Python deployment would pay.
The C++-hosted numbers stay in latency_<dtype>.csv; plot.py overlays both.

MJX baseline: the same FK / frame-Jacobian workload in MuJoCo MJX (jax.vmap+jit,
contacts off), timed up to batch 1024 (its JIT compile is too slow above that).
To make the comparison "the same thing", BOTH backends evaluate the *identical*
link set — the MuJoCo body frames (excluding 'world'), looked up by name on the
Pinocchio side — so neither computes more frames than the other. MJX runs in a
separate process per dtype (the jax x64 toggle must precede any jnp op).

For each dtype this emits the kernel and, per function, both a per-stage and an
unrolled plan header into ./build/<dtype>, compiles latency.cu against the runtime
header, runs it (CSV -> ./results/latency_<dtype>.csv), benchmarks MJX
(-> ./results/mjx_<robot>_<fpXX>.json), then plots the curves with MJX overlaid.

    python examples/paper_results/run.py

Env overrides: CUSADI2_GPU (device index, default "1"), NVCC (path to nvcc).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import casadi as ca

import cusadi2
from cusadi2.kernel_template import write_kernel_source
from cusadi2.generate import generate_native_plan
from cusadi2.dag import extract_level_plan

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = cusadi2.__path__[0]          # installed package dir (holds cusadi2.h)
BUILD = os.path.join(HERE, "build")
RESULTS = os.path.join(HERE, "results")

DEVICE = os.environ.get("CUSADI2_GPU", "1")
NVCC = os.environ.get("NVCC", "nvcc")

# Stage-axis horizon (fixed) and batch sweep (the x-axis). N=8 keeps every
# unrolled plan within the device's per-block shared-memory limit.
N_STAGES = 8
BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
N_WARMUP = 5
N_TIMED = 20

# MJX / mujoco-warp: full batch sweep (wall-clock timing, not cudaEvent), with
# their own warmup/timed counts. Per-batch cases that OOM are recorded as null.
MJX_BATCHES = BATCHES
MJX_N_WARMUP = 8
MJX_N_EVALS = 20

DTYPES = ("double", "float")          # cusadi2 scalar types
DT2FP = {"double": "fp64", "float": "fp32"}

# robot tag -> (Pinocchio robot_descriptions name, MuJoCo robot_descriptions name)
ROBOTS = {
    "a1": {"pin": "a1_description", "mj": "a1_mj_description"},
    "h1": {"pin": "h1_description", "mj": "h1_mj_description"},
}


# ---------------------------------------------------------------------------
# Common frame set. BOTH backends evaluate the MuJoCo body frames (excluding
# 'world'); every such body name is also a Pinocchio BODY frame (verified for
# a1/h1), so the FK/Jacobian workloads are identical link-for-link.
# ---------------------------------------------------------------------------
def body_frame_names(robot):
    """MuJoCo body names excluding 'world', in MuJoCo body order."""
    import mujoco
    from robot_descriptions.loaders.mujoco import load_robot_description as mjload
    m = mjload(ROBOTS[robot]["mj"])
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    return [n for n in names if n != "world"]


# ---------------------------------------------------------------------------
# Pinocchio (cusadi2) side: FK and frame Jacobians restricted to the common
# link set, free-floating base, no contacts.
# ---------------------------------------------------------------------------
def _floating_base_model(description):
    import pinocchio as pin
    import pinocchio.casadi as cpin
    from robot_descriptions.loaders.pinocchio import load_robot_description
    robot = load_robot_description(description, root_joint=pin.JointModelFreeFlyer())
    cmodel = cpin.Model(robot.model)
    cdata = cmodel.createData()
    q = ca.SX.sym("q", cmodel.nq)
    return cmodel, cdata, q


def _frame_ids(cmodel, robot):
    fids = []
    for n in body_frame_names(robot):
        assert cmodel.existFrame(n), f"{robot}: frame {n!r} missing in Pinocchio model"
        fids.append(cmodel.getFrameId(n))
    return fids


def _fk(name, robot):
    """Forward kinematics f(q) -> stacked world placement (translation + flattened
    rotation) of every common link frame."""
    import pinocchio.casadi as cpin
    cmodel, cdata, q = _floating_base_model(ROBOTS[robot]["pin"])
    cpin.framesForwardKinematics(cmodel, cdata, q)
    placements = [ca.vertcat(cdata.oMf[i].translation,
                             ca.reshape(cdata.oMf[i].rotation, 9, 1))
                  for i in _frame_ids(cmodel, robot)]
    return ca.Function(name, [q], [ca.vertcat(*placements)])


def _fk_jac(name, robot):
    """Frame Jacobians f(q) -> stacked 6xnv LOCAL_WORLD_ALIGNED Jacobian of every
    common link frame."""
    import pinocchio as pin
    import pinocchio.casadi as cpin
    cmodel, cdata, q = _floating_base_model(ROBOTS[robot]["pin"])
    cpin.forwardKinematics(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)
    cpin.computeJointJacobians(cmodel, cdata, q)
    jacs = [cpin.getFrameJacobian(cmodel, cdata, i, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            for i in _frame_ids(cmodel, robot)]
    flat = [ca.reshape(J, J.shape[0] * J.shape[1], 1) for J in jacs]
    return ca.Function(name, [q], [ca.vertcat(*flat)])


def _unroll(f, n_stages):
    """Stamp per-stage f n_stages times into one function — the no-stage-axis
    (original-cusadi) horizon. Each stamp gets its own fresh inputs; the outputs
    are concatenated stage-major, so the unrolled DAG is ~n_stages x as wide/deep
    and its peak live set is ~n_stages x the per-stage plan's."""
    stamp_ins = [[ca.SX.sym(f"in{a}_{k}", f.sparsity_in(a)) for a in range(f.n_in())]
                 for k in range(n_stages)]
    outs = []
    for k in range(n_stages):
        res = f(*stamp_ins[k])
        res = res if isinstance(res, (list, tuple)) else [res]
        outs += [ca.reshape(o, o.numel(), 1) for o in res]
    flat_ins = [stamp_ins[k][a] for k in range(n_stages) for a in range(f.n_in())]
    return ca.Function(f.name() + "_unr", flat_ins, [ca.vertcat(*outs)])


# (per-stage function builder, plan-header stem, C prefix). Prefixes must match
# latency.cu; the unrolled plan reuses the stem/prefix with an "_unr" suffix.
FUNCS = [
    (lambda: _fk("a1_fk", "a1"),         "a1_fk",     "A1_FK"),
    (lambda: _fk_jac("a1_fkjac", "a1"),  "a1_fkjac",  "A1_FKJAC"),
    (lambda: _fk("h1_fk", "h1"),         "h1_fk",     "H1_FK"),
    (lambda: _fk_jac("h1_fkjac", "h1"),  "h1_fkjac",  "H1_FKJAC"),
]


# ---------------------------------------------------------------------------
# MJX side: same FK / frame-Jacobian workload in MuJoCo MJX (jax.vmap+jit).
# Runs as `python run.py --mjx --robot R --dtype fpXX --out J` so the jax x64
# toggle and GPU allocation are isolated to a fresh process.
# ---------------------------------------------------------------------------
def run_mjx(robot, dtype_str, batches, n_stages, n_warmup, n_evals):
    """Two MJX curves per quantity, both doing B*N total FK/Jacobian evaluations
    at plotted batch B (matching cusadi2's stage axis):

      *_batched  — jax.vmap over B*N independent per-knot configs, one jit launch.
                   The horizon evaluated fully in parallel (cusadi2's stage axis).
      *_rollout  — jax.vmap over B instances, jax.lax.scan over N sequential
                   mjx.step()s, computing the quantity at each knot. The realistic
                   simulate-a-trajectory usage: sequential in time, so ~N launches'
                   worth of dispatch. (Includes the dynamics step that evolves state;
                   contacts off.)
    """
    import time
    import jax
    if dtype_str == "fp64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from mujoco import mjx
    import mujoco.mjx._src.support as support
    from robot_descriptions.loaders.mujoco import load_robot_description as mjload

    dtype = jnp.float64 if dtype_str == "fp64" else jnp.float32
    m = mjload(ROBOTS[robot]["mj"])
    m.geom_contype[:] = 0           # contacts OFF
    m.geom_conaffinity[:] = 0
    print(f"[mjx] {robot} {dtype_str}: nbody={m.nbody} nq={m.nq} nv={m.nv} N={n_stages} "
          f"bodies(excl world)={m.nbody - 1}  jax {jax.__version__} {jax.devices()}")

    mx = mjx.put_model(m)
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)
    body_ids = jnp.arange(1, m.nbody)          # exclude world (body 0)
    q0 = np.asarray(d.qpos)

    # quantity extracted from an already-forwarded Data (xpos/xmat/cdof populated)
    def fk_of(data):                           # world placement of every body
        return jnp.concatenate([data.xpos[1:].reshape(-1), data.xmat[1:].reshape(-1)])

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
            "n_bodies": int(m.nbody - 1), "nq": int(m.nq), "nv": int(m.nv),
            "batches": list(batches), "results": results, "compile": compile_t}


def run_warp(robot, batches, n_stages, n_warmup, n_evals):
    """mujoco-warp forward kinematics over nworld = B*N worlds (matches cusadi2 /
    MJX-batched B*N work). fp32 only, and no autodiff — so FK only, no Jacobian."""
    import time
    import numpy as np
    import warp as wp
    import mujoco
    import mujoco_warp as mjw
    from robot_descriptions.loaders.mujoco import load_robot_description as mjload

    wp.init()
    try:                            # cold cache so the first call measures compile
        wp.clear_kernel_cache()
    except Exception:
        pass
    m = mjload(ROBOTS[robot]["mj"])
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


def run_pylib(lib_path, fp, batches, n_stages, n_warmup, n_timed, out_csv):
    """Time the native cusadi2 kernel from Python via ctypes — the host-language-
    symmetric counterpart to the MJX/warp wall-clock numbers. Same sweep and CSV
    schema as the C++ binary; wall_us here includes Python + FFI dispatch. Runs as
    `python run.py --pylib ...` in its own process (clean CUDA context, same
    CUDA_VISIBLE_DEVICES treatment as the MJX subprocesses)."""
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
    for _, stem, _ in FUNCS:
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


def gpu_arch() -> str:
    cc = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", DEVICE]
    ).decode().splitlines()[0].strip()
    return "sm_" + cc.replace(".", "")


def main() -> int:
    # --- MJX subprocess mode -------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjx", action="store_true", help="(internal) run the MJX bench and exit")
    ap.add_argument("--warp", action="store_true", help="(internal) run the mujoco-warp FK bench and exit")
    ap.add_argument("--pylib", action="store_true", help="(internal) run the Python-called cusadi2 bench and exit")
    ap.add_argument("--robot", choices=list(ROBOTS))
    ap.add_argument("--dtype", choices=["fp32", "fp64"])
    ap.add_argument("--lib", help="(--pylib) path to the shared latency library")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.pylib:
        run_pylib(args.lib, args.dtype, BATCHES, N_STAGES, N_WARMUP, N_TIMED, args.out)
        return 0
    if args.mjx or args.warp:
        if args.mjx:
            out = run_mjx(args.robot, args.dtype, MJX_BATCHES, N_STAGES, MJX_N_WARMUP, MJX_N_EVALS)
        else:
            out = run_warp(args.robot, MJX_BATCHES, N_STAGES, MJX_N_WARMUP, MJX_N_EVALS)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
        return 0

    # --- cusadi2 sweep (both dtypes) -----------------------------------------
    arch = gpu_arch()
    os.makedirs(RESULTS, exist_ok=True)
    rc = 0
    csvs = []
    for dtype in DTYPES:
        outdir = os.path.join(BUILD, dtype)
        os.makedirs(outdir, exist_ok=True)

        write_kernel_source(os.path.join(outdir, "kernel.cu"), dtype=dtype, tiled=False)
        print(f"=== building {dtype} plans (N={N_STAGES}) ===")
        # cusadi2 compile = per-function codegen (extract plan + emit header) plus
        # the one-time, function-independent nvcc kernel compile (shared). Both are
        # batch-independent — the whole point vs MJX's per-shape JIT.
        codegen_s = {}
        for builder, stem, prefix in FUNCS:
            t0 = time.perf_counter()
            fn = builder()
            generate_native_plan(fn, os.path.join(outdir, f"{stem}_plan.h"),
                                  prefix=prefix, dtype=dtype)
            unr = _unroll(fn, N_STAGES)
            generate_native_plan(unr, os.path.join(outdir, f"{stem}_unr_plan.h"),
                                  prefix=prefix + "_UNR", dtype=dtype)
            codegen_s[stem] = time.perf_counter() - t0
            ps = extract_level_plan(fn)
            pu = extract_level_plan(unr)
            print(f"  {stem}: nout={fn.nnz_out(0)} per-stage instr={fn.n_instructions()} "
                  f"smem={ps.peak_live_slots*8/1024:.1f}KB | "
                  f"unrolled instr={unr.n_instructions()} "
                  f"smem={pu.peak_live_slots*8/1024:.1f}KB  codegen={codegen_s[stem]:.2f}s")

        binpath = os.path.join(outdir, "latency")
        cmd = [NVCC, "-std=c++14", "-O3", f"-arch={arch}",
               f"-DCUSADI2_SCALAR={dtype}", f"-I{PKG}", f"-I{outdir}",
               os.path.join(HERE, "latency.cu"), os.path.join(outdir, "kernel.cu"),
               "-o", binpath]
        print("nvcc:", " ".join(cmd))
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True)
        nvcc_s = time.perf_counter() - t0
        with open(os.path.join(RESULTS, f"compile_cusadi2_{dtype}.json"), "w") as f:
            json.dump({"dtype": dtype, "codegen_s": codegen_s, "nvcc_s": nvcc_s}, f, indent=2)
        print(f"  nvcc compile: {nvcc_s:.1f}s (function-independent, shared across all funcs/batches)")

        csv = os.path.join(RESULTS, f"latency_{dtype}.csv")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE)
        cmd = [binpath, str(N_STAGES), str(N_WARMUP), str(N_TIMED)] + [str(b) for b in BATCHES]
        print(f"=== running {dtype} (N={N_STAGES}, batches={BATCHES}) ===")
        with open(csv, "w") as f:
            rc |= subprocess.run(cmd, env=env, stdout=f).returncode
        print(f"  wrote {csv}")
        csvs.append(csv)

        # Same sweep, called from Python: latency.cu rebuilt as a ctypes-loadable
        # shared library (-DCUSADI2_BENCH_LIB drops main), timed in a subprocess
        # with time.perf_counter() — host-language-symmetric with MJX/warp.
        libpath = os.path.join(outdir, "latency_lib.so")
        cmd = [NVCC, "-std=c++14", "-O3", f"-arch={arch}", "-shared",
               "-Xcompiler", "-fPIC", "-DCUSADI2_BENCH_LIB",
               f"-DCUSADI2_SCALAR={dtype}", f"-I{PKG}", f"-I{outdir}",
               os.path.join(HERE, "latency.cu"), os.path.join(outdir, "kernel.cu"),
               "-o", libpath]
        subprocess.run(cmd, check=True)
        pycsv = os.path.join(RESULTS, f"latency_py_{dtype}.csv")
        print(f"=== running {dtype} from Python (ctypes) ===")
        cmd = [sys.executable, os.path.abspath(__file__), "--pylib",
               "--dtype", DT2FP[dtype], "--lib", libpath, "--out", pycsv]
        rc |= subprocess.run(cmd, env=env).returncode

    # --- MJX baseline (one subprocess per robot x dtype) ---------------------
    mjx_env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE,
                   XLA_PYTHON_CLIENT_PREALLOCATE="false")
    print(f"=== running MJX baseline (batches={MJX_BATCHES}) ===")
    for robot in ROBOTS:
        for fp in DT2FP.values():
            out = os.path.join(RESULTS, f"mjx_{robot}_{fp}.json")
            cmd = [sys.executable, os.path.abspath(__file__), "--mjx",
                   "--robot", robot, "--dtype", fp, "--out", out]
            rc |= subprocess.run(cmd, env=mjx_env).returncode

    # --- mujoco-warp baseline (FK only, fp32; no autodiff/Jacobian) ----------
    print(f"=== running mujoco-warp baseline (FK only, fp32, batches={MJX_BATCHES}) ===")
    for robot in ROBOTS:
        out = os.path.join(RESULTS, f"warp_{robot}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--warp",
               "--robot", robot, "--out", out]
        rc |= subprocess.run(cmd, env=mjx_env).returncode

    # --- plot (best-effort; CSVs/JSONs are the primary artifacts) ------------
    sys.path.insert(0, HERE)
    from plot import plot_all, plot_throughput_all, plot_compile_all
    plot_all(csvs, RESULTS, n_stages=N_STAGES)             # 8 wall-clock latency panels
    plot_throughput_all(csvs, RESULTS, n_stages=N_STAGES)  # 8 GPU-throughput panels
    plot_compile_all(RESULTS, DTYPES, n_stages=N_STAGES)   # 8 compile-time panels
    return rc


if __name__ == "__main__":
    sys.exit(main())
