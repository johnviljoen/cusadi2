#!/usr/bin/env python
"""Latency-vs-batch saturation benchmark for cusadi2's parallelism axes, with
MuJoCo MJX and mujoco-warp baselines.

For A1 and H1 forward kinematics, frame Jacobians and frame Hessians (Pinocchio,
free-floating base, no contacts) it sweeps batch size B at fixed horizon N and times one kernel
launch for four configurations that toggle the stage and level-parallel axes:

    full      staged + level-parallel + batch    per-stage plan, level kernel
    staged    staged + batch (no LP)             per-stage plan, serial kernel
    lp        level-parallel + batch (no stages) unrolled plan,  level kernel
    batch     batch only (= original cusadi)     unrolled plan,  serial kernel

full/staged use the per-stage plan and spread the horizon across B*N items;
lp/batch use the unrolled plan (the per-stage function stamped N times into one
DAG, ~N x the work vector) over B items. full/lp run the level-parallel kernel
(one block per item, work vector in shared memory); staged/batch run the serial
kernel (one thread per item, work vector in global scratch: the original cusadi
execution model). All four do B*N evaluations. N=8 is the largest horizon whose
unrolled FK/Jacobian plans fit the per-block shared-memory limit; the unrolled
Hessians do not, so their lp config is nan (a gap in the curve).

Both backends evaluate the identical link set: the MuJoCo body frames excluding
'world' (every such name is also a Pinocchio BODY frame). FK returns each link's
world placement (translation + flattened rotation); the Jacobian returns each
link's 6 x nv LOCAL_WORLD_ALIGNED geometric Jacobian; the Hessian is the
derivative of that stacked Jacobian with respect to q, a (6 nv n_links) x nq
matrix (MJX: jax.jacfwd of MuJoCo's analytical body Jacobians).

Per dtype this writes the kernel and, per function, a per-stage and an unrolled
plan into build/<dtype>/, compiles latency.cu, runs the sweep, then reruns it
through a ctypes-loaded shared build of latency.cu timed from Python (the
host-language-symmetric counterpart to the Python-timed MJX/warp baselines).
Baselines run in subprocesses (jax x64 must be set before any jnp op; MJX is
capped at B=1024 by its per-shape JIT time; warp is fp32 FK only). The CPU
reference is CasADi's own C code generation for each per-stage function,
compiled with gcc -O3 and timed from a small C driver: one problem (a star at
B=1 on every panel) and B*N problems evaluated back to back on one core (a line
over the batch sweep). Plots follow.

    python examples/paper_results/bench.py [--no-baselines] [--plot]

Outputs in results/: latency_<dtype>.csv, latency_py_<dtype>.csv,
compile_cusadi2_<dtype>.json, cpu_<dtype>.json, mjx_<robot>_<fpXX>.json,
warp_<robot>.json, and latency_/throughput_/compile_<dtype>.png/.pdf.
Env: CUSADI2_GPU (device index, default 0), NVCC (default "nvcc"), CC ("gcc").
"""
import argparse
import csv
import ctypes
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import casadi as ca

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
RESULTS = os.path.join(HERE, "results")
DEVICE = os.environ.get("CUSADI2_GPU", "0")
NVCC = os.environ.get("NVCC", "nvcc")
CC = os.environ.get("CC", "gcc")

N_STAGES = 8
BATCHES = [1, 4, 16, 64, 256, 1024, 4096, 16384]
N_WARMUP, N_TIMED = 5, 20
MJX_BATCHES = [b for b in BATCHES if b <= 1024]
MJX_N_WARMUP, MJX_N_EVALS = 8, 20
DTYPES = ("double", "float")
DT2FP = {"double": "fp64", "float": "fp32"}

ROBOTS = {"a1": ("a1_description", "a1_mj_description"),
          "h1": ("h1_description", "h1_mj_description")}
FUNCS = [("a1_fk", "A1 forward kinematics"), ("a1_fkjac", "A1 frame Jacobians"),
         ("a1_fkhess", "A1 frame Hessians"),
         ("h1_fk", "H1 forward kinematics"), ("h1_fkjac", "H1 frame Jacobians"),
         ("h1_fkhess", "H1 frame Hessians")]
KIND = {"fk": "fk", "fkjac": "jac", "fkhess": "hess"}    # function stem suffix -> MJX quantity
CONFIGS = [("full", "staged + LP + batch (full)"), ("staged", "staged + batch (no LP)"),
           ("lp", "LP + batch, unrolled (no stages)"), ("batch", "batch only, unrolled (~ orig cusadi)")]


# ---------------------------------------------------------------------------
# CasADi functions (Pinocchio side)
# ---------------------------------------------------------------------------
def body_frame_names(robot):
    """MuJoCo body names excluding 'world', in MuJoCo body order."""
    import mujoco
    from robot_descriptions.loaders.mujoco import load_robot_description
    m = load_robot_description(ROBOTS[robot][1])
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    return [n for n in names if n != "world"]


def _model(robot):
    import pinocchio as pin
    import pinocchio.casadi as cpin
    from robot_descriptions.loaders.pinocchio import load_robot_description
    r = load_robot_description(ROBOTS[robot][0], root_joint=pin.JointModelFreeFlyer())
    cmodel = cpin.Model(r.model)
    fids = []
    for n in body_frame_names(robot):
        assert cmodel.existFrame(n), f"{robot}: frame {n!r} missing in Pinocchio model"
        fids.append(cmodel.getFrameId(n))
    return cmodel, cmodel.createData(), ca.SX.sym("q", cmodel.nq), fids


def build_function(stem):
    """stem = '<robot>_fk' | '<robot>_fkjac' | '<robot>_fkhess' -> per-stage CasADi Function of q."""
    import pinocchio as pin
    import pinocchio.casadi as cpin
    robot, kind = stem.split("_")
    if kind == "fkhess":                   # d/dq of the stacked frame Jacobians
        J = build_function(f"{robot}_fkjac")
        q = ca.SX.sym("q", J.nnz_in(0))
        return ca.Function(stem, [q], [ca.jacobian(J(q), q)])
    cmodel, cdata, q, fids = _model(robot)
    if kind == "fk":
        cpin.framesForwardKinematics(cmodel, cdata, q)
        outs = [ca.vertcat(cdata.oMf[i].translation, ca.reshape(cdata.oMf[i].rotation, 9, 1))
                for i in fids]
    else:
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)
        cpin.computeJointJacobians(cmodel, cdata, q)
        outs = [ca.reshape(J, J.numel(), 1) for J in
                (cpin.getFrameJacobian(cmodel, cdata, i, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                 for i in fids)]
    return ca.Function(stem, [q], [ca.vertcat(*outs)])


def unroll(f, n):
    """Stamp per-stage f n times into one function: the no-stage-axis horizon."""
    ins = [[ca.SX.sym(f"in{a}_{k}", f.sparsity_in(a)) for a in range(f.n_in())] for k in range(n)]
    outs = []
    for k in range(n):
        res = f(*ins[k])
        outs += [ca.reshape(o, o.numel(), 1) for o in (res if isinstance(res, (list, tuple)) else [res])]
    return ca.Function(f.name() + "_unr", [x for stamp in ins for x in stamp], [ca.vertcat(*outs)])


def config_plan(plan_dir, stem, cfg):
    """config -> (plan path, serial kernel?, n_stages)."""
    unrolled = cfg in ("lp", "batch")
    path = os.path.join(plan_dir, f"{stem}{'_unr' if unrolled else ''}.plan")
    return path, int(cfg in ("staged", "batch")), (1 if unrolled else N_STAGES)


# ---------------------------------------------------------------------------
# cusadi2 sweep: codegen, nvcc, C++-hosted run, Python-hosted run
# ---------------------------------------------------------------------------
def gpu_arch():
    cc = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader",
                                  "-i", DEVICE]).decode().split()[0]
    return "sm_" + cc.replace(".", "")


def run_cusadi2(dtype):
    import cusadi2
    outdir = os.path.join(BUILD, dtype)
    plan_dir = os.path.join(outdir, "plans")
    os.makedirs(plan_dir, exist_ok=True)
    cusadi2.write_kernel_source(os.path.join(outdir, "kernel.cu"), dtype=dtype)
    print(f"=== {dtype}: plans (N={N_STAGES}) ===")
    codegen_s = {}
    for stem, _ in FUNCS:
        t0 = time.perf_counter()
        fn = build_function(stem)
        cusadi2.write_plan(fn, os.path.join(plan_dir, f"{stem}.plan"), n_stages=N_STAGES)
        cusadi2.write_plan(unroll(fn, N_STAGES), os.path.join(plan_dir, f"{stem}_unr.plan"))
        codegen_s[stem] = time.perf_counter() - t0
        ps = cusadi2.read_plan(os.path.join(plan_dir, f"{stem}.plan"))
        pu = cusadi2.read_plan(os.path.join(plan_dir, f"{stem}_unr.plan"))
        print(f"  {stem}: nout={fn.nnz_out(0)} instr={fn.n_instructions()} "
              f"smem={ps['peak_live_slots'] * 8 / 1024:.1f}KB block={ps['block_size']} | "
              f"unrolled smem={pu['peak_live_slots'] * 8 / 1024:.1f}KB  codegen={codegen_s[stem]:.2f}s")

    common = [NVCC, "-std=c++17", "-O3", f"-arch={gpu_arch()}", f"-DCUSADI2_SCALAR={dtype}",
              f"-I{cusadi2.INCLUDE_DIR}", os.path.join(HERE, "latency.cu"),
              os.path.join(outdir, "kernel.cu")]
    binpath, libpath = os.path.join(outdir, "latency"), os.path.join(outdir, "latency_lib.so")
    t0 = time.perf_counter()
    subprocess.run(common + ["-o", binpath], check=True)
    nvcc_s = time.perf_counter() - t0
    subprocess.run(common + ["-shared", "-Xcompiler", "-fPIC", "-DCUSADI2_BENCH_LIB", "-o", libpath],
                   check=True)
    with open(os.path.join(RESULTS, f"compile_cusadi2_{dtype}.json"), "w") as f:
        json.dump({"dtype": dtype, "codegen_s": codegen_s, "nvcc_s": nvcc_s}, f, indent=2)
    print(f"  nvcc: {nvcc_s:.1f}s (function-independent)")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE)
    csv_path = os.path.join(RESULTS, f"latency_{dtype}.csv")
    print(f"=== {dtype}: C++ sweep batches={BATCHES} ===")
    with open(csv_path, "w") as f:
        subprocess.run([binpath, plan_dir, str(N_WARMUP), str(N_TIMED)] + [str(b) for b in BATCHES],
                       env=env, stdout=f, check=True)
    print(f"=== {dtype}: Python (ctypes) sweep ===")
    subprocess.run([sys.executable, __file__, "--pylib", "--dtype", DT2FP[dtype], "--lib", libpath,
                    "--plans", plan_dir, "--out", os.path.join(RESULTS, f"latency_py_{dtype}.csv")],
                   env=env, check=True)
    return csv_path


def run_pylib(lib_path, fp, plan_dir, out_csv):
    """Same sweep as the binary, each evaluate() timed with time.perf_counter()
    from Python: wall_us includes interpreter + FFI dispatch."""
    lib = ctypes.CDLL(lib_path)
    lib.c2_setup.restype, lib.c2_setup.argtypes = ctypes.c_int, [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.c2_eval.restype, lib.c2_eval.argtypes = ctypes.c_double, [ctypes.c_int]
    lib.c2_teardown.argtypes = [ctypes.c_int]
    dt = {"fp32": "f32", "fp64": "f64"}[fp]
    rows = ["dtype,func,config,batch,n_stages,grid_blocks,latency_us,wall_us"]
    for stem, _ in FUNCS:
        for B in BATCHES:
            for cfg, _ in CONFIGS:
                path, serial, nst = config_plan(plan_dir, stem, cfg)
                h = lib.c2_setup(path.encode(), serial, B)
                if h < 0:
                    rows.append(f"{dt},{stem},{cfg},{B},{nst},{B * nst},nan,nan")
                    continue
                for _ in range(N_WARMUP):
                    lib.c2_eval(h)
                kus, wus = [], []
                for _ in range(N_TIMED):
                    t0 = time.perf_counter()
                    k = lib.c2_eval(h)
                    wus.append((time.perf_counter() - t0) * 1e6)
                    kus.append(k * 1e6)
                lib.c2_teardown(h)
                k_med, w_med = sorted(kus)[N_TIMED // 2], sorted(wus)[N_TIMED // 2]
                rows.append(f"{dt},{stem},{cfg},{B},{nst},{B * nst},{k_med:.3f},{w_med:.3f}")
                print(f"[pylib] {stem:9s} {cfg:6s} B={B:6d}: wall {w_med:9.1f} us (kernel {k_med:.1f} us)")
    with open(out_csv, "w") as f:
        f.write("\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# CPU reference: CasADi C codegen, one problem, timed from C
# ---------------------------------------------------------------------------
CPU_DRIVER = r"""// Times a CasADi-generated function over n_items problems evaluated back to
// back (distinct inputs, one output buffer): median of n_timed repetitions, us.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include FN_H
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
static casadi_int nnz(const casadi_int* sp) {         // {nrow, ncol, 1} is compact dense
    return sp[2] == 1 ? sp[0] * sp[1] : sp[2 + sp[1]];
}
static int cmp(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y);
}
int main(int argc, char** argv) {
    int n_warm = atoi(argv[1]), n_timed = atoi(argv[2]);
    long n_items = atol(argv[3]);
    casadi_int sz_arg, sz_res, sz_iw, sz_w, n_in = CAT(FN, _n_in)();
    CAT(FN, _work)(&sz_arg, &sz_res, &sz_iw, &sz_w);
    const casadi_real** arg = calloc(sz_arg + 1, sizeof(*arg));
    casadi_real** res = calloc(sz_res + 1, sizeof(*res));
    casadi_int* iw = calloc(sz_iw + 1, sizeof(*iw));
    casadi_real* w = calloc(sz_w + 1, sizeof(*w));
    casadi_real** in = malloc(n_in * sizeof(*in));       // per input: n_items problems
    casadi_int* nin = malloc(n_in * sizeof(*nin));
    for (casadi_int i = 0; i < n_in; ++i) {
        nin[i] = nnz(CAT(FN, _sparsity_in)(i));
        in[i] = malloc(n_items * nin[i] * sizeof(casadi_real));
        for (long k = 0; k < n_items * nin[i]; ++k) in[i][k] = (casadi_real)(2.0 * rand() / RAND_MAX - 1.0);
    }
    for (casadi_int i = 0; i < CAT(FN, _n_out)(); ++i)
        res[i] = malloc(nnz(CAT(FN, _sparsity_out)(i)) * sizeof(casadi_real));
#define EVAL_ALL                                                                \
    for (long j = 0; j < n_items; ++j) {                                        \
        for (casadi_int i = 0; i < n_in; ++i) arg[i] = in[i] + j * nin[i];    \
        FN(arg, res, iw, w, 0);                                                 \
    }
    for (int k = 0; k < n_warm; ++k) EVAL_ALL
    double* t = malloc(n_timed * sizeof(double));
    struct timespec t0, t1;
    for (int k = 0; k < n_timed; ++k) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        EVAL_ALL
        clock_gettime(CLOCK_MONOTONIC, &t1);
        t[k] = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) * 1e-3;
    }
    qsort(t, n_timed, sizeof(double), cmp);
    printf("%.4f\n", t[n_timed / 2]);
    return 0;
}
"""


def run_cpu(dtype, n_timed=2000):
    """CasADi's own C codegen for each per-stage function, gcc -O3 -march=native,
    timed from C: latency_us is one problem (one instance, one stage); seq_us[B]
    is B*N problems evaluated back to back on one core, the sequential CPU
    counterpart of every GPU curve. compile_s is codegen + gcc."""
    outdir = os.path.join(BUILD, dtype, "cpu")
    os.makedirs(outdir, exist_ok=True)
    driver = os.path.join(outdir, "driver.c")
    with open(driver, "w") as f:
        f.write(CPU_DRIVER)
    lat, seq, comp = {}, {}, {}

    def run(exe, n_items):                 # median us for n_items sequential evaluations
        reps = max(5, n_timed // n_items)
        return float(subprocess.check_output([exe, str(max(1, reps // 10)), str(reps), str(n_items)]))

    print(f"=== {dtype}: CasADi C codegen on CPU (1 problem, then B*N sequential) ===")
    for stem, _ in FUNCS:
        t0 = time.perf_counter()
        gen = ca.CodeGenerator(stem, {"with_header": True, "casadi_real": dtype})
        gen.add(build_function(stem))
        gen.generate(outdir + os.sep)
        exe = os.path.join(outdir, stem)
        subprocess.run([CC, "-O3", "-march=native", f"-DFN={stem}", f'-DFN_H="{stem}.h"', f"-I{outdir}",
                        driver, os.path.join(outdir, f"{stem}.c"), "-lm", "-o", exe], check=True)
        comp[stem] = time.perf_counter() - t0
        lat[stem] = run(exe, 1)
        seq[stem] = {str(B): run(exe, B * N_STAGES) for B in BATCHES}
        print(f"  {stem}: {lat[stem]:9.2f} us/problem   codegen + gcc {comp[stem]:5.1f} s   "
              f"B=16384: {seq[stem]['16384'] / 1e3:9.1f} ms")
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    with open(os.path.join(RESULTS, f"cpu_{dtype}.json"), "w") as f:
        json.dump({"dtype": dtype, "cpu": cpu, "latency_us": lat, "seq_us": seq, "compile_s": comp},
                  f, indent=2)


# ---------------------------------------------------------------------------
# Baselines: MuJoCo MJX (fk/jac, batched + rollout + GPU-only) and mujoco-warp (fk)
# ---------------------------------------------------------------------------
def _median_wall(fn, n_warmup, n_evals, sync):
    import numpy as np
    for _ in range(n_warmup):
        sync(fn())
    s = np.empty(n_evals)
    for i in range(n_evals):
        t = time.perf_counter()
        sync(fn())
        s[i] = time.perf_counter() - t
    return float(np.median(s))


def run_mjx(robot, fp):
    """Two MJX curves per quantity, both doing B*N evaluations at plotted B:
    *_batched (vmap over B*N configs, one jit) and *_rollout (vmap over B,
    lax.scan over N mjx.step()s; includes the dynamics step). *_batched_gpu is
    the repeat-subtraction marginal GPU time per launch, overhead-free."""
    import jax
    if fp == "fp64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import mujoco
    from mujoco import mjx
    import mujoco.mjx._src.support as support
    from robot_descriptions.loaders.mujoco import load_robot_description
    from functools import partial

    dtype = jnp.float64 if fp == "fp64" else jnp.float32
    m = load_robot_description(ROBOTS[robot][1])
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    mx, dx = mjx.put_model(m), mjx.put_data(m, mujoco.MjData(m))
    body_ids = jnp.arange(1, m.nbody)
    q0 = np.asarray(mujoco.MjData(m).qpos)
    print(f"[mjx] {robot} {fp}: nbody={m.nbody} nq={m.nq} nv={m.nv} jax {jax.__version__}")

    def fk_of(d):
        return jnp.concatenate([d.xpos[1:].reshape(-1), d.xmat[1:].reshape(-1)])

    def jac_of(d):
        def one(bid):
            jp, jr = support.jac(mx, d, d.xpos[bid], bid)
            return jnp.concatenate([jp.reshape(-1), jr.reshape(-1)])
        return jax.vmap(one)(body_ids).reshape(-1)

    def kin(q):
        return mjx.com_pos(mx, mjx.kinematics(mx, dx.replace(qpos=q)))

    def hess_of(d):                        # d/dq of the stacked body Jacobians (forward mode: nq inputs)
        return jax.jacfwd(lambda q: jac_of(kin(q)))(d.qpos).reshape(-1)

    def make_rollout(of):
        def roll(q):
            def step(d, _):
                d = mjx.step(mx, d)
                return d, of(d)
            return jax.lax.scan(step, dx.replace(qpos=q), None, length=N_STAGES)[1]
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

    rng = np.random.default_rng(0)
    results, compile_t = {}, {}
    for kind, of in (("fk", fk_of), ("jac", jac_of), ("hess", hess_of)):
        for variant, fn, per_inst in (("batched", jax.jit(jax.vmap(lambda q, of=of: of(kin(q)))), N_STAGES),
                                      ("rollout", make_rollout(of), 1)):
            key = f"{kind}_{variant}"
            results[key], compile_t[key] = {}, {}
            for B in MJX_BATCHES:
                qb = qbatch(rng, B * per_inst)
                try:
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(qb))
                    comp = time.perf_counter() - t0
                    med = _median_wall(lambda: fn(qb), MJX_N_WARMUP, MJX_N_EVALS, jax.block_until_ready)
                    results[key][str(B)], compile_t[key][str(B)] = med, comp
                    print(f"[mjx]   {key:11s} B={B:6d}: {med * 1e6:10.1f} us   compile {comp:6.2f} s")
                except Exception as e:
                    results[key][str(B)] = compile_t[key][str(B)] = None
                    print(f"[mjx]   {key:11s} B={B:6d}: FAILED ({repr(e)[:70]})")
        gkey, rep = f"{kind}_batched_gpu", make_repeat(of)
        results[gkey] = {}
        for B in MJX_BATCHES:
            qb = qbatch(rng, B * N_STAGES)
            try:
                med = lambda R: _median_wall(lambda: rep(qb, R), 4, MJX_N_EVALS, jax.block_until_ready)
                g = (med(200) - med(1)) / 199
                results[gkey][str(B)] = g
                print(f"[mjx]   {gkey:15s} B={B:6d}: {g * 1e6:10.2f} us/launch")
            except Exception as e:
                results[gkey][str(B)] = None
                print(f"[mjx]   {gkey:15s} B={B:6d}: FAILED ({repr(e)[:60]})")
    return {"robot": robot, "dtype": fp, "n_stages": N_STAGES, "n_bodies": int(m.nbody - 1),
            "nq": int(m.nq), "nv": int(m.nv), "batches": MJX_BATCHES,
            "results": results, "compile": compile_t}


def run_warp(robot):
    """mujoco-warp FK over nworld = B*N worlds. fp32 only, no autodiff."""
    import numpy as np
    import warp as wp
    import mujoco
    import mujoco_warp as mjw
    from robot_descriptions.loaders.mujoco import load_robot_description
    wp.init()
    try:
        wp.clear_kernel_cache()          # cold cache so the first call measures compile
    except Exception:
        pass
    m = load_robot_description(ROBOTS[robot][1])
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    mjw_m = mjw.put_model(m)
    rng = np.random.default_rng(0)
    q0 = np.asarray(mujoco.MjData(m).qpos)
    results, compile_once = {"fk": {}}, None
    for B in MJX_BATCHES:
        nworld = B * N_STAGES
        try:
            mjw_d = mjw.put_data(m, mujoco.MjData(m), nworld=nworld)
            qpos = (rng.normal(0.0, 0.3, size=(nworld, m.nq)) + q0).astype(np.float32)
            mjw_d.qpos = wp.from_numpy(qpos, dtype=wp.float32, device="cuda:0")
            t0 = time.perf_counter()
            mjw.kinematics(mjw_m, mjw_d)
            wp.synchronize_device("cuda:0")
            if compile_once is None:
                compile_once = time.perf_counter() - t0
            med = _median_wall(lambda: mjw.kinematics(mjw_m, mjw_d), MJX_N_WARMUP, MJX_N_EVALS,
                               lambda _: wp.synchronize_device("cuda:0"))
            results["fk"][str(B)] = med
            print(f"[mjw]   fk B={B:6d} (nworld={nworld}): {med * 1e6:9.1f} us")
        except Exception as e:
            results["fk"][str(B)] = None
            print(f"[mjw]   fk B={B:6d}: FAILED ({repr(e)[:70]})")
    return {"robot": robot, "backend": "mujoco_warp", "dtype": "fp32", "n_stages": N_STAGES,
            "nq": int(m.nq), "nv": int(m.nv), "batches": MJX_BATCHES, "results": results,
            "compile_once_s": compile_once}


# ---------------------------------------------------------------------------
# Plots: latency (wall-clock), throughput (GPU-only, per eval), compile time
# ---------------------------------------------------------------------------
def _read_csv(path, col):
    """csv -> {func: {config: ([B...], [us...])}}; nan rows dropped."""
    if not os.path.exists(path):
        return {}
    rows = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows[r["func"]][r["config"]].append((int(r["batch"]), float(r[col])))
            except ValueError:
                continue
    return {fn: {c: tuple(zip(*sorted(p))) for c, p in by.items()} for fn, by in rows.items()}


def _json(name):
    path = os.path.join(RESULTS, name)
    return json.load(open(path)) if os.path.exists(path) else {}


def _series(by_b, scale=1e6):
    """{B: seconds|None} -> ([B...], [scaled...]) or None."""
    pts = sorted((int(b), v * scale) for b, v in (by_b or {}).items() if v is not None)
    return tuple(zip(*pts)) if pts else None


def _cpu_point(ax, dtype, key, func, scale=1.0, per_eval=False):
    """CasADi C codegen on the CPU: a star at B=1 for one problem, and (latency /
    throughput panels) a line for B*N problems evaluated sequentially."""
    cpu = _json(f"cpu_{dtype}.json")
    v = cpu.get(key, {}).get(func)
    if v is not None:
        ax.plot([1], [v * scale], marker="*", ms=13, color="gold", markeredgecolor="black", ls="none",
                label="CasADi C codegen, CPU, 1 problem")
    s = _series(cpu.get("seq_us", {}).get(func), scale) if key == "latency_us" else None
    if s:
        ys = [y / (b * N_STAGES) for b, y in zip(*s)] if per_eval else s[1]
        ax.plot(s[0], ys, marker="*", ms=7, color="gold", markeredgecolor="black", ls="-", lw=1.5,
                label="CasADi C codegen, CPU, B·N sequential")


def _figure(dtype, ylabel, title, name, draw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fp = DT2FP[dtype]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, (func, ftitle) in zip(axes.flat, FUNCS):
        robot, kind = func.split("_")
        draw(ax, func, robot, KIND[kind], fp)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(ftitle)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.set_xlabel("batch size B")
        ax.set_ylabel(ylabel)
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{title}  ({fp}, horizon N={N_STAGES}, total work = B·N)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    stem = os.path.join(RESULTS, f"{name}_{dtype}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


def plot_latency(dtype):
    data = _read_csv(os.path.join(RESULTS, f"latency_{dtype}.csv"), "wall_us")
    py = _read_csv(os.path.join(RESULTS, f"latency_py_{dtype}.csv"), "wall_us")

    def draw(ax, func, robot, kind, fp):
        for cfg, label in CONFIGS:
            s = data.get(func, {}).get(cfg)
            if s:
                (ln,) = ax.plot(*s, marker="o", ms=4, label=label)
                p = py.get(func, {}).get(cfg)
                if p:
                    ax.plot(*p, color=ln.get_color(), ls="--", marker="x", ms=5, lw=1, alpha=0.8)
        mjx = _json(f"mjx_{robot}_{fp}.json").get("results", {})
        for suffix, ls, mk, lab in (("batched", "--", "s", "MJX batched, B·N"),
                                    ("rollout", ":", "^", "MJX rollout, N steps")):
            s = _series(mjx.get(f"{kind}_{suffix}"))
            if s:
                ax.plot(*s, marker=mk, ms=4, color="black", ls=ls, label=f"{lab} ({fp})")
        if kind == "fk" and fp == "fp32":
            s = _series(_json(f"warp_{robot}.json").get("results", {}).get("fk"))
            if s:
                ax.plot(*s, marker="D", ms=4, color="#9467bd", ls="-.", label="mujoco-warp (fp32)")
        _cpu_point(ax, dtype, "latency_us", func)
        if py and func == FUNCS[0][0]:
            ax.plot([], [], color="gray", ls="--", marker="x", ms=5, lw=1,
                    label="dashed ×: same, called from Python (ctypes)")
    _figure(dtype, "wall-clock latency (µs)", "Single-call wall-clock latency vs batch", "latency", draw)


def plot_throughput(dtype):
    data = _read_csv(os.path.join(RESULTS, f"latency_{dtype}.csv"), "latency_us")

    def draw(ax, func, robot, kind, fp):
        for cfg, label in CONFIGS:
            s = data.get(func, {}).get(cfg)
            if s:
                ax.plot(s[0], [u * 1e3 / (b * N_STAGES) for b, u in zip(*s)], marker="o", ms=4, label=label)
        g = _json(f"mjx_{robot}_{fp}.json").get("results", {}).get(f"{kind}_batched_gpu")
        s = _series(g, 1e9)
        if s:
            ax.plot(s[0], [ns / (b * N_STAGES) for b, ns in zip(*s)], marker="s", ms=4, color="black",
                    ls="--", label=f"MJX batched, GPU-only ({fp})")
        _cpu_point(ax, dtype, "latency_us", func, 1e3, per_eval=True)   # ns per eval
    _figure(dtype, "GPU time per eval (ns)", "Per-eval GPU throughput vs batch (launch overhead removed)",
            "throughput", draw)


def plot_compile(dtype):
    cc = _json(f"compile_cusadi2_{dtype}.json")
    xb = [BATCHES[0], BATCHES[-1]]

    def draw(ax, func, robot, kind, fp):
        if cc:
            y = cc["codegen_s"].get(func, 0.0) + cc["nvcc_s"]
            ax.plot(xb, [y, y], color="C0", lw=2, label="cusadi2 (codegen + nvcc, once)")
        comp = _json(f"mjx_{robot}_{fp}.json").get("compile", {})
        for suffix, ls, mk, lab in (("batched", "--", "s", "MJX batched"), ("rollout", ":", "^", "MJX rollout")):
            s = _series(comp.get(f"{kind}_{suffix}"), 1.0)
            if s:
                ax.plot(*s, color="black", ls=ls, marker=mk, ms=4, label=f"{lab} JIT ({fp})")
        if kind == "fk" and fp == "fp32":
            w = _json(f"warp_{robot}.json").get("compile_once_s")
            if w:
                ax.plot(xb, [w, w], color="#9467bd", ls="-.", lw=2, label="mujoco-warp (once, fp32)")
        _cpu_point(ax, dtype, "compile_s", func)
    _figure(dtype, "compile time (s)", "Compile / codegen time vs batch", "compile", draw)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-baselines", action="store_true", help="skip the MJX and mujoco-warp runs")
    ap.add_argument("--plot", action="store_true", help="only re-plot existing results")
    ap.add_argument("--cpu", action="store_true", help="only run the CPU reference, then plot")
    ap.add_argument("--pylib", action="store_true", help="(internal) Python-timed cusadi2 sweep")
    ap.add_argument("--mjx", action="store_true", help="(internal) MJX baseline")
    ap.add_argument("--warp", action="store_true", help="(internal) mujoco-warp baseline")
    ap.add_argument("--robot", choices=list(ROBOTS))
    ap.add_argument("--dtype", choices=["fp32", "fp64"])
    ap.add_argument("--lib"); ap.add_argument("--plans"); ap.add_argument("--out")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if args.pylib:
        return run_pylib(args.lib, args.dtype, args.plans, args.out)
    if args.mjx or args.warp:
        out = run_mjx(args.robot, args.dtype) if args.mjx else run_warp(args.robot)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        return print(f"wrote {args.out}")

    if args.cpu:
        for dtype in DTYPES:
            run_cpu(dtype)
    elif not args.plot:
        for dtype in DTYPES:
            run_cusadi2(dtype)
            run_cpu(dtype)
        if not args.no_baselines:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=DEVICE, XLA_PYTHON_CLIENT_PREALLOCATE="false")
            for robot in ROBOTS:
                for fp in DT2FP.values():
                    subprocess.run([sys.executable, __file__, "--mjx", "--robot", robot, "--dtype", fp,
                                    "--out", os.path.join(RESULTS, f"mjx_{robot}_{fp}.json")], env=env)
                subprocess.run([sys.executable, __file__, "--warp", "--robot", robot,
                                "--out", os.path.join(RESULTS, f"warp_{robot}.json")], env=env)
    for dtype in DTYPES:
        plot_latency(dtype)
        plot_throughput(dtype)
        plot_compile(dtype)


if __name__ == "__main__":
    main()
