#!/usr/bin/env python
"""Head-to-head latency benchmark: cusadi2 vs frax (Morton & Pavone, arXiv 2604.04310).

Robots are the two frax ships, Franka Panda (7 dof) and Unitree G1 (29 joints +
frax's floating base, 35 dof). Both libraries evaluate the same four functions of
the same model on the same inputs:

    rnea   inverse dynamics tau(q, qd, qdd), gravity on, no external forces
    crba   mass matrix M(q), full symmetric
    jac    d tau / d q            frax: jax.vmap(jax.jacfwd(rnea))
                                  cusadi2: ca.jacobian
    hess   d^2 tau / d q^2        frax: jax.vmap(jax.hessian(rnea))
                                  cusadi2: ca.jacobian of the Jacobian

The cusadi2 side builds its CasADi functions with Pinocchio from frax's own URDFs,
in frax's joint order. frax models the G1's floating base as six virtual joints
(x, y, z prismatic, then roll, pitch, yaw revolute), not a quaternion free-flyer,
so the Pinocchio root is the same composite joint and every output matches frax's
element by element. cusadi2 outputs hold CasADi's structural nonzeros only; frax
returns dense arrays.

Timing is identical on both sides: time.perf_counter around one batched call
from Python (cusadi2 through ctypes, driver.cu), median and 90th percentile after
warm-up, in two variants:

    includes_io=0   inputs and outputs stay on the device; the call returns when
                    the result is ready (block_until_ready / stream sync)
    includes_io=1   inputs start as host numpy arrays and the result ends as one:
                    frax np.asarray(f(numpy...)), cusadi2 cudaMemcpy in, launch,
                    cudaMemcpy out into a preallocated numpy buffer

Compile cost is recorded apart from latency: cusadi2 codegen per function (CasADi
build + plan) and one function-independent nvcc per dtype; frax first-call time
per function and batch shape (trace + XLA compile + one run).

    python examples/frax_bench/bench.py [--dtypes double float] [--plot]

Writes results/latency.csv (robot, function, library, dtype, batch, median_us,
p90_us, evals_per_s, includes_io), results/compile.json, results/check.json (the
two libraries' outputs compared at batch CHECK_BATCH) and one figure per function
and dtype. Env: CUSADI2_GPU (device index, default 0), NVCC (default "nvcc").
"""
import argparse
import csv
import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
RESULTS = os.path.join(HERE, "results")
DEVICE = os.environ.get("CUSADI2_GPU", "0")
NVCC = os.environ.get("NVCC", "nvcc")

BATCHES = [1, 4, 16, 64, 256, 1024, 4096]
CHECK_BATCH = 64                           # batch whose outputs the two libraries are compared at
DTYPES = ("double", "float")
DT2FP = {"double": "fp64", "float": "fp32"}
NP = {"double": np.float64, "float": np.float32}
ROBOTS = [("panda", "Franka Panda (7 dof)"), ("g1", "Unitree G1 (35 dof)")]
FUNCS = [("rnea", "RNEA (inverse dynamics)"), ("crba", "CRBA (mass matrix)"),
         ("jac", "Jacobian of RNEA w.r.t. q"), ("hess", "Hessian of RNEA w.r.t. q")]
COLUMNS = ["robot", "function", "library", "dtype", "batch", "median_us", "p90_us", "evals_per_s", "includes_io"]
BASE_ANGLE = 0.5                           # floating-base roll/pitch/yaw sampled in +-BASE_ANGLE rad


def n_args(kind):
    return 1 if kind == "crba" else 3      # (q) or (q, qd, qdd)


# ---------------------------------------------------------------------------
# Models, functions and inputs
# ---------------------------------------------------------------------------
def load_frax(robot):
    import frax
    return frax.load_panda() if robot == "panda" else frax.load_g1()


def pin_model(robot):
    """Pinocchio model of frax's URDF; the G1 gets frax's floating base, six
    virtual joints (x, y, z prismatic then x, y, z revolute)."""
    import pinocchio as pin
    from frax.assets import FRANKA_ASSETS_DIR, G1_ASSETS_DIR
    if robot == "panda":
        return pin.buildModelFromUrdf(str(FRANKA_ASSETS_DIR / "panda.urdf"))
    root = pin.JointModelComposite()
    for j in (pin.JointModelPX(), pin.JointModelPY(), pin.JointModelPZ(),
              pin.JointModelRX(), pin.JointModelRY(), pin.JointModelRZ()):
        root.addJoint(j)
    return pin.buildModelFromUrdf(str(G1_ASSETS_DIR / "g1_29dof_rev_1_0.urdf"), root)


def build_function(robot, kind, frobot=None):
    """CasADi Function '<robot>_<kind>' of (q, qd, qdd) (crba: q) in frax's joint order."""
    import casadi as ca
    import pinocchio.casadi as cpin
    frobot = frobot or load_frax(robot)
    model = pin_model(robot)
    n, base = frobot.num_joints, 6 if frobot.includes_floating_dof else 0
    assert model.nq == model.nv == n
    perm = list(range(base)) + [model.joints[model.getJointId(name)].idx_q for name in frobot.joint_names[base:]]
    assert sorted(perm) == list(range(n))

    def to_pin(x):                         # frax order -> Pinocchio order
        return ca.vertcat(*[x[perm.index(i)] for i in range(n)])

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    q, qd, qdd = (ca.SX.sym(s, n) for s in ("q", "qd", "qdd"))
    if kind == "crba":
        M = cpin.crba(cmodel, cdata, to_pin(q))            # upper triangle
        M = ca.triu(M) + ca.triu(M, False).T
        return ca.Function(f"{robot}_crba", [q], [M[perm, perm]])
    out = cpin.rnea(cmodel, cdata, to_pin(q), to_pin(qd), to_pin(qdd))[perm]
    if kind in ("jac", "hess"):
        out = ca.jacobian(out, q)
    if kind == "hess":                     # (n*n) x n: row i + n*j is d/dq of d tau_i / d q_j
        out = ca.jacobian(out, q)
    return ca.Function(f"{robot}_{kind}", [q, qd, qdd], [out])


def make_inputs(robot, frobot):
    """max(BATCHES) samples, fp64: q uniform inside the joint limits (floating
    base: position in +-1 m, roll/pitch/yaw in +-BASE_ANGLE rad, clear of the
    pitch singularity), qd and qdd standard normal. Batch B uses the first B rows."""
    rng = np.random.default_rng(sum(map(ord, robot)))
    n, B = frobot.num_joints, max(BATCHES)
    lo, hi = np.array(frobot.joint_lower_limits, float), np.array(frobot.joint_upper_limits, float)
    if frobot.includes_floating_dof:
        lo[:3], hi[:3] = -1.0, 1.0
        lo[3:6], hi[3:6] = -BASE_ANGLE, BASE_ANGLE
    return {"q": rng.uniform(lo, hi, size=(B, n)), "qd": rng.normal(size=(B, n)), "qdd": rng.normal(size=(B, n))}


def load_inputs(robot, kind, B, dtype):
    z = np.load(os.path.join(BUILD, f"inputs_{robot}.npz"))
    return [np.ascontiguousarray(z[k][:B], dtype=NP[dtype]) for k in ("q", "qd", "qdd")[:n_args(kind)]]


def timed(fn, budget_s=2.0, n_min=20, n_max=2000, warm_s=0.3, warm_min=5):
    """Warm up (at least warm_min calls and warm_s seconds), then time
    clamp(budget_s / call, n_min, n_max) calls -> (median us, p90 us, n)."""
    t_end, k = time.perf_counter() + warm_s, 0
    while k < warm_min or time.perf_counter() < t_end:
        t0 = time.perf_counter()
        fn()
        one = time.perf_counter() - t0
        k += 1
    n = int(min(n_max, max(n_min, budget_s / max(one, 1e-9))))
    s = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        fn()
        s[i] = time.perf_counter() - t0
    return float(np.median(s) * 1e6), float(np.percentile(s, 90) * 1e6), n


class Rows:
    """Appends result rows to a part CSV as they are measured."""

    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(COLUMNS)

    def add(self, robot, kind, lib, dtype, B, med, p90, io):
        self.w.writerow([robot, kind, lib, DT2FP[dtype], B, f"{med:.3f}", f"{p90:.3f}", f"{B / med * 1e6:.1f}", io])
        self.f.flush()


# ---------------------------------------------------------------------------
# cusadi2: codegen, nvcc, Python-timed sweep
# ---------------------------------------------------------------------------
def gpu_arch():
    cc = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader",
                                  "-i", DEVICE]).decode().split()[0]
    return "sm_" + cc.replace(".", "")


def run_codegen(dtypes):
    """Inputs, per-function metadata and, per dtype, the plans (max_live = the
    device's shared-memory limit, block size left to the runtime)."""
    import cusadi2
    os.makedirs(BUILD, exist_ok=True)
    meta = {}
    codegen = {dt: {} for dt in dtypes}
    for robot, _ in ROBOTS:
        frobot = load_frax(robot)
        np.savez(os.path.join(BUILD, f"inputs_{robot}.npz"), **make_inputs(robot, frobot))
        for kind, _ in FUNCS:
            t0 = time.perf_counter()
            fn = build_function(robot, kind, frobot)
            build_s = time.perf_counter() - t0
            sp = fn.sparsity_out(0)
            rows, cols = sp.get_triplet()
            m = {"n": frobot.num_joints, "instructions": fn.n_instructions(), "nnz_out": sp.nnz(),
                 "dense_out": sp.numel(), "rows": list(rows), "cols": list(cols), "plan": {}}
            for dt in dtypes:
                path = os.path.join(BUILD, dt, "plans", f"{fn.name()}.plan")
                t0 = time.perf_counter()
                cusadi2.write_plan(fn, path, max_live=cusadi2.device_max_live(dt, int(DEVICE)))
                codegen[dt][fn.name()] = build_s + time.perf_counter() - t0
                p = cusadi2.read_plan(path)
                m["plan"][dt] = {"levels": len(p["level_widths"]), "peak_live_slots": p["peak_live_slots"]}
                print(f"  {fn.name():11s} {dt:6s}: instr={m['instructions']:7d} nnz_out={m['nnz_out']:5d}/{m['dense_out']:5d} "
                      f"levels={m['plan'][dt]['levels']:5d} peak_live={p['peak_live_slots']:6d}  "
                      f"codegen {codegen[dt][fn.name()]:.2f} s", flush=True)
            meta[fn.name()] = m
    with open(os.path.join(BUILD, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(BUILD, "codegen.json"), "w") as f:
        json.dump(codegen, f, indent=2)


def run_nvcc(dtype):
    import cusadi2
    outdir = os.path.join(BUILD, dtype)
    cusadi2.write_kernel_source(os.path.join(outdir, "kernel.cu"), dtype=dtype)
    lib = os.path.join(outdir, "driver.so")
    t0 = time.perf_counter()
    subprocess.run([NVCC, "-std=c++17", "-O3", f"-arch={gpu_arch()}", f"-DCUSADI2_SCALAR={dtype}",
                    f"-I{cusadi2.INCLUDE_DIR}", os.path.join(HERE, "driver.cu"), os.path.join(outdir, "kernel.cu"),
                    "-shared", "-Xcompiler", "-fPIC", "-o", lib], check=True)
    nvcc_s = time.perf_counter() - t0
    print(f"  nvcc {dtype}: {nvcc_s:.1f} s (kernel + driver, function-independent)", flush=True)
    return nvcc_s


def scatter_dense(nz, m, kind):
    """(B, nnz) structural nonzeros -> frax's dense output shape."""
    B, n = nz.shape[0], m["n"]
    rows, cols = np.array(m["rows"]), np.array(m["cols"])
    if kind == "rnea":
        out = np.zeros((B, n), nz.dtype)
        out[:, rows] = nz
    elif kind in ("crba", "jac"):
        out = np.zeros((B, n, n), nz.dtype)
        out[:, rows, cols] = nz
    else:
        out = np.zeros((B, n, n, n), nz.dtype)
        out[:, rows % n, rows // n, cols] = nz
    return out


def run_cusadi2(dtype):
    S = {"double": ctypes.c_double, "float": ctypes.c_float}[dtype]
    lib = ctypes.CDLL(os.path.join(BUILD, dtype, "driver.so"))
    ptr = ctypes.POINTER(S)
    lib.fb_setup.restype, lib.fb_setup.argtypes = ctypes.c_int, [ctypes.c_char_p, ctypes.c_int]
    for name in ("fb_in_len", "fb_out_len"):
        getattr(lib, name).restype, getattr(lib, name).argtypes = ctypes.c_longlong, [ctypes.c_int, ctypes.c_int]
    lib.fb_block_size.restype, lib.fb_block_size.argtypes = ctypes.c_int, [ctypes.c_int]
    lib.fb_upload.argtypes = lib.fb_download.argtypes = [ctypes.c_int, ctypes.c_int, ptr]
    lib.fb_eval.argtypes = lib.fb_teardown.argtypes = [ctypes.c_int]
    lib.fb_eval_io.argtypes = [ctypes.c_int, ctypes.POINTER(ptr), ctypes.POINTER(ptr)]
    meta = json.load(open(os.path.join(BUILD, "meta.json")))
    rows = Rows(os.path.join(BUILD, f"rows_cusadi2_{dtype}.csv"))
    info = {}
    for robot, _ in ROBOTS:
        for kind, _ in FUNCS:
            stem = f"{robot}_{kind}"
            plan = os.path.join(BUILD, dtype, "plans", f"{stem}.plan").encode()
            for B in BATCHES:
                h = lib.fb_setup(plan, B)
                if h < 0:
                    print(f"[cusadi2] {stem:11s} B={B:5d}: plan exceeds shared memory, skipped", flush=True)
                    break
                ins = load_inputs(robot, kind, B, dtype)
                assert all(lib.fb_in_len(h, a) == x.size for a, x in enumerate(ins))
                for a, x in enumerate(ins):
                    lib.fb_upload(h, a, x.ctypes.data_as(ptr))
                out = np.empty(lib.fb_out_len(h, 0), NP[dtype])
                h_in = (ptr * len(ins))(*[x.ctypes.data_as(ptr) for x in ins])
                h_out = (ptr * 1)(out.ctypes.data_as(ptr))
                med, p90, n = timed(lambda: lib.fb_eval(h))
                rows.add(robot, kind, "cusadi2", dtype, B, med, p90, 0)
                med_io, p90_io, _ = timed(lambda: lib.fb_eval_io(h, h_in, h_out))
                rows.add(robot, kind, "cusadi2", dtype, B, med_io, p90_io, 1)
                info[stem] = {"block_size": lib.fb_block_size(h)}
                if B == CHECK_BATCH:
                    np.save(os.path.join(BUILD, f"check_cusadi2_{stem}_{dtype}.npy"),
                            scatter_dense(out.reshape(B, -1), meta[stem], kind))
                lib.fb_teardown(h)
                print(f"[cusadi2] {stem:11s} {dtype:6s} B={B:5d}: {med:10.1f} us   with I/O {med_io:10.1f} us   "
                      f"(n={n}, block {info[stem]['block_size']})", flush=True)
    with open(os.path.join(BUILD, f"info_cusadi2_{dtype}.json"), "w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# frax: jit(vmap(...)), Python-timed sweep
# ---------------------------------------------------------------------------
def run_frax(dtype):
    import jax
    jax.config.update("jax_enable_x64", dtype == "double")
    import jax.numpy as jnp
    rows = Rows(os.path.join(BUILD, f"rows_frax_{dtype}.csv"))
    jit_s = {}
    print(f"[frax] jax {jax.__version__} {DT2FP[dtype]} on {jax.devices()}", flush=True)
    for robot, _ in ROBOTS:
        frobot = load_frax(robot)
        gravity = jnp.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])

        def rnea(q, qd, qdd):
            return frobot.rnea(q, qd, qdd, gravity, None)

        fns = {"rnea": rnea, "crba": frobot.crba, "jac": jax.jacfwd(rnea), "hess": jax.hessian(rnea)}
        for kind, _ in FUNCS:
            stem = f"{robot}_{kind}"
            f = jax.jit(jax.vmap(fns[kind]))
            jit_s[stem] = {}
            for B in BATCHES:
                host = load_inputs(robot, kind, B, dtype)
                try:
                    dev = [jax.block_until_ready(jax.device_put(x)) for x in host]
                    t0 = time.perf_counter()
                    out = jax.block_until_ready(f(*dev))               # trace + XLA compile + one run
                    jit_s[stem][str(B)] = time.perf_counter() - t0
                    assert out.dtype == NP[dtype]
                    med, p90, n = timed(lambda: jax.block_until_ready(f(*dev)))
                    rows.add(robot, kind, "frax", dtype, B, med, p90, 0)
                    med_io, p90_io, _ = timed(lambda: np.asarray(f(*host)))
                    rows.add(robot, kind, "frax", dtype, B, med_io, p90_io, 1)
                    if B == CHECK_BATCH:
                        np.save(os.path.join(BUILD, f"check_frax_{stem}_{dtype}.npy"), np.asarray(out))
                    del out, dev
                    print(f"[frax]    {stem:11s} {dtype:6s} B={B:5d}: {med:10.1f} us   with I/O {med_io:10.1f} us   "
                          f"(n={n}, first call {jit_s[stem][str(B)]:.2f} s)", flush=True)
                except Exception as e:                                 # out of device memory: larger batches too
                    jit_s[stem][str(B)] = None
                    print(f"[frax]    {stem:11s} {dtype:6s} B={B:5d}: FAILED ({repr(e)[:200]})", flush=True)
                    break
    with open(os.path.join(BUILD, f"jit_frax_{dtype}.json"), "w") as f:
        json.dump(jit_s, f, indent=2)


# ---------------------------------------------------------------------------
# Report: merged CSV, output check, compile times, figures
# ---------------------------------------------------------------------------
def read_rows():
    out = []
    for dt in DTYPES:
        for lib in ("cusadi2", "frax"):
            path = os.path.join(BUILD, f"rows_{lib}_{dt}.csv")
            if os.path.exists(path):
                out += list(csv.DictReader(open(path)))
    return out


def report():
    os.makedirs(RESULTS, exist_ok=True)
    rows = read_rows()
    with open(os.path.join(RESULTS, "latency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)

    check = {}
    for dt in DTYPES:
        for robot, _ in ROBOTS:
            for kind, _ in FUNCS:
                a, b = (os.path.join(BUILD, f"check_{lib}_{robot}_{kind}_{dt}.npy") for lib in ("cusadi2", "frax"))
                if os.path.exists(a) and os.path.exists(b):
                    a, b = np.load(a).astype(float), np.load(b).astype(float)
                    assert a.shape == b.shape
                    check[f"{robot}_{kind}_{DT2FP[dt]}"] = {
                        "batch": CHECK_BATCH, "max_abs_frax": float(np.abs(b).max()),
                        "max_rel_err": float(np.abs(a - b).max() / np.abs(b).max())}
    with open(os.path.join(RESULTS, "check.json"), "w") as f:
        json.dump(check, f, indent=2)

    def load(name):
        path = os.path.join(BUILD, name)
        return json.load(open(path)) if os.path.exists(path) else {}

    meta = load("meta.json")
    comp = {"cusadi2": {"codegen_s": load("codegen.json"), "nvcc_s": load("nvcc.json"),
                        "block_size": {dt: load(f"info_cusadi2_{dt}.json") for dt in DTYPES},
                        "functions": {k: {x: v[x] for x in ("n", "instructions", "nnz_out", "dense_out", "plan")}
                                      for k, v in meta.items()}},
            "frax": {"first_call_s": {dt: load(f"jit_frax_{dt}.json") for dt in DTYPES}}}
    with open(os.path.join(RESULTS, "compile.json"), "w") as f:
        json.dump(comp, f, indent=2)
    for dt in DTYPES:
        plot(rows, DT2FP[dt])
    summary(rows, check, comp)


SERIES = [("cusadi2", "#2a78d6"), ("frax", "#eb6834")]      # color = library
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot(rows, fp):
    """One figure per function: latency vs batch, one panel per robot. Color is
    the library; solid filled = data on the device, dashed hollow = with host transfer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    for kind, ftitle in FUNCS:
        sel = [r for r in rows if r["function"] == kind and r["dtype"] == fp]
        if not sel:
            continue
        fig, axes = plt.subplots(1, len(ROBOTS), figsize=(10.5, 4.4), sharey=True, facecolor=SURFACE)
        for ax, (robot, rtitle) in zip(axes, ROBOTS):
            for lib, color in SERIES:
                for io, ls, face in (("0", "-", color), ("1", "--", SURFACE)):
                    pts = sorted((int(r["batch"]), float(r["median_us"])) for r in sel
                                 if (r["robot"], r["library"], r["includes_io"]) == (robot, lib, io))
                    if pts:
                        ax.plot(*zip(*pts), ls, color=color, lw=2, marker="o", ms=6, markerfacecolor=face,
                                markeredgecolor=color, markeredgewidth=1.5)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(BATCHES, [str(b) for b in BATCHES])
            ax.minorticks_off()
            ax.set_facecolor(SURFACE)
            ax.grid(True, which="major", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=9, length=0)
            ax.set_title(rtitle, fontsize=10.5, color=INK, loc="left")
            ax.set_xlabel("batch size", fontsize=9.5, color=MUTED)
        axes[0].set_ylabel("median latency per batch (µs)", fontsize=9.5, color=MUTED)
        handles = [Line2D([], [], color=c, lw=2, marker="o", ms=6, label=lib) for lib, c in SERIES]
        handles += [Line2D([], [], color=MUTED, lw=2, ls="-", marker="o", ms=6, label="data stays on the GPU"),
                    Line2D([], [], color=MUTED, lw=2, ls="--", marker="o", ms=6, markerfacecolor=SURFACE,
                           label="with host↔device transfer")]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5, labelcolor=INK)
        fig.suptitle(f"{ftitle}: latency vs batch size ({fp}, RTX 4080)", fontsize=12, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))
        path = os.path.join(RESULTS, f"latency_{kind}_{fp}.png")
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print(f"  wrote {path}")


def summary(rows, check, comp):
    med = {(r["robot"], r["function"], r["library"], r["dtype"], int(r["batch"]), r["includes_io"]): float(r["median_us"])
           for r in rows}
    for fp in DT2FP.values():
        for io in ("0", "1"):
            print(f"\n=== {fp}, {'with host transfer' if io == '1' else 'data on the GPU'}: "
                  f"median us cusadi2 / frax (frax time / cusadi2 time) ===")
            for robot, _ in ROBOTS:
                for kind, _ in FUNCS:
                    cells = []
                    for B in BATCHES:
                        c, f = med.get((robot, kind, "cusadi2", fp, B, io)), med.get((robot, kind, "frax", fp, B, io))
                        cells.append(f"{c:.0f}/{f:.0f} ({f / c:.2f}x)" if c and f else
                                     f"{c:.0f}/-" if c else f"-/{f:.0f}" if f else "-")
                    print(f"  {robot + '_' + kind:11s} " + "  ".join(f"{x:>22s}" for x in cells))
    print("\n=== output check (max |cusadi2 - frax| / max |frax|) ===")
    for k, v in check.items():
        print(f"  {k:18s} {v['max_rel_err']:.2e}")
    print("\n=== compile ===")
    for dt in DTYPES:
        cg, jit = comp["cusadi2"]["codegen_s"].get(dt, {}), comp["frax"]["first_call_s"].get(dt, {})
        print(f"  {dt}: cusadi2 nvcc once {comp['cusadi2']['nvcc_s'].get(dt, float('nan')):.1f} s")
        for stem in cg:
            js = [v for v in jit.get(stem, {}).values() if v is not None]
            print(f"    {stem:11s} cusadi2 codegen {cg[stem]:6.2f} s   frax first call per batch shape: "
                  + (f"{min(js):.2f}-{max(js):.2f} s, {sum(js):.1f} s over {len(js)} shapes" if js else "-"))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    ap.add_argument("--plot", action="store_true", help="only rebuild the CSV, check, and figures from build/")
    ap.add_argument("--phase", choices=["codegen", "cusadi2", "frax"], help="(internal) one phase in this process")
    args = ap.parse_args()

    if args.phase == "codegen":
        return run_codegen(args.dtypes)
    if args.phase:
        return {"cusadi2": run_cusadi2, "frax": run_frax}[args.phase](args.dtypes[0])
    if not args.plot:
        def phase(name, dtypes, **env):
            subprocess.run([sys.executable, __file__, "--phase", name, "--dtypes", *dtypes],
                           env=dict(os.environ, **env), check=True)

        print("=== codegen: CasADi functions, plans, inputs ===", flush=True)
        phase("codegen", args.dtypes, JAX_PLATFORMS="cpu")       # frax only for the model data here
        nvcc = {dt: run_nvcc(dt) for dt in args.dtypes}
        with open(os.path.join(BUILD, "nvcc.json"), "w") as f:
            json.dump(nvcc, f, indent=2)
        for dt in args.dtypes:
            phase("cusadi2", [dt], CUDA_VISIBLE_DEVICES=DEVICE)
            phase("frax", [dt], CUDA_VISIBLE_DEVICES=DEVICE)
    report()


if __name__ == "__main__":
    main()
