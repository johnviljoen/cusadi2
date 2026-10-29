"""Native CPU baselines for the Menagerie campaign: pure Pinocchio and MuJoCo.

Two more CPU references next to the paper's CasADi-C-codegen one, both the
libraries' own compiled algorithms (no CasADi, no code generation), timed with
the SAME protocol as paper_results' CPU driver (n_items problems split
statically over OpenMP threads, each thread with its own workspace and output
buffer, median of n_timed repetitions) by two small drivers in this directory:

  pin_driver.cpp   Pinocchio 4.1 C++: framesForwardKinematics (fk),
                   computeJointJacobians + getFrameJacobian LOCAL_WORLD_ALIGNED
                   (fkjac), computeJointKinematicHessians +
                   getFrameKinematicHessian LOCAL_WORLD_ALIGNED (fkhess) of the
                   gated Pinocchio model (model.load_pinocchio, the one the
                   CasADi functions are built from), double and float
                   (Model::cast<float>()).
  mj_driver.c      libmujoco C (double only): mj_kinematics + mj_comPos (fk),
                   + mj_jacBody per named body (jac), and the MJX rollout series
                   (n_stages mj_step per instance, fk / jac extracted after every
                   step) on the campaign's MJCF with contact detection and the
                   constraint solver disabled, exactly as the MJX / warp
                   baselines run it (run._disable_contacts_and_constraints).

Both drivers have an idealized build (data in host memory) and a -DTRANSFER
build (inputs copied GPU->CPU before, outputs CPU->GPU after, inside the timed
region), as the CasADi CPU reference.

Per robot: build/<robot>/native/{model.pin, frames.txt, model.mjb, bodies.txt}
(prep) and native_gate.json (the gate: the drivers' check mode at random
configurations against the CasADi fk / dense fkjac / fkhess functions
(Pinocchio side, fkhess for nq == nv models, central differences of the
Jacobian along the tangent otherwise) and against MuJoCo's Python bindings
(MuJoCo side, fk / jac / an n_stages rollout)).
Per run k: results/<robot>/runs/<k>/pin_<dtype>.json (cpu_<dtype>.json's schema:
latency_us, seq_us / seq_transfer_us [func][threads][B]) and
results/<robot>/runs/<k>/mjcpu.json (the same for fk / fkjac, plus rollout_us /
rollout_transfer_us [func][threads][B], the rollout series: B instances x
n_stages steps).
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THREADS = [int(t) for t in os.environ.get("CUSADI2_NATIVE_THREADS", "16").split(",")]   # the paper's 16; latency_us is always 1 thread
PIN_MARCH = os.environ.get("CUSADI2_PIN_MARCH", "")    # e.g. "native": add -march=... to the Pinocchio build (default: none, measured faster)
PIN_EXPAND = os.environ.get("CUSADI2_PIN_EXPAND", "1") != "0"   # split JointModelComposite into one-dof joints (see prep_robot)
GATE_TOL = {"fk": 1e-8, "fkjac": 1e-8, "fkhess": 1e-6, "fkhess_fd": 1e-4, "mj": 1e-10}
VRAM_CAP = float(os.environ.get("CUSADI2_NATIVE_VRAM_GB", "12")) * 2**30   # transfer builds: device copies of inputs + outputs
STEMS = ["fk", "fkjac", "fkhess"]
MJ_SERIES = {"fk": "fk", "fkjac": "jac"}          # cpu-schema func -> driver series
BOOST_DEFS = ["-DBOOST_MPL_CFG_NO_PREPROCESSED_HEADERS", "-DBOOST_MPL_LIMIT_LIST_SIZE=30",
              "-DBOOST_MPL_LIMIT_VECTOR_SIZE=30", "-DBOOST_FUSION_INVOKE_MAX_ARITY=12"]


def _run():
    import run
    return run


def _sha(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def _prefix():
    return sys.prefix


def _mujoco_dir():
    import mujoco
    return os.path.dirname(mujoco.__file__)


# ---------------------------------------------------------------------------
# drivers: built once in build/native/ (sha-stamped)
# ---------------------------------------------------------------------------
def driver_paths():
    """{name: path}: pin_<dtype>, pin_<dtype>_transfer, mj, mj_transfer."""
    d = os.path.join(_run().BUILD, "native")
    out = {}
    for dtype in ("double", "float"):
        out[f"pin_{dtype}"] = os.path.join(d, f"pin_{dtype}")
        out[f"pin_{dtype}_transfer"] = os.path.join(d, f"pin_{dtype}_transfer")
    out["mj"] = os.path.join(d, "mj")
    out["mj_transfer"] = os.path.join(d, "mj_transfer")
    return out


def compile_drivers(force=False) -> dict:
    """g++/gcc the six driver binaries (cached by source + flags sha in
    build/native/stamp.json). Returns the compile record (flags, seconds, errors)."""
    run = _run()
    bench = run._bench()
    d = os.path.join(run.BUILD, "native")
    os.makedirs(d, exist_ok=True)
    P = _prefix()
    MJ = _mujoco_dir()
    cuda_inc, cuda_lib = bench.cuda_root()
    gxx = os.path.join(P, "bin", "g++") if os.path.exists(os.path.join(P, "bin", "g++")) else "g++"
    gcc = os.path.join(P, "bin", "gcc") if os.path.exists(os.path.join(P, "bin", "gcc")) else "gcc"
    pin_src, mj_src = os.path.join(HERE, "pin_driver.cpp"), os.path.join(HERE, "mj_driver.c")
    mj_lib = glob.glob(os.path.join(MJ, "libmujoco.so*"))[0]
    # Header-only instantiation of Pinocchio's templates (double and float alike),
    # -O3 without -march: measured on this machine, -march=native makes Pinocchio's
    # Eigen-heavy kinematics 20-45% SLOWER (fk/fkjac on a1 and flybody) while the
    # generic build matches conda-forge's prebuilt libpinocchio_default; the
    # faster one is the baseline. CUSADI2_PIN_MARCH=native overrides.
    pin_common = [gxx, "-O3", *([f"-march={PIN_MARCH}"] if PIN_MARCH else []), "-fopenmp", "-std=c++17", "-DNDEBUG",
                  *BOOST_DEFS, f"-I{P}/include", f"-I{P}/include/eigen3"]
    pin_link = [f"-L{P}/lib", "-lboost_serialization", f"-Wl,-rpath,{P}/lib"]
    lib_extra, lib_link = [], []
    mj_common = [gcc, "-O3", "-march=native", "-fopenmp", f"-I{MJ}/include"]
    mj_link = [f"-L{MJ}", f"-l:{os.path.basename(mj_lib)}", f"-Wl,-rpath,{MJ}", "-lm"]
    cuda = [f"-I{cuda_inc}"], [f"-L{cuda_lib}", "-lcudart", f"-Wl,-rpath,{cuda_lib}"]
    jobs = []
    paths = driver_paths()
    for dtype in ("double", "float"):
        extra = lib_extra if dtype == "double" else []
        xlink = lib_link if dtype == "double" else []
        base = pin_common + [f"-DSCALAR={dtype}"] + extra
        jobs.append((f"pin_{dtype}", base + [pin_src] + pin_link + xlink + ["-o", paths[f"pin_{dtype}"]]))
        jobs.append((f"pin_{dtype}_transfer", base + ["-DTRANSFER"] + cuda[0] + [pin_src] + pin_link + xlink + cuda[1]
                     + ["-o", paths[f"pin_{dtype}_transfer"]]))
    jobs.append(("mj", mj_common + [mj_src] + mj_link + ["-o", paths["mj"]]))
    jobs.append(("mj_transfer", mj_common + ["-DTRANSFER"] + cuda[0] + [mj_src] + mj_link + cuda[1] + ["-o", paths["mj_transfer"]]))
    stamp_path = os.path.join(d, "stamp.json")
    stamp = json.load(open(stamp_path)) if os.path.exists(stamp_path) else {}
    rec = {"pin_march": PIN_MARCH, "flags": {}, "seconds": {}, "errors": {}}
    procs = []
    for name, cmd in jobs:
        src = pin_src if name.startswith("pin") else mj_src
        h = _sha(src) + ":" + " ".join(cmd)
        rec["flags"][name] = " ".join(cmd)
        if not force and stamp.get(name) == h and os.path.exists(paths[name]):
            continue
        log = open(os.path.join(d, f"{name}.log"), "w")
        procs.append((name, h, time.perf_counter(), subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    for name, h, t0, p, log in procs:
        p.wait(); log.close()
        rec["seconds"][name] = round(time.perf_counter() - t0, 1)
        if p.returncode == 0:
            stamp[name] = h
        else:
            stamp.pop(name, None)
            rec["errors"][name] = open(os.path.join(d, f"{name}.log")).read()[-2000:]
    with open(stamp_path, "w") as f:
        json.dump(stamp, f, indent=2)
    with open(os.path.join(d, "compile.json"), "w") as f:
        json.dump(rec, f, indent=2)
    if procs:
        print("[native] drivers built: " + ", ".join(f"{n} {rec['seconds'][n]}s" for n, *_ in procs)
              + (f"  ERRORS: {list(rec['errors'])}" if rec["errors"] else ""), flush=True)
    return rec


# ---------------------------------------------------------------------------
# per-robot prep + gate (subprocess: run.py --native-prep --robot R)
# ---------------------------------------------------------------------------
def prep_dir(robot):
    return os.path.join(_run().BUILD, robot, "native")


def expand_composites(pm):
    """The same kinematic tree with every JointModelComposite replaced by its
    sub-joints in sequence (massless intermediate bodies; inertia and frames on
    the last one), the rest copied; (new model, number expanded). The
    configuration vector is unchanged (a composite's q is its sub-joints' q in
    order). Pinocchio's MJCF parser builds a composite for every body carrying
    several <joint>s; Pinocchio's kinematic Hessian (getFrameKinematicHessian)
    is WRONG for the pairs of columns inside one composite (verified against
    central differences: flybody, 1356 entries off by O(1), every other pair
    exact) and that is the only reason for the expansion -- a URDF-loaded
    Pinocchio model of the same robot has one-dof joints to begin with. The
    gate checks the expanded model's fk / fkjac against the CasADi functions
    built from the composite one (1e-8)."""
    import pinocchio as pin
    new = pin.Model()
    new.name = pm.name
    jmap = {0: 0}
    n_exp = 0
    for jid in range(1, pm.njoints):
        jm, parent, M, name = pm.joints[jid], jmap[pm.parents[jid]], pm.jointPlacements[jid], pm.names[jid]
        if jm.shortname() == "JointModelComposite":
            comp = jm.extract()
            subs, places = list(comp.joints), list(comp.jointPlacements)
            last = parent
            for s_, (sj, sp) in enumerate(zip(subs, places)):
                last = new.addJoint(last, sj, (M * sp) if s_ == 0 else sp,
                                    name if s_ == len(subs) - 1 else f"{name}__sub{s_}")
                if s_ < len(subs) - 1:
                    new.appendBodyToJoint(last, pin.Inertia.Zero(), pin.SE3.Identity())
            jmap[jid] = last
            n_exp += 1
        else:
            jmap[jid] = new.addJoint(parent, jm.extract(), M, name)
        new.appendBodyToJoint(jmap[jid], pm.inertias[jid], pin.SE3.Identity())
    for f in list(pm.frames)[1:]:
        new.addFrame(pin.Frame(f.name, jmap[f.parentJoint], f.parentFrame, f.placement, f.type), False)
    assert new.nq == pm.nq and new.nv == pm.nv, (new.nq, pm.nq, new.nv, pm.nv)
    return new, n_exp


def prep_robot(manifest, robot) -> dict:
    """Write the drivers' inputs: the gated Pinocchio model (binary; composites
    expanded, see expand_composites) + frame ids, the contact-free MuJoCo model
    (mjb) + named body ids. Returns the sizes."""
    import mujoco
    import model
    run = _run()
    xml = run.robot_xml(manifest, robot)
    d = prep_dir(robot)
    os.makedirs(d, exist_ok=True)
    m = model.load_mujoco(xml)
    pmodel = model.load_pinocchio(xml, m=m)
    names = model.body_names(m)
    idx = model.qpos_mj_to_pin_index(m, pmodel)
    corr = model.root_correction(m, pmodel, idx)
    if corr is not None:         # never active in the current Menagerie (gate: root_corr false everywhere)
        raise RuntimeError("root_correction active: the native drivers do not fold a root offset")
    n_exp = 0
    if PIN_EXPAND and any(j.shortname() == "JointModelComposite" for j in pmodel.joints):
        pmodel, n_exp = expand_composites(pmodel)
    fids = model.frame_ids(pmodel, names)
    pmodel.saveToBinary(os.path.join(d, "model.pin"))
    with open(os.path.join(d, "frames.txt"), "w") as f:
        f.write("\n".join(str(i) for i in fids) + "\n")
    mc = model.load_mujoco(xml)
    run._disable_contacts_and_constraints(mc)
    mujoco.mj_saveModel(mc, os.path.join(d, "model.mjb"), None)
    bids = model.named_body_ids(m)
    with open(os.path.join(d, "bodies.txt"), "w") as f:
        f.write("\n".join(str(i) for i in bids) + "\n")
    sizes = {"nq": int(pmodel.nq), "nv": int(pmodel.nv), "n_frames": len(fids), "mj_nq": int(m.nq), "mj_nv": int(m.nv),
             "n_bodies": len(bids), "n_stages": run.N_STAGES, "composites_expanded": n_exp}
    with open(os.path.join(d, "sizes.json"), "w") as f:
        json.dump(sizes, f, indent=2)
    return sizes


def _check(driver, model_path, list_path, series, q, extra=()):
    """Run a driver's check mode on one configuration -> output vector (double)."""
    d = os.path.dirname(model_path)
    qf, of = os.path.join(d, f"check_q_{os.getpid()}.bin"), os.path.join(d, f"check_out_{os.getpid()}.bin")
    np.asarray(q, dtype=np.float64).tofile(qf)
    subprocess.check_output([driver, model_path, list_path, series, "check", qf, of, *map(str, extra)],
                            stderr=subprocess.STDOUT)
    out = np.fromfile(of, dtype=np.float64)
    os.remove(qf); os.remove(of)
    return out


def gate_robot(manifest, robot, n_configs=3) -> dict:
    """The drivers' outputs at random configurations against the campaign's
    references. Pinocchio side: fk and dense fkjac against the CasADi functions
    the GPU kernels are built from (1e-8); fkhess against CasADi's d(vec J)/dq for
    nq == nv models (1e-6) or, for models with quaternion joints (nq > nv, where
    the CasADi Hessian is taken in q and Pinocchio's in the tangent), against
    central differences of the driver's own LOCAL_WORLD_ALIGNED Jacobian along
    the tangent (1e-4). MuJoCo side: fk / jac / rollout against MuJoCo's Python
    bindings on the same contact-free model (1e-10)."""
    import casadi as ca
    import mujoco
    import pinocchio as pin
    import model
    run = _run()
    xml = run.robot_xml(manifest, robot)
    d = prep_dir(robot)
    sizes = prep_robot(manifest, robot)
    drv = driver_paths()
    m = model.load_mujoco(xml)
    pmodel = model.load_pinocchio(xml, m=m)          # the CasADi functions' model (composites intact)
    names = model.body_names(m)
    idx = model.qpos_mj_to_pin_index(m, pmodel)
    fids = model.frame_ids(pmodel, names)
    bids = model.named_body_ids(m)
    nq, nv, nf = pmodel.nq, pmodel.nv, len(fids)
    fk = model.build_fk(pmodel, names)
    jac = model.build_fkjac(pmodel, names, sparse=False)
    qs = ca.SX.sym("q", nq)
    hess = ca.Function("hess_dense", [qs], [ca.jacobian(jac(qs), qs)]) if nq == nv else None
    pin_path, fr_path = os.path.join(d, "model.pin"), os.path.join(d, "frames.txt")
    pdrv = pin.Model()                                 # the driver's model (what is timed)
    pdrv.loadFromBinary(pin_path)
    fids_drv = model.frame_ids(pdrv, names)
    mjb_path, bd_path = os.path.join(d, "model.mjb"), os.path.join(d, "bodies.txt")
    mc = mujoco.MjModel.from_binary_path(mjb_path)
    dc = mujoco.MjData(mc)
    rng = np.random.default_rng(1)
    err = {k: 0.0 for k in ("fk", "fkjac", "fkhess", "fk_float", "fkjac_float", "fkhess_float",
                            "mj_fk", "mj_jac", "mj_fk_rollout", "mj_jac_rollout")}
    hess_ref = "casadi" if nq == nv else "finite_differences"
    data = pdrv.createData()

    def jac_pin(q):          # the reference the fd Hessian differentiates: the driver model's own LWA frame Jacobians
        pin.computeJointJacobians(pdrv, data, q)
        pin.updateFramePlacements(pdrv, data)
        return np.concatenate([pin.getFrameJacobian(pdrv, data, i, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).reshape(-1, order="F")
                               for i in fids_drv])

    for _ in range(n_configs):
        q_mj = model.sample_qpos(m, rng)
        q = q_mj[idx]
        for dtype in ("double", "float"):
            sfx = "" if dtype == "double" else "_float"
            o = _check(drv[f"pin_{dtype}"], pin_path, fr_path, "fk", q)
            err["fk" + sfx] = max(err["fk" + sfx], float(np.abs(o - np.asarray(fk(q)).ravel()).max()))
            o = _check(drv[f"pin_{dtype}"], pin_path, fr_path, "fkjac", q)
            err["fkjac" + sfx] = max(err["fkjac" + sfx], float(np.abs(o - np.asarray(jac(q)).ravel(order="F")).max()))
            o = _check(drv[f"pin_{dtype}"], pin_path, fr_path, "fkhess", q)
            H = o.reshape(nf, nv, nv, 6).transpose(0, 3, 2, 1)       # col-major 6 x nv x nv per frame -> [f, i, j, k]
            if hess is not None:
                ref = np.asarray(ca.DM(hess(q))).reshape(nf, nv, 6, nq).transpose(0, 2, 1, 3)   # rows i + 6 j per frame, cols k
            else:
                h = 1e-6
                cols = []
                for k in range(nv):
                    e = np.zeros(nv); e[k] = h
                    cols.append((jac_pin(pin.integrate(pdrv, q, e)) - jac_pin(pin.integrate(pdrv, q, -e))) / (2 * h))
                ref = np.stack(cols, -1).reshape(nf, nv, 6, nv).transpose(0, 2, 1, 3)
            err["fkhess" + sfx] = max(err["fkhess" + sfx], float(np.abs(H - ref).max()))
        # MuJoCo side
        o = _check(drv["mj"], mjb_path, bd_path, "fk", q_mj, extra=[run.N_STAGES])
        dc.qpos[:] = q_mj
        mujoco.mj_kinematics(mc, dc); mujoco.mj_comPos(mc, dc)
        ref = np.concatenate([np.concatenate([dc.xpos[b], dc.xmat[b]]) for b in bids])
        err["mj_fk"] = max(err["mj_fk"], float(np.abs(o - ref).max()))
        o = _check(drv["mj"], mjb_path, bd_path, "jac", q_mj, extra=[run.N_STAGES])
        jp, jr = np.zeros((3, mc.nv)), np.zeros((3, mc.nv))
        ref = []
        for b in bids:
            mujoco.mj_jacBody(mc, dc, jp, jr, b)
            ref += [jp.ravel().copy(), jr.ravel().copy()]
        err["mj_jac"] = max(err["mj_jac"], float(np.abs(o - np.concatenate(ref)).max()))
        for series, key in (("fk_rollout", "mj_fk_rollout"), ("jac_rollout", "mj_jac_rollout")):
            o = _check(drv["mj"], mjb_path, bd_path, series, q_mj, extra=[run.N_STAGES])
            mujoco.mj_resetData(mc, dc)
            dc.qpos[:] = q_mj
            ref = []
            for _k in range(run.N_STAGES):
                mujoco.mj_step(mc, dc)
                if series == "fk_rollout":
                    ref += [np.concatenate([dc.xpos[b].copy(), dc.xmat[b].copy()]) for b in bids]
                else:
                    for b in bids:
                        mujoco.mj_jacBody(mc, dc, jp, jr, b)
                        ref += [jp.ravel().copy(), jr.ravel().copy()]
            ref = np.concatenate(ref)
            scale = max(1.0, float(np.abs(ref).max()))
            err[key] = max(err[key], float(np.abs(o - ref).max()) / scale)
    tol = {"fk": GATE_TOL["fk"], "fkjac": GATE_TOL["fkjac"], "fkhess": GATE_TOL["fkhess" if nq == nv else "fkhess_fd"],
           "mj_fk": GATE_TOL["mj"], "mj_jac": GATE_TOL["mj"], "mj_fk_rollout": 1e-8, "mj_jac_rollout": 1e-8}
    bad = [k for k, t in tol.items() if not (err[k] <= t)]
    # float is informative only (fp32 kinematics is ~1e-7 off), not gated. A
    # Pinocchio function that fails is recorded in "failed" and not timed; the
    # robot is skipped only when fk / fkjac or the MuJoCo side fail.
    failed = {k: f"max err {err[k]:.2e} > {tol[k]:g} ({hess_ref})" for k in bad if k == "fkhess"}
    fatal = [k for k in bad if k != "fkhess"]
    out = {"status": "skip" if fatal else "pass", "reason": (f"native_gate: {fatal} exceed tolerance" if fatal else None),
           "failed": failed, "err": err, "tol": tol, "hess_reference": hess_ref, "n_configs": n_configs, **sizes}
    with open(os.path.join(d, "native_gate.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# sweeps (in-process, sequential; the drivers are subprocesses)
# ---------------------------------------------------------------------------
def _point(exe, args, n_items, threads, need_host, need_dev, skipped, label, extra=()):
    """Median us of one driver point, or None (recorded in skipped) when the
    buffers would not fit (host: run.CPU_MEM_FRACTION of MemAvailable; device:
    VRAM_CAP) or the driver fails."""
    run = _run()
    avail = run._mem_available_bytes()
    if need_host > run.CPU_MEM_FRACTION * avail:
        skipped[label] = f"host buffers {need_host / 2**30:.1f} GB > {run.CPU_MEM_FRACTION:.0%} of available {avail / 2**30:.1f} GB"
        return None
    if need_dev > VRAM_CAP:
        skipped[label] = f"device buffers {need_dev / 2**30:.1f} GB > cap {VRAM_CAP / 2**30:.0f} GB"
        return None
    reps = max(5, run.CPU_N_TIMED // n_items)
    try:
        return float(subprocess.check_output([exe, *args, str(max(1, reps // 10)), str(reps), str(n_items), str(threads),
                                              *map(str, extra)], stderr=subprocess.DEVNULL))
    except (subprocess.CalledProcessError, ValueError) as e:
        skipped[label] = repr(e)[:200]
        return None


def _cpu_name():
    return next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")


def run_pin(robot, dtype, k, force=False) -> dict:
    """Run k of the pure-Pinocchio sweep -> results/<robot>/runs/<k>/pin_<dtype>.json."""
    run = _run()
    d = prep_dir(robot)
    sizes = json.load(open(os.path.join(d, "sizes.json")))
    gate = json.load(open(os.path.join(d, "native_gate.json")))
    failed = gate.get("failed") or {}
    rundir = os.path.join(run.RESULTS, robot, "runs", str(k))
    os.makedirs(rundir, exist_ok=True)
    out = os.path.join(rundir, f"pin_{dtype}.json")
    if not force and os.path.exists(out):
        return {"status": "pass", "cached": True}
    drv = driver_paths()
    exe, exe_tr = drv[f"pin_{dtype}"], drv[f"pin_{dtype}_transfer"]
    args = [os.path.join(d, "model.pin"), os.path.join(d, "frames.txt")]
    nq, nv, nf = sizes["nq"], sizes["nv"], sizes["n_frames"]
    nnz = {"fk": (nq, 12 * nf), "fkjac": (nq, 6 * nv * nf), "fkhess": (nq, 6 * nv * nv * nf)}
    bytes_per = 4 if dtype == "float" else 8
    t0 = time.perf_counter()
    lat, seq, seq_tr, skipped = {}, {}, {}, {}
    for stem in STEMS:
        if stem in failed:
            continue
        n_in, n_out = nnz[stem]

        def pt(transfer, n_items, thr):
            host = n_items * n_in * bytes_per + (n_items if transfer else thr) * n_out * bytes_per
            dev = n_items * (n_in + n_out) * bytes_per if transfer else 0
            return _point(exe_tr if transfer else exe, args + [stem], n_items, thr, host, dev, skipped,
                          f"{stem}{'_transfer' if transfer else ''} B*N={n_items} thr={thr}")
        lat[stem] = pt(False, 1, 1)
        seq[stem] = {str(t): {str(B): pt(False, B * run.N_STAGES, t) for B in run.BATCHES} for t in THREADS}
        seq_tr[stem] = {str(t): {str(B): pt(True, B * run.N_STAGES, t) for B in run.BATCHES} for t in THREADS}
        top = str(run.BATCHES[-1])
        fmt = lambda dd: "  ".join(f"{t}thr " + (f"{dd[str(t)][top] / 1e3:7.1f} ms" if dd[str(t)][top] is not None else "   null")
                                   for t in THREADS)
        print(f"  [pin {robot} {dtype}] {stem}: {lat[stem] if lat[stem] is None else round(lat[stem], 2)} us/problem"
              f"   B={top}: {fmt(seq[stem])}   transfer: {fmt(seq_tr[stem])}", flush=True)
    import pinocchio
    with open(out + ".tmp", "w") as f:
        json.dump({"robot": robot, "dtype": dtype, "backend": "pinocchio", "pinocchio": pinocchio.__version__,
                   "pin_march": PIN_MARCH, "composites_expanded": sizes.get("composites_expanded", 0),
                   "cpu": _cpu_name(), "threads": os.cpu_count(),
                   "n_stages": run.N_STAGES, "cpu_threads": THREADS, "n_timed": run.CPU_N_TIMED,
                   "sizes": {s: {"nnz_in": a, "nnz_out": b} for s, (a, b) in nnz.items()},
                   "hess_convention": "getFrameKinematicHessian LOCAL_WORLD_ALIGNED, dense 6 x nv x nv per frame",
                   "latency_us": lat, "seq_us": seq, "seq_transfer_us": seq_tr, "skipped": skipped,
                   "failed": failed}, f, indent=2)
    os.replace(out + ".tmp", out)
    rec = {"status": "pass", "seconds": round(time.perf_counter() - t0, 2)}
    if skipped:
        rec["skipped"] = len(skipped)
    if failed:
        rec["failed"] = list(failed)
    return rec


def run_mjcpu(robot, k, force=False) -> dict:
    """Run k of the MuJoCo CPU sweep -> results/<robot>/runs/<k>/mjcpu.json:
    seq_us / seq_transfer_us [fk|fkjac][threads][B] (B*N configurations) and
    rollout_us / rollout_transfer_us [fk|fkjac][threads][B] (B instances x N steps)."""
    run = _run()
    d = prep_dir(robot)
    sizes = json.load(open(os.path.join(d, "sizes.json")))
    rundir = os.path.join(run.RESULTS, robot, "runs", str(k))
    os.makedirs(rundir, exist_ok=True)
    out = os.path.join(rundir, "mjcpu.json")
    if not force and os.path.exists(out):
        return {"status": "pass", "cached": True}
    drv = driver_paths()
    exe, exe_tr = drv["mj"], drv["mj_transfer"]
    args = [os.path.join(d, "model.mjb"), os.path.join(d, "bodies.txt")]
    nq, nv, nb, N = sizes["mj_nq"], sizes["mj_nv"], sizes["n_bodies"], run.N_STAGES
    block = {"fk": 12 * nb, "fkjac": 6 * nv * nb}
    t0 = time.perf_counter()
    lat, seq, seq_tr, roll, roll_tr, skipped = {}, {}, {}, {}, {}, {}
    for stem in ("fk", "fkjac"):
        def pt(transfer, series, n_items, thr, n_out):
            host = n_items * nq * 8 + (n_items if transfer else thr) * n_out * 8
            dev = n_items * (nq + n_out) * 8 if transfer else 0
            return _point(exe_tr if transfer else exe, args + [series], n_items, thr, host, dev, skipped,
                          f"{series}{'_transfer' if transfer else ''} items={n_items} thr={thr}", extra=[N])
        s = MJ_SERIES[stem]
        lat[stem] = pt(False, s, 1, 1, block[stem])
        seq[stem] = {str(t): {str(B): pt(False, s, B * N, t, block[stem]) for B in run.BATCHES} for t in THREADS}
        seq_tr[stem] = {str(t): {str(B): pt(True, s, B * N, t, block[stem]) for B in run.BATCHES} for t in THREADS}
        roll[stem] = {str(t): {str(B): pt(False, s + "_rollout", B, t, block[stem] * N) for B in run.BATCHES} for t in THREADS}
        roll_tr[stem] = {str(t): {str(B): pt(True, s + "_rollout", B, t, block[stem] * N) for B in run.BATCHES} for t in THREADS}
        top = str(run.BATCHES[-1])
        fmt = lambda dd: "  ".join(f"{t}thr " + (f"{dd[str(t)][top] / 1e3:7.1f} ms" if dd[str(t)][top] is not None else "   null")
                                   for t in THREADS)
        print(f"  [mjcpu {robot}] {stem}: {lat[stem] if lat[stem] is None else round(lat[stem], 2)} us/problem"
              f"   B={top}: {fmt(seq[stem])}   transfer: {fmt(seq_tr[stem])}   rollout: {fmt(roll[stem])}", flush=True)
    import mujoco
    with open(out + ".tmp", "w") as f:
        json.dump({"robot": robot, "dtype": "double", "backend": "mujoco", "mujoco": mujoco.__version__,
                   "cpu": _cpu_name(), "threads": os.cpu_count(), "n_stages": N, "cpu_threads": THREADS,
                   "n_timed": run.CPU_N_TIMED,
                   "sizes": {s: {"nnz_in": nq, "nnz_out": b} for s, b in block.items()},
                   "latency_us": lat, "seq_us": seq, "seq_transfer_us": seq_tr,
                   "rollout_us": roll, "rollout_transfer_us": roll_tr, "skipped": skipped}, f, indent=2)
    os.replace(out + ".tmp", out)
    rec = {"status": "pass", "seconds": round(time.perf_counter() - t0, 2)}
    if skipped:
        rec["skipped"] = len(skipped)
    return rec


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def native_phase(manifest, statuses, runs, force, jobs):
    """Drivers once; per robot (gate-passed): prep + native gate in a subprocess
    pool, then sequential sweeps run k = 0..runs-1 of pin_double, pin_float and
    mjcpu (status stages native_gate, and per-run records in status "native_runs")."""
    import concurrent.futures
    run = _run()
    rec = compile_drivers(force=False)
    if rec["errors"]:
        raise RuntimeError(f"native driver build failed: {list(rec['errors'])}")
    self_py = [sys.executable, os.path.join(HERE, "run.py")]
    order = sorted((r for r, st in statuses.items() if st.gate_ok()), key=lambda r: statuses[r].d["gate"]["nv"])

    def gate(robot):
        gpath = os.path.join(prep_dir(robot), "native_gate.json")
        if not force and os.path.exists(gpath):
            try:
                return json.load(open(gpath)), True
            except Exception:
                pass
        log = os.path.join(run.BUILD, robot, "native_gate.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        r = run._run_stage(self_py + ["--native-prep", "--robot", robot], run.STAGE_TIMEOUT_S["gate"], log=log)
        if r["status"] != "pass":
            return {"status": "skip", "reason": f"native_gate_{r['status']}: {r.get('reason')}"}, False
        return json.load(open(gpath)), False

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for robot, (g, cached) in zip(order, pool.map(gate, order)):
            st = statuses[robot]
            st.stage("native_gate", {"status": g["status"], "reason": g.get("reason"), "cached": cached,
                                     "failed": g.get("failed"), "err": g.get("err"), "hess_reference": g.get("hess_reference"),
                                     "composites_expanded": g.get("composites_expanded")})
            print(f"[native_gate] {robot}: {g['status']}" + (" (cached)" if cached else "")
                  + (f" ({g.get('reason')})" if g["status"] != "pass" else
                     "  max err fk %.1e jac %.1e hess %.1e (%s) | mj fk %.1e jac %.1e roll %.1e" % (
                         g["err"]["fk"], g["err"]["fkjac"], g["err"]["fkhess"], g["hess_reference"],
                         g["err"]["mj_fk"], g["err"]["mj_jac"], max(g["err"]["mj_fk_rollout"], g["err"]["mj_jac_rollout"]))
                     + (f"  FAILED {list(g['failed'])}" if g.get("failed") else "")
                     + (f"  ({g['composites_expanded']} composite joint(s) expanded)" if g.get("composites_expanded") else "")),
                  flush=True)
    for k in range(runs):
        print(f"=== native CPU run {k + 1}/{runs} ===", flush=True)
        for robot in order:
            st = statuses[robot]
            if st.d["stages"].get("native_gate", {}).get("status") != "pass":
                continue
            for stage in ("pin_double", "pin_float", "mjcpu"):
                if stage == "mjcpu":
                    r = run_mjcpu(robot, k, force=force)
                else:
                    r = run_pin(robot, stage.split("_")[1], k, force=force)
                st.d.setdefault("native_runs", {}).setdefault(str(k), {})[stage] = r
                st.save()
                print(f"[{stage} {k}] {robot}: {r['status']}" + (" (cached)" if r.get("cached") else f" {r.get('seconds', 0):.0f}s")
                      + (f" ({r['skipped']} point(s) skipped)" if r.get("skipped") else ""), flush=True)
    for robot in order:
        st = statuses[robot]
        st.d["n_runs"] = runs
        st.d["complete"] = True
        st.d["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st.save()


# ---------------------------------------------------------------------------
# cross-robot summary: results_native/native_summary.csv (+ .md)
# ---------------------------------------------------------------------------
SUMMARY_BATCHES = (256, 16384)


def _mean_runs(resdir, name, key, func, threads):
    """{B: mean us over runs} of one cpu-schema json's key[func][threads]."""
    acc = {}
    for path in sorted(glob.glob(os.path.join(resdir, "runs", "*", name))):
        try:
            d = (json.load(open(path)).get(key) or {}).get(func) or {}
        except Exception:
            continue
        for b, v in (d.get(str(threads)) or {}).items():
            if v is not None:
                acc.setdefault(int(b), []).append(v)
    return {b: sum(v) / len(v) for b, v in acc.items()}


def _cusadi2_best(resdir, dtype, func):
    """{B: best mean wall us over configs} from the Python-hosted sweeps (as summarize.py)."""
    import csv
    acc = {}
    for path in sorted(glob.glob(os.path.join(resdir, "runs", "*", f"latency_py_{dtype}.csv"))):
        for r in csv.DictReader(open(path)):
            if r["func"] == func:
                try:
                    acc.setdefault((int(r["batch"]), r["config"]), []).append(float(r["wall_us"]))
                except ValueError:
                    pass
    best = {}
    for (b, cfg), v in acc.items():
        m = sum(v) / len(v)
        if b not in best or m < best[b]:
            best[b] = m
    return best


def summarize(native_results, results, threads=16):
    """One row per robot x func x dtype: the 16-thread wall (us) of pure Pinocchio
    (idealized / transfer), MuJoCo CPU (idealized / transfer; fk, fkjac only; fp64
    reported on both dtype rows), the paper's CasADi CPU reference (idealized /
    transfer, from `results`) and cusadi2's best config, at B in SUMMARY_BATCHES,
    plus the speedups of cusadi2 over each; and a markdown table of the medians."""
    import csv
    run = _run()
    robots = sorted(r for r in os.listdir(native_results) if os.path.isdir(os.path.join(native_results, r, "runs")))
    cols = ["robot", "nq", "nv", "n_frames", "composites_expanded", "func", "dtype", "B",
            "pin_us", "pin_transfer_us", "mjcpu_us", "mjcpu_transfer_us", "cpu_casadi_us", "cpu_casadi_transfer_us",
            "cusadi2_best_us", "speedup_pin", "speedup_pin_transfer", "speedup_mjcpu", "speedup_mjcpu_transfer",
            "speedup_cpu_casadi", "speedup_cpu_casadi_transfer"]
    rows = []
    for robot in robots:
        nd, rd = os.path.join(native_results, robot), os.path.join(results, robot)
        try:
            sz = json.load(open(os.path.join(prep_dir(robot), "sizes.json")))
        except Exception:
            sz = {}
        for func in STEMS:
            for dtype in ("double", "float"):
                pin = _mean_runs(nd, f"pin_{dtype}.json", "seq_us", func, threads)
                pin_tr = _mean_runs(nd, f"pin_{dtype}.json", "seq_transfer_us", func, threads)
                mj = _mean_runs(nd, "mjcpu.json", "seq_us", func, threads)
                mj_tr = _mean_runs(nd, "mjcpu.json", "seq_transfer_us", func, threads)
                cpu = _mean_runs(rd, f"cpu_{dtype}.json", "seq_us", func, threads)
                cpu_tr = _mean_runs(rd, f"cpu_{dtype}.json", "seq_transfer_us", func, threads)
                cus = _cusadi2_best(rd, dtype, func)
                for B in SUMMARY_BATCHES:
                    if B not in pin and B not in mj:
                        continue
                    c = cus.get(B)
                    sp = lambda v: round(v / c, 3) if (v is not None and c) else None
                    rows.append({"robot": robot, "nq": sz.get("nq"), "nv": sz.get("nv"), "n_frames": sz.get("n_frames"),
                                 "composites_expanded": sz.get("composites_expanded"), "func": func, "dtype": dtype, "B": B,
                                 "pin_us": pin.get(B), "pin_transfer_us": pin_tr.get(B), "mjcpu_us": mj.get(B),
                                 "mjcpu_transfer_us": mj_tr.get(B), "cpu_casadi_us": cpu.get(B), "cpu_casadi_transfer_us": cpu_tr.get(B),
                                 "cusadi2_best_us": c, "speedup_pin": sp(pin.get(B)), "speedup_pin_transfer": sp(pin_tr.get(B)),
                                 "speedup_mjcpu": sp(mj.get(B)), "speedup_mjcpu_transfer": sp(mj_tr.get(B)),
                                 "speedup_cpu_casadi": sp(cpu.get(B)), "speedup_cpu_casadi_transfer": sp(cpu_tr.get(B))})
    path = os.path.join(native_results, "native_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else (round(v, 3) if isinstance(v, float) else v)) for k, v in r.items()})
    # markdown: median speedup across robots (and the n) per func x dtype x B
    lines = ["# Native CPU baselines: median speedup of cusadi2 (best config) across robots",
             f"({threads} threads; n = robots with both numbers; MuJoCo CPU is fp64 on both dtype rows, no Hessian)", "",
             "| func | dtype | B | Pinocchio ideal. | Pinocchio + transfer | MuJoCo CPU ideal. | MuJoCo CPU + transfer | CasADi CPU ideal. | CasADi CPU + transfer |",
             "|---|---|---|---|---|---|---|---|---|"]
    for func in STEMS:
        for dtype in ("double", "float"):
            for B in SUMMARY_BATCHES:
                sel = [r for r in rows if r["func"] == func and r["dtype"] == dtype and r["B"] == B]
                cell = []
                for k in ("speedup_pin", "speedup_pin_transfer", "speedup_mjcpu", "speedup_mjcpu_transfer",
                          "speedup_cpu_casadi", "speedup_cpu_casadi_transfer"):
                    v = [r[k] for r in sel if r[k] is not None]
                    cell.append(f"{np.median(v):.2f}x (n={len(v)})" if v else "—")
                lines.append(f"| {func} | {dtype} | {B} | " + " | ".join(cell) + " |")
    with open(os.path.join(native_results, "native_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path} ({len(rows)} rows) and native_summary.md")
    return rows
