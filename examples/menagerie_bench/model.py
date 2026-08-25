"""Shared model loading, MJCF<->Pinocchio consistency gate, and CasADi function
builders for the per-robot Menagerie benchmark.

Both backends load the SAME MJCF XML: MuJoCo via `MjModel.from_xml_path`,
Pinocchio via `pin.buildModelFromMJCF` (Pinocchio >= 4.0). Base handling comes
from the MJCF itself — a `<freejoint>` becomes a Pinocchio FreeFlyer, no root
joint is ever forced.

Pinocchio 4.0's MJCF parser has three defects that `load_pinocchio` repairs
(see the README's "Pinocchio MJCF parser defects" section); the load path is

    flatten <include>s (A) -> wrap multi-tree <worldbody> (B)
    -> pin.buildModelFromMJCF -> repair dropped joint anchors (C)

MuJoCo always loads the ORIGINAL xml; only Pinocchio sees the rewritten temp
file, and the repairs are numeric no-ops wherever parsing was already correct.
The gate's random-config FK cross-check verifies the repaired model per robot.

Frame set: the NAMED MuJoCo bodies excluding 'world', in MuJoCo body order,
matched 1:1 by name to Pinocchio BODY frames. Unnamed bodies (mj_id2name ->
None; e.g. google_barkour_vb's decorative torso weld, umi_gripper's articulated
root) cannot be matched by name, so they are excluded from the OUTPUT set on
both sides — but they are still computed internally by both backends (MuJoCo
kinematics always covers every body; Pinocchio keeps unnamed bodies in the
kinematic tree under auto-generated names like 'Bis'/'torsoBis'). The dof gate
check (nq/nv equality) runs BEFORE frame matching, so an unnamed articulated
body whose joints were dropped by either parser cannot silently shrink the
workload. FK = stacked world placements (translation + column-major-flattened
rotation); Jacobian = stacked 6xnv LOCAL_WORLD_ALIGNED frame Jacobians.

qpos convention gotcha: MuJoCo stores free-joint quaternions as (w,x,y,z) and
ball-joint quaternions as (w,x,y,z), while Pinocchio's FreeFlyer / Spherical
joints expect (x,y,z,w). `qpos_mj_to_pin_index` builds a per-joint index map so
q_pin = q_mj[idx]; the numerical FK cross-check in the gate validates the whole
mapping end to end.

Root-placement gotcha: Pinocchio's MJCF parser drops the world placement of the
root body (e.g. ur5e's `<body name="base" quat="0 0 0 -1">`), leaving the whole
Pinocchio model expressed in the root-body frame instead of MuJoCo's world.
`root_correction` derives the constant rigid offset numerically; the builders
fold it into the CasADi functions (a constant rotation/translation per frame) so
the benchmark workload stays "world placements", identical to what MJX computes.
The gate's random-config cross-check runs on the corrected functions, so any
mismatch that is NOT a constant offset still fails the gate.
"""
import os
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

FK_TOL = 1e-8          # fp64 CasADi eval on CPU vs MuJoCo double kinematics
N_CHECK_CONFIGS = 5    # random configurations for the FK cross-check

PIN_WRAPPER_NAME = "__pin_multitree_root__"

# permissive: single/double quotes, arbitrary whitespace, self-closing or not
_INCLUDE_RE = re.compile(
    r"""<include\s+file\s*=\s*["']([^"']+)["']\s*(?:/\s*>|>\s*</include\s*>)""")
_ROOT_RE = re.compile(r"<mujoco(?:include)?[^>]*>(.*)</mujoco(?:include)?\s*>", re.S)


def load_mujoco(xml_path):
    import mujoco
    return mujoco.MjModel.from_xml_path(xml_path)


# ---------------------------------------------------------------------------
# Pinocchio load path: three repairs for Pinocchio 4.0's MJCF parser
# ---------------------------------------------------------------------------
def _flatten_includes(xml_path):
    """(A) Recursively inline MJCF <include> elements, which Pinocchio's parser
    silently ignores. MuJoCo include semantics = literal textual insertion of
    the included file's root-element contents, paths relative to the including
    file — so the numerics of the flattened text are exactly the original's.
    Returns (text, n_includes_inlined); byte-identical text for include-free
    files (n=0)."""
    base = os.path.dirname(os.path.abspath(xml_path))
    text = open(xml_path).read()
    count = 0

    def repl(mo):
        nonlocal count
        count += 1
        inner, n = _flatten_includes(os.path.join(base, mo.group(1)))
        count += n
        return _ROOT_RE.search(inner).group(1)

    return _INCLUDE_RE.sub(repl, text), count


def preprocess_mjcf_for_pinocchio(xml_path, rec=None):
    """Rewrite the MJCF for the Pinocchio parse ONLY (MuJoCo always loads the
    original): (A) inline <include>s, then (B) wrap all <worldbody> root bodies
    in a single static dummy body PIN_WRAPPER_NAME — Pinocchio builds only the
    FIRST kinematic tree under <worldbody>, silently dropping sibling roots.
    The wrapper carries an explicit zero <inertial> so models compiled with
    inertiafromgeom="false" (e.g. crazyflie) still parse. Body/joint names,
    defaults/childclass and document order are untouched. Returns the temp-file
    path (or the original path when nothing needed rewriting). `rec`, if a
    dict, records {"includes_flattened": bool, "multitree_wrapped": bool} —
    multitree_wrapped flags the >1-root case where the wrapper is load-bearing
    (it is applied whenever any root body exists; as a side effect it also
    preserves the root body's world placement, so root_correction is usually
    inactive but kept as a verified fallback)."""
    rec = {} if rec is None else rec
    text, n_inc = _flatten_includes(xml_path)
    rec["includes_flattened"] = n_inc > 0
    root = ET.fromstring(text)
    wb = root.find("worldbody")
    bodies = wb.findall("body") if wb is not None else []
    rec["multitree_wrapped"] = len(bodies) > 1
    if bodies:
        if any(b.get("name") == PIN_WRAPPER_NAME for b in wb.iter("body")):
            raise RuntimeError(f"body name collision with {PIN_WRAPPER_NAME}")
        wrapper = ET.Element("body", {"name": PIN_WRAPPER_NAME})
        ET.SubElement(wrapper, "inertial",
                      {"pos": "0 0 0", "mass": "0", "diaginertia": "0 0 0"})
        for b in bodies:
            wb.remove(b)
            wrapper.append(b)
        wb.append(wrapper)
    elif n_inc == 0:
        return xml_path                       # byte-identical no-op
    out = os.path.join(tempfile.mkdtemp(prefix="pin_mjcf_"),
                       os.path.basename(xml_path))
    ET.ElementTree(root).write(out)
    return out


def fix_dropped_joint_anchors(m, pmodel):
    """(C) Post-load repair: Pinocchio's MJCF parser drops a joint anchor
    pos="..." inherited from a <default> class (it honors inline pos only),
    e.g. robotiq_2f85, stanford_tidybot. Ground truth is MuJoCo's compiled
    m.jnt_pos. For each non-free joint matched by name with nonzero anchor p,
    the anchor Pinocchio actually applied is read off the child BODY frame
    placement (applied = -placement.translation); when delta = p - applied is
    nonzero the joint frame is moved by delta and everything attached to it is
    conjugated back so only the anchor changes:

        jointPlacements[jid] *= T(delta)
        inertias[jid]         = T(-delta).act(inertias[jid])
        frames with parentJoint == jid (except the JOINT frame):
                              placement = T(-delta) * placement
        child joints cj with parents[cj] == jid:
                              jointPlacements[cj] = T(-delta) * jointPlacements[cj]

    Delta-detection makes this a no-op wherever parsing was correct. Returns
    the number of joints repaired."""
    import mujoco
    import pinocchio as pin

    # child BODY frame of each joint: frames are in tree order, so the first
    # BODY frame carried by a joint is its own child body (joints composed
    # inside one MJCF body carry a BODY frame only on the last of the chain)
    body_frame_id = {}
    for fi in range(pmodel.nframes):
        f = pmodel.frames[fi]
        if f.type == pin.FrameType.BODY and int(f.parentJoint) not in body_frame_id:
            body_frame_id[int(f.parentJoint)] = fi

    n_fixed = 0
    for jid in range(1, pmodel.njoints):
        mj_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, pmodel.names[jid])
        if mj_j < 0 or m.jnt_type[mj_j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        p = np.asarray(m.jnt_pos[mj_j], dtype=float)
        if np.abs(p).max() == 0.0:
            continue
        fi = body_frame_id.get(jid)
        if fi is None:
            continue
        applied = -np.asarray(pmodel.frames[fi].placement.translation)
        delta = p - applied
        if np.abs(delta).max() < 1e-12:
            continue                          # parser applied the anchor correctly
        T = pin.SE3(np.eye(3), delta)
        Tinv = pin.SE3(np.eye(3), -delta)
        pmodel.jointPlacements[jid] = pmodel.jointPlacements[jid] * T
        pmodel.inertias[jid] = Tinv.act(pmodel.inertias[jid])
        for k in range(pmodel.nframes):
            fr = pmodel.frames[k]
            if int(fr.parentJoint) == jid and fr.type != pin.FrameType.JOINT:
                fr.placement = Tinv * fr.placement
        for cj in range(1, pmodel.njoints):
            if int(pmodel.parents[cj]) == jid:
                pmodel.jointPlacements[cj] = Tinv * pmodel.jointPlacements[cj]
        n_fixed += 1
    return n_fixed


def load_pinocchio(xml_path, m=None, preproc=None):
    """Build the Pinocchio model from the SAME MJCF via the repaired load path

        flatten <include>s (A) -> wrap multi-tree <worldbody> (B)
        -> pin.buildModelFromMJCF -> fix dropped joint anchors (C)

    (the single Pinocchio load path — the gate and the benchmark codegen both
    go through here, so what is benchmarked is what was gated). `m` is the
    MuJoCo model of the same xml, loaded on demand when omitted (ground truth
    for repair C). `preproc`, when a dict is passed, is filled with
    {"includes_flattened": bool, "multitree_wrapped": bool, "anchors_fixed": int}.
    """
    import pinocchio as pin
    rec = preproc if preproc is not None else {}
    pmodel = pin.buildModelFromMJCF(preprocess_mjcf_for_pinocchio(xml_path, rec))
    if m is None:
        m = load_mujoco(xml_path)
    rec["anchors_fixed"] = fix_dropped_joint_anchors(m, pmodel)
    return pmodel


def named_body_ids(m):
    """MuJoCo ids of the NAMED bodies excluding 'world' (body 0), in body order.

    Unnamed bodies (mj_id2name -> None) cannot be matched by name to Pinocchio
    frames, so they are excluded from the common OUTPUT frame set — both
    backends still compute them internally as part of the kinematic tree."""
    import mujoco
    return [i for i in range(1, m.nbody)
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)]


def body_names(m):
    """Names of the NAMED MuJoCo bodies excluding 'world', in MuJoCo body order
    (the common frame set — see named_body_ids for the unnamed-body policy)."""
    import mujoco
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in named_body_ids(m)]


def missing_body_frames(pmodel, names):
    """Body names with no matching Pinocchio BODY frame."""
    import pinocchio as pin
    return [n for n in names if not pmodel.existFrame(n, pin.FrameType.BODY)]


def frame_ids(pmodel, names):
    import pinocchio as pin
    return [pmodel.getFrameId(n, pin.FrameType.BODY) for n in names]


# ---------------------------------------------------------------------------
# qpos mapping MuJoCo -> Pinocchio
# ---------------------------------------------------------------------------
def qpos_mj_to_pin_index(m, pmodel):
    """Index array idx (length nq) such that q_pin = q_mj[idx].

    Walks the MuJoCo joints in document order (which is also qpos-address order)
    and assigns consecutive Pinocchio qpos slots, reordering each free/ball
    quaternion from MuJoCo's (w,x,y,z) to Pinocchio's (x,y,z,w). Assumes both
    parsers enumerate the MJCF joints in the same (document) order — the gate's
    numerical FK cross-check verifies this assumption per robot.
    """
    import mujoco
    assert pmodel.nq == m.nq and pmodel.nv == m.nv
    idx = np.empty(m.nq, dtype=int)
    pos = 0
    for j in range(m.njnt):
        adr = int(m.jnt_qposadr[j])
        t = m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            idx[pos:pos + 3] = np.arange(adr, adr + 3)               # x,y,z
            idx[pos + 3:pos + 7] = [adr + 4, adr + 5, adr + 6, adr + 3]  # wxyz -> xyzw
            pos += 7
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            idx[pos:pos + 4] = [adr + 1, adr + 2, adr + 3, adr + 0]  # wxyz -> xyzw
            pos += 4
        else:                                                        # hinge / slide
            idx[pos] = adr
            pos += 1
    assert pos == m.nq, f"qpos walk covered {pos} of {m.nq} slots"
    return idx


def sample_qpos(m, rng, scale=0.3):
    """One random MuJoCo qpos around qpos0 with normalized quaternions."""
    import mujoco
    q = np.asarray(m.qpos0).copy() + rng.normal(0.0, scale, size=m.nq)
    for j in range(m.njnt):
        adr = int(m.jnt_qposadr[j])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            q[adr + 3:adr + 7] /= np.linalg.norm(q[adr + 3:adr + 7])
        elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
            q[adr:adr + 4] /= np.linalg.norm(q[adr:adr + 4])
    return q


def root_correction(m, pmodel, idx):
    """Constant world-frame offset (R, t) with M_world = (R, t) . M_pin, or None.

    Pinocchio's MJCF parser ignores the root body's world placement; when that
    placement is non-identity every Pinocchio placement is off by the SAME rigid
    transform. Derived numerically at qpos0 from the first real body:
    R = R_mj . R_pin^T, t = t_mj - R . t_pin. Returns None when the offset is
    identity (the common case: floating bases, arms mounted at the origin).
    Whether the offset really is constant is verified by the gate's
    random-configuration FK cross-check on the corrected functions.
    """
    import mujoco
    import pinocchio as pin
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    mujoco.mj_kinematics(m, d)
    data = pmodel.createData()
    pin.framesForwardKinematics(pmodel, data, np.asarray(m.qpos0)[idx])
    bid = named_body_ids(m)[0]                # first NAMED body (body 1 may be unnamed)
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
    fid = pmodel.getFrameId(name, pin.FrameType.BODY)
    M = data.oMf[fid]
    R = d.xmat[bid].reshape(3, 3) @ M.rotation.T
    t = d.xpos[bid] - R @ M.translation
    if np.abs(R - np.eye(3)).max() < 1e-10 and np.abs(t).max() < 1e-10:
        return None
    return R, t


# ---------------------------------------------------------------------------
# CasADi function builders (cusadi2 side)
# ---------------------------------------------------------------------------
def _casadi_model(pmodel):
    import casadi as ca
    import pinocchio.casadi as cpin
    cmodel = cpin.Model(pmodel)
    cdata = cmodel.createData()
    q = ca.SX.sym("q", cmodel.nq)
    return cmodel, cdata, q


def build_fk(pmodel, names, name="fk", root_corr=None):
    """Forward kinematics f(q) -> stacked world placement (translation +
    column-major-flattened rotation) of every common link frame. `root_corr`
    (see root_correction) is folded in as a constant transform per frame."""
    import casadi as ca
    import pinocchio.casadi as cpin
    cmodel, cdata, q = _casadi_model(pmodel)
    cpin.framesForwardKinematics(cmodel, cdata, q)
    placements = []
    for i in frame_ids(pmodel, names):
        t, R = cdata.oMf[i].translation, cdata.oMf[i].rotation
        if root_corr is not None:
            Rc, tc = ca.DM(root_corr[0]), ca.DM(root_corr[1])
            t, R = ca.mtimes(Rc, t) + tc, ca.mtimes(Rc, R)
        placements.append(ca.vertcat(t, ca.reshape(R, 9, 1)))
    return ca.Function(name, [q], [ca.vertcat(*placements)])


def build_fkjac(pmodel, names, name="fkjac", root_corr=None):
    """Frame Jacobians f(q) -> stacked 6xnv LOCAL_WORLD_ALIGNED Jacobian of
    every common link frame. Under the constant world remap `root_corr` both the
    linear and angular rows rotate by R (the translation part cancels because
    LOCAL_WORLD_ALIGNED is taken at the frame's own origin)."""
    import casadi as ca
    import pinocchio as pin
    import pinocchio.casadi as cpin
    cmodel, cdata, q = _casadi_model(pmodel)
    cpin.forwardKinematics(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)
    cpin.computeJointJacobians(cmodel, cdata, q)
    jacs = [cpin.getFrameJacobian(cmodel, cdata, i, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            for i in frame_ids(pmodel, names)]
    if root_corr is not None:
        import casadi as _ca
        Rc = _ca.DM(root_corr[0])
        jacs = [_ca.vertcat(_ca.mtimes(Rc, J[:3, :]), _ca.mtimes(Rc, J[3:, :]))
                for J in jacs]
    flat = [ca.reshape(J, J.shape[0] * J.shape[1], 1) for J in jacs]
    return ca.Function(name, [q], [ca.vertcat(*flat)])


def unroll(f, n_stages):
    """Stamp per-stage f n_stages times into one function — the no-stage-axis
    (original-cusadi) horizon. Each stamp gets its own fresh inputs; the outputs
    are concatenated stage-major, so the unrolled DAG is ~n_stages x as wide/deep
    and its peak live set is ~n_stages x the per-stage plan's."""
    import casadi as ca
    stamp_ins = [[ca.SX.sym(f"in{a}_{k}", f.sparsity_in(a)) for a in range(f.n_in())]
                 for k in range(n_stages)]
    outs = []
    for k in range(n_stages):
        res = f(*stamp_ins[k])
        res = res if isinstance(res, (list, tuple)) else [res]
        outs += [ca.reshape(o, o.numel(), 1) for o in res]
    flat_ins = [stamp_ins[k][a] for k in range(n_stages) for a in range(f.n_in())]
    return ca.Function(f.name() + "_unr", flat_ins, [ca.vertcat(*outs)])


# ---------------------------------------------------------------------------
# Compatibility gate
# ---------------------------------------------------------------------------
def run_gate(xml_path):
    """Per-robot compatibility gate. Returns a JSON-serializable dict:

        {"status": "pass"|"skip", "reason": <str|None>,
         "nq":, "nv":, "n_bodies":, "n_bodies_unnamed":, "floating":,
         "fk_max_err": <float|None>}

    n_bodies counts the NAMED bodies (the common frame set); n_bodies_unnamed
    counts the unnamed non-world bodies excluded from that set.

    Every failure is a recorded skip reason, never an exception (parser errors
    are caught and reported).
    """
    import mujoco

    out = {"status": "skip", "reason": None, "nq": None, "nv": None,
           "n_bodies": None, "n_bodies_unnamed": None, "floating": None,
           "fk_max_err": None, "root_corr": False, "preproc": None}

    # MuJoCo side (the reference; if even this fails, the model is unusable)
    try:
        m = load_mujoco(xml_path)
    except Exception as e:
        out["reason"] = f"mujoco_parse_failed: {e!r}"[:500]
        return out
    out["nq"], out["nv"] = int(m.nq), int(m.nv)
    body_ids = named_body_ids(m)              # the common OUTPUT frame set
    out["n_bodies"] = len(body_ids)
    out["n_bodies_unnamed"] = int(m.nbody - 1 - len(body_ids))
    out["floating"] = bool(any(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                               for j in range(m.njnt)))

    # 5. zero-dof models (static props, cameras, ...) have no kinematics to batch
    if m.nq == 0:
        out["reason"] = "zero_dof"
        return out

    # 1. Pinocchio parses the same MJCF (through the repaired load path; the
    #    preproc record shows which parser repairs were load-bearing)
    preproc = {}
    try:
        pmodel = load_pinocchio(xml_path, m=m, preproc=preproc)
    except Exception as e:
        out["preproc"] = preproc or None
        out["reason"] = f"pin_mjcf_parse_failed: {e!r}"[:500]
        return out
    out["preproc"] = preproc

    # 2. exact nq/nv agreement (catches silently-partial Pinocchio builds,
    #    e.g. aloha where only one arm survives). Runs BEFORE frame matching so
    #    that an unnamed ARTICULATED body (e.g. umi_gripper's root) whose joints
    #    were dropped by either parser cannot silently shrink the workload —
    #    unnamed bodies are excluded from the frame set below, but their dofs
    #    still count here.
    if pmodel.nq != m.nq or pmodel.nv != m.nv:
        out["reason"] = (f"dof_mismatch: pin nq={pmodel.nq} nv={pmodel.nv} "
                         f"vs mujoco nq={m.nq} nv={m.nv}")
        return out

    # 3. every NAMED MuJoCo body has a Pinocchio BODY frame of the same name
    names = body_names(m)
    missing = missing_body_frames(pmodel, names)
    if missing:
        out["reason"] = f"missing_frames: {missing[:10]}"
        return out

    # 4. numerical FK cross-check at random configurations (on the corrected
    #    functions — see root_correction)
    try:
        idx = qpos_mj_to_pin_index(m, pmodel)
        corr = root_correction(m, pmodel, idx)
        out["root_corr"] = corr is not None
        fk = build_fk(pmodel, names, root_corr=corr)
        d = mujoco.MjData(m)
        rng = np.random.default_rng(0)
        max_err = 0.0
        for _ in range(N_CHECK_CONFIGS):
            q_mj = sample_qpos(m, rng)
            d.qpos[:] = q_mj
            mujoco.mj_kinematics(m, d)
            y = np.asarray(fk(q_mj[idx])).ravel()
            for k, bid in enumerate(body_ids):   # NAMED bodies only, body order
                seg = y[12 * k:12 * (k + 1)]
                t_pin, R_pin = seg[:3], seg[3:].reshape(3, 3, order="F")
                t_mj, R_mj = d.xpos[bid], d.xmat[bid].reshape(3, 3)
                max_err = max(max_err,
                              float(np.abs(t_pin - t_mj).max()),
                              float(np.abs(R_pin - R_mj).max()))
        out["fk_max_err"] = max_err
        if max_err > FK_TOL:
            out["reason"] = f"fk_mismatch: max_err={max_err:.3e} (tol {FK_TOL:g})"
            return out
    except Exception as e:
        out["reason"] = f"fk_check_failed: {e!r}"[:500]
        return out

    out["status"] = "pass"
    return out
