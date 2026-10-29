"""MuJoCo-faithful simulation step on cusadi2, build-time side.

A MuJoCo step is split at the line between fixed and data-dependent structure:

    smooth(qpos, qvel, ctrl) -> M, qfrc_smooth, geom_pos, geom_rot, cdof          (cusadi2 plan)
    [contact kernels: narrowphase, constraint rows, Newton solve -> qfrc_constraint] (cusadi2_contact.h)
    integrate(qpos, qvel, M, qfrc_smooth, qfrc_constraint) -> qpos', qvel', qacc    (cusadi2 plan)

The two CasADi functions are built here from the MJCF through Pinocchio (casadi bindings) and
carry MuJoCo's own conventions on their interface: `qpos`/`qvel`/`ctrl` are MjData's arrays
(free-joint quaternion wxyz, base linear velocity in the world frame, angular in the body frame),
`M` is MuJoCo's mass matrix in those coordinates (its nonzeros only: the tree's dof-ancestor pattern), `qfrc_smooth` is
qfrc_passive + qfrc_actuator - qfrc_bias, and `integrate` is MuJoCo's Euler or implicitfast
advance. `cdof` is MuJoCo's per-dof motion axis in world coordinates (6 x nv: linear velocity
induced at the base position qpos[0:3], then angular), `geom_pos`/`geom_rot` the world placement of
every collision geom named by a <pair>; the contact kernels form each contact row from the dofs in
the geom body's ancestry: jacp = cdof_lin + cdof_ang x (p - qpos[0:3]), exactly as MuJoCo's mj_jac.

Supported MJCF feature set (anything else raises at load):
  explicit <pair> contacts between a plane and a sphere or capsule, condim 1 or 3, pyramidal cone;
  hinge / slide joint limits; dof frictionloss; joint damping and armature; <position> actuators
  (kp / kv, ctrlrange, actuatorfrcrange); Euler or implicitfast; no equality constraints,
  tendons, gravity compensation, fluid forces or actuator dynamics.

Outputs of the plans are in CasADi's column-major nonzero order (dense except M); `cdof` is 6 x nv so the six
numbers of dof k are contiguous. `write_model_header` emits the constants the kernels need as a C++ header.
"""
import numpy as np

SPHERE, CAPSULE, PLANE = 2, 3, 0
_INT_EULER, _INT_IMPLICITFAST = 0, 3


class Unsupported(ValueError):
    pass


def _quat_to_mat(q):
    """wxyz quaternion to rotation matrix (casadi or numpy)."""
    import casadi as ca
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
                      ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
                      ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def _quat_mul(a, b):
    import casadi as ca
    return ca.vertcat(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])


def _quat_integrate(q, w, dt):
    """MuJoCo's mju_quatIntegrate: q (wxyz) rotated by the body-frame angular velocity w for dt."""
    import casadi as ca
    n = ca.sqrt(ca.dot(w, w))
    half = 0.5 * n * dt
    s = ca.if_else(n > 1e-10, ca.sin(half) / n, 0.5 * dt)
    qn = _quat_mul(q, ca.vertcat(ca.cos(half), s * w))
    return qn / ca.sqrt(ca.dot(qn, qn))


def _ldl_solve(A, b, n, order=None):
    """x = A^-1 b for symmetric positive definite A (casadi SX, n x n), LDL^T without pivoting in the
    elimination order `order` (default 0..n-1). CasADi drops products with structural zeros, so on a
    mass matrix with the kinematic tree's sparsity, eliminating the leaf dofs first (MuJoCo's order,
    n-1 down to 0) keeps the factor as sparse as the matrix and the expression depth at the tree depth."""
    import casadi as ca
    order = list(range(n)) if order is None else list(order)
    pos = {d: k for k, d in enumerate(order)}
    L = {}
    D = [None] * n
    for jj, j in enumerate(order):
        s = A[j, j]
        for k in order[:jj]:
            if (j, k) in L:
                s = s - L[(j, k)] * L[(j, k)] * D[k]
        D[j] = s
        for i in order[jj + 1:]:
            t = A[i, j]
            for k in order[:jj]:
                if (i, k) in L and (j, k) in L:
                    t = t - L[(i, k)] * L[(j, k)] * D[k]
            if not (isinstance(t, ca.SX) and t.is_zero()):
                L[(i, j)] = t / D[j]
    y = {}
    for ii, i in enumerate(order):
        s = b[i]
        for k in order[:ii]:
            if (i, k) in L:
                s = s - L[(i, k)] * y[k]
        y[i] = s
    x = {}
    for ii in reversed(range(n)):
        i = order[ii]
        s = y[i] / D[i]
        for k in order[ii + 1:]:
            if (k, i) in L:
                s = s - L[(k, i)] * x[k]
        x[i] = s
    return ca.vertcat(*[x[i] for i in range(n)])


class Model:
    """The MuJoCo model, the Pinocchio model built from the same MJCF, and the constants the step needs.

    `pmodel`: a Pinocchio model of the same MJCF (e.g. the Menagerie loader's repaired one); by default
    pin.buildModelFromMJCF on the file. The qpos order is assumed to walk the MJCF joints in document
    order on both sides (free quaternion reordered wxyz -> xyzw); `check()` verifies that numerically.
    """

    def __init__(self, xml_path, pmodel=None):
        import mujoco
        import pinocchio as pin
        self.xml_path = xml_path
        m = mujoco.MjModel.from_xml_path(xml_path)
        self.m = m
        if pmodel is None:
            pmodel = pin.buildModelFromMJCF(xml_path)
        self.pm = pmodel
        self.nq, self.nv, self.nu = int(m.nq), int(m.nv), int(m.nu)
        if pmodel.nq != m.nq or pmodel.nv != m.nv:
            raise Unsupported(f"Pinocchio nq/nv {pmodel.nq}/{pmodel.nv} != MuJoCo {m.nq}/{m.nv}")
        FREE, HINGE, SLIDE = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        if m.njnt < 1 or int(m.jnt_type[0]) != FREE or m.jnt_qposadr[0] != 0:
            raise Unsupported("expected a free joint first (floating base)")
        for j in range(1, m.njnt):
            if int(m.jnt_type[j]) not in (HINGE, SLIDE):
                raise Unsupported(f"joint {j}: only hinge / slide joints after the free joint")
        # qpos map: q_pin = q_mj[idx]
        idx = np.arange(m.nq)
        idx[3:7] = [4, 5, 6, 3]
        self.idx = idx
        # options
        o = m.opt
        if o.integrator not in (_INT_EULER, _INT_IMPLICITFAST):
            raise Unsupported("integrator must be Euler or implicitfast")
        if o.cone != 0:
            raise Unsupported("only the pyramidal cone")
        if o.solver != 2:
            raise Unsupported("only the Newton solver")
        if m.neq or m.ntendon or m.nflex or m.nplugin:
            raise Unsupported("equality constraints, tendons, flex and plugins are not supported")
        if o.density or o.viscosity or np.any(o.wind):
            raise Unsupported("fluid forces are not supported")
        if np.any(m.body_gravcomp):
            raise Unsupported("gravity compensation is not supported")
        self.dt = float(o.timestep)
        self.gravity = np.asarray(o.gravity, dtype=float)
        self.integrator = "implicitfast" if o.integrator == _INT_IMPLICITFAST else "euler"
        self.eulerdamp = not bool(o.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        self.opt = dict(iterations=int(o.iterations), ls_iterations=int(o.ls_iterations), tolerance=float(o.tolerance),
                        ls_tolerance=float(o.ls_tolerance), impratio=float(o.impratio), timestep=self.dt,
                        integrator=self.integrator)
        # dofs
        self.armature = np.asarray(m.dof_armature, dtype=float).copy()
        self.damping = np.asarray(m.dof_damping, dtype=float).copy()
        self.frictionloss = np.asarray(m.dof_frictionloss, dtype=float).copy()
        self.dof_solref = np.asarray(m.dof_solref, dtype=float).copy()
        self.dof_solimp = np.asarray(m.dof_solimp, dtype=float).copy()
        self.dof_invweight0 = np.asarray(m.dof_invweight0, dtype=float).copy()
        # joint limits (hinge / slide)
        self.limits = []
        for j in range(1, m.njnt):
            if m.jnt_limited[j]:
                self.limits.append(dict(joint=j, dof=int(m.jnt_dofadr[j]), qpos=int(m.jnt_qposadr[j]),
                                        range=np.asarray(m.jnt_range[j], dtype=float).copy(), margin=float(m.jnt_margin[j]),
                                        solref=np.asarray(m.jnt_solref[j], dtype=float).copy(),
                                        solimp=np.asarray(m.jnt_solimp[j], dtype=float).copy()))
        # actuators: <position> = gain kp on ctrl, bias [0, -kp, -kv], no dynamics
        self.actuators = []
        for i in range(m.nu):
            if m.actuator_dyntype[i] != 0 or m.actuator_gaintype[i] != 0 or m.actuator_biastype[i] != 1:
                raise Unsupported(f"actuator {i}: only stateless position / affine actuators")
            if m.actuator_trntype[i] != 0:
                raise Unsupported(f"actuator {i}: only joint transmission")
            if m.actuator_gear[i][0] != 1 or np.any(m.actuator_gear[i][1:]):
                raise Unsupported(f"actuator {i}: gear must be 1")
            j = int(m.actuator_trnid[i][0])
            bias = np.asarray(m.actuator_biasprm[i], dtype=float)
            if bias[0] != 0:
                raise Unsupported(f"actuator {i}: constant bias term not supported")
            self.actuators.append(dict(joint=j, dof=int(m.jnt_dofadr[j]), qpos=int(m.jnt_qposadr[j]),
                                       gain=float(m.actuator_gainprm[i][0]), bias_q=float(bias[1]), bias_v=float(bias[2]),
                                       ctrllimited=bool(m.actuator_ctrllimited[i]),
                                       ctrlrange=np.asarray(m.actuator_ctrlrange[i], dtype=float).copy(),
                                       forcelimited=bool(m.actuator_forcelimited[i]),
                                       forcerange=np.asarray(m.actuator_forcerange[i], dtype=float).copy()))
        self.jnt_actfrclimited = np.asarray(m.jnt_actfrclimited, dtype=bool).copy()
        self.jnt_actfrcrange = np.asarray(m.jnt_actfrcrange, dtype=float).copy()
        # contact pairs: plane vs sphere / capsule, from the explicit <pair> list
        self.plane = None
        self.geoms = []          # one entry per pair (a geom may appear once per plane)
        for p in range(m.npair):
            g1, g2 = int(m.pair_geom1[p]), int(m.pair_geom2[p])
            if m.geom_type[g1] == PLANE and m.geom_type[g2] != PLANE:
                g1, g2 = g2, g1
            if m.geom_type[g2] != PLANE or m.geom_type[g1] not in (SPHERE, CAPSULE):
                raise Unsupported(f"pair {p}: only plane vs sphere / capsule pairs")
            if m.geom_bodyid[g2] != 0:
                raise Unsupported(f"pair {p}: the plane must be static (in the world body)")
            plane = (np.asarray(m.geom_pos[g2], dtype=float).copy(), _quat_to_mat_np(np.asarray(m.geom_quat[g2], dtype=float))[:, 2].copy())
            if self.plane is None:
                self.plane = plane
            elif not (np.allclose(plane[0], self.plane[0]) and np.allclose(plane[1], self.plane[1])):
                raise Unsupported("all pairs must share one plane")
            if int(m.pair_dim[p]) not in (1, 3):
                raise Unsupported(f"pair {p}: condim must be 1 or 3")
            bid = int(m.geom_bodyid[g1])
            self.geoms.append(dict(geom=g1, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1), type=int(m.geom_type[g1]),
                                   size=np.asarray(m.geom_size[g1], dtype=float).copy(), body=bid,
                                   body_name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid),
                                   pos=np.asarray(m.geom_pos[g1], dtype=float).copy(), quat=np.asarray(m.geom_quat[g1], dtype=float).copy(),
                                   condim=int(m.pair_dim[p]), solref=np.asarray(m.pair_solref[p], dtype=float).copy(),
                                   solreffriction=np.asarray(m.pair_solreffriction[p], dtype=float).copy(),
                                   solimp=np.asarray(m.pair_solimp[p], dtype=float).copy(),
                                   friction=np.asarray(m.pair_friction[p], dtype=float).copy(),
                                   margin=float(m.pair_margin[p]), gap=float(m.pair_gap[p])))
        # Pinocchio side: gravity, armature, one operational frame per pair geom
        import pinocchio as pin
        pmodel.gravity.linear = self.gravity
        pmodel.armature = self.armature.copy()
        self.geom_fids = []
        for g in self.geoms:
            fid = pmodel.getFrameId(g["body_name"], pin.FrameType.BODY)
            if fid >= pmodel.nframes:
                raise Unsupported(f"body {g['body_name']} has no Pinocchio BODY frame")
            b = pmodel.frames[fid]
            place = b.placement * pin.SE3(_quat_to_mat_np(g["quat"]), g["pos"])
            self.geom_fids.append(pmodel.addFrame(pin.Frame(f"geom_{g['name']}", b.parentJoint, place, pin.FrameType.OP_FRAME)))

    @property
    def n_geom(self):
        return len(self.geoms)

    def body_dofs(self, body):
        """The dofs that move `body`: its own and every ancestor body's, ascending (MuJoCo's dof ancestry)."""
        m = self.m
        dofs = []
        b = body
        while b > 0:
            dofs += list(range(int(m.body_dofadr[b]), int(m.body_dofadr[b]) + int(m.body_dofnum[b])))
            b = int(m.body_parentid[b])
        return sorted(dofs)

    def write_model_header(self, path, name="model"):
        """C++ header with every constant cusadi2_contact.h needs for this model: a struct
        `cusadi2_contact::<name>` of constexpr scalars (sizes, solver options, dt) and accessor functions
        over `__constant__` device arrays (the plane, the pair geoms with their dof ancestry, the
        frictionloss dofs, the limited joints, M's sparsity). Include from exactly one translation unit.
        Call after build_smooth (for M's sparsity)."""
        import mujoco
        m = self.m
        if not hasattr(self, "M_sparsity"):
            raise RuntimeError("write_model_header needs build_smooth first")
        nv = self.nv
        sp = self.M_sparsity
        rows, cols = sp.get_triplet()
        fl = [d for d in range(nv) if self.frictionloss[d] > 0]
        geoms = self.geoms
        max_chain = max(len(self.body_dofs(g["body"])) for g in geoms) if geoms else 1
        n_points = sum(2 if g["type"] == CAPSULE else 1 for g in geoms)
        refsafe = not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE)
        islands = not bool(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ISLAND)   # MuJoCo >= 3.3 solves per island by default
        lim = self.limits
        chains = [(self.body_dofs(g["body"]) + [-1] * max_chain)[:max_chain] for g in geoms]
        # (NAME, ctype, inner dim or None, values)
        arrays = [
            ("PLANE_POS", "double", None, self.plane[0]), ("PLANE_NORMAL", "double", None, self.plane[1]),
            ("M_ROW", "int", None, rows), ("M_COL", "int", None, cols),
            ("GEOM_TYPE", "int", None, [g["type"] for g in geoms]),
            ("GEOM_RADIUS", "double", None, [g["size"][0] for g in geoms]),
            ("GEOM_HALFLEN", "double", None, [g["size"][1] for g in geoms]),
            ("GEOM_CONDIM", "int", None, [g["condim"] for g in geoms]),
            ("GEOM_NCHAIN", "int", None, [len(self.body_dofs(g["body"])) for g in geoms]),
            ("GEOM_CHAIN", "int", max_chain, chains),
            ("GEOM_INVWEIGHT", "double", None, [m.body_invweight0[g["body"]][0] for g in geoms]),
            ("GEOM_SOLREF", "double", 2, [g["solref"] for g in geoms]),
            ("GEOM_SOLIMP", "double", 5, [g["solimp"] for g in geoms]),
            ("GEOM_FRICTION", "double", 5, [g["friction"] for g in geoms]),
            ("GEOM_MARGIN", "double", None, [g["margin"] for g in geoms]),
            ("GEOM_GAP", "double", None, [g["gap"] for g in geoms]),
            ("FRICTION_DOF", "int", None, fl),
            ("FRICTION_LOSS", "double", None, [self.frictionloss[d] for d in fl]),
            ("FRICTION_INVWEIGHT", "double", None, [self.dof_invweight0[d] for d in fl]),
            ("FRICTION_SOLREF", "double", 2, [self.dof_solref[d] for d in fl]),
            ("FRICTION_SOLIMP", "double", 5, [self.dof_solimp[d] for d in fl]),
            ("LIMIT_DOF", "int", None, [l["dof"] for l in lim]),
            ("LIMIT_QPOS", "int", None, [l["qpos"] for l in lim]),
            ("LIMIT_RANGE", "double", 2, [l["range"] for l in lim]),
            ("LIMIT_MARGIN", "double", None, [l["margin"] for l in lim]),
            ("LIMIT_INVWEIGHT", "double", None, [self.dof_invweight0[l["dof"]] for l in lim]),
            ("LIMIT_SOLREF", "double", 2, [l["solref"] for l in lim]),
            ("LIMIT_SOLIMP", "double", 5, [l["solimp"] for l in lim]),
        ]

        def lit(v):
            v = v.item() if hasattr(v, "item") else v
            return repr(float(v)) if isinstance(v, float) else repr(int(v))

        L = [f"// generated by cusadi2.contact.Model.write_model_header from {self.xml_path}", "#pragma once",
             "#include <cuda_runtime.h>", "namespace cusadi2_contact {", f"namespace {name}_const {{"]
        for nm, ct, inner, vals in arrays:
            vals = list(vals)
            n = max(len(vals), 1)
            if inner is None:
                body = ", ".join(lit(v) for v in vals) or "0"
                L.append(f"__constant__ {ct} {nm}[{n}] = {{{body}}};")
            else:
                body = ", ".join("{" + ", ".join(lit(v) for v in row) + "}" for row in vals) or "{0}"
                L.append(f"__constant__ {ct} {nm}[{n}][{inner}] = {{{body}}};")
        L += [f"}}  // namespace {name}_const", f"struct {name} {{",
              f"    static constexpr int NQ = {self.nq}, NV = {nv}, NU = {self.nu}, NGEOM = {len(geoms)}, NPOINT = {n_points}, "
              f"NFRICTION = {len(fl)}, NLIMIT = {len(lim)}, MAX_CHAIN = {max_chain}, M_NNZ = {sp.nnz()};",
              f"    static constexpr int ITERATIONS = {self.opt['iterations']}, LS_ITERATIONS = {self.opt['ls_iterations']};",
              f"    static constexpr double DT = {self.dt!r}, TOLERANCE = {self.opt['tolerance']!r}, LS_TOLERANCE = {self.opt['ls_tolerance']!r}, "
              f"IMPRATIO = {self.opt['impratio']!r}, MEANINERTIA = {float(m.stat.meaninertia)!r};",
              f"    static constexpr bool REFSAFE = {'true' if refsafe else 'false'};",
              f"    static constexpr bool ISLANDS = {'true' if islands else 'false'};   // tolerance scale 1 / trace(M) (one island) instead of 1 / (meaninertia nv)"]
        for nm, ct, inner, vals in arrays:
            ret = f"const {ct}*" if inner is not None else ct
            L.append(f"    __device__ static __forceinline__ {ret} {nm.lower()}(int i) {{ return {name}_const::{nm}[i]; }}")
        L += ["};", "}  // namespace cusadi2_contact", ""]
        with open(path, "w") as f:
            f.write("\n".join(L))
        return path


    # ------------------------------------------------------------------ smooth
    def _base_rotation(self, qpos):
        return _quat_to_mat(qpos[3:7])

    def _v_pin(self, qvel, R):
        import casadi as ca
        return ca.vertcat(ca.mtimes(R.T, qvel[0:3]), qvel[3:])

    def build_smooth(self, geoms=True, name="smooth"):
        """smooth(qpos, qvel, ctrl) -> [M (nv x nv, sparse), qfrc_smooth (nv), geom_pos (3 x ng), geom_rot (9 x ng),
        cdof (6 x nv)]; with geoms=False only the first two (the contact-free step)."""
        import casadi as ca
        import pinocchio as pin
        import pinocchio.casadi as cpin
        cm = cpin.Model(self.pm)
        cd = cm.createData()
        nq, nv, nu = self.nq, self.nv, self.nu
        qpos, qvel, ctrl = ca.SX.sym("qpos", nq), ca.SX.sym("qvel", nv), ca.SX.sym("ctrl", nu)
        R = self._base_rotation(qpos)
        q = qpos[self.idx]
        v = self._v_pin(qvel, R)
        Mp = cpin.crba(cm, cd, q)
        Mp = ca.triu(Mp, True) + ca.triu(Mp, False).T            # crba fills the upper triangle
        cp = cpin.nonLinearEffects(cm, cd, q, v)                 # C v + g, Pinocchio coordinates
        # MuJoCo coordinates: v_pin = T v_mj with T = diag(R^T, I); a_pin = T a_mj + delta,
        # delta = [-w x v_loc, 0, ...]. Power invariance gives tau_mj = T^T tau_pin, hence
        # M_mj = T^T M_pin T and bias_mj = T^T (bias_pin + M_pin delta).
        w, vloc = qvel[3:6], v[0:3]
        delta = ca.vertcat(-ca.cross(w, vloc), ca.SX.zeros(nv - 3))
        bias_p = cp + ca.mtimes(Mp, delta)
        T = ca.SX.eye(nv)
        T[0:3, 0:3] = R.T
        M = ca.sparsify(ca.mtimes(ca.mtimes(T.T, Mp), T))      # the kinematic tree's dof-ancestor pattern
        bias = ca.mtimes(T.T, bias_p)
        # passive: joint damping
        qfrc_passive = -self.damping * qvel
        # actuators
        qfrc_act = ca.SX.zeros(nv)
        for i, a in enumerate(self.actuators):
            u = ctrl[i]
            if a["ctrllimited"]:
                u = ca.fmin(ca.fmax(u, a["ctrlrange"][0]), a["ctrlrange"][1])
            f = a["gain"] * u + a["bias_q"] * qpos[a["qpos"]] + a["bias_v"] * qvel[a["dof"]]
            if a["forcelimited"]:
                f = ca.fmin(ca.fmax(f, a["forcerange"][0]), a["forcerange"][1])
            qfrc_act[a["dof"]] = qfrc_act[a["dof"]] + f
        for j in range(1, self.m.njnt):
            if self.jnt_actfrclimited[j]:
                d = int(self.m.jnt_dofadr[j])
                qfrc_act[d] = ca.fmin(ca.fmax(qfrc_act[d], self.jnt_actfrcrange[j][0]), self.jnt_actfrcrange[j][1])
        qfrc_smooth = qfrc_passive + qfrc_act - bias
        outs, out_names = [M, qfrc_smooth], ["M", "qfrc_smooth"]
        if geoms:
            cpin.computeJointJacobians(cm, cd, q)
            cpin.updateFramePlacements(cm, cd)
            pos, rot = [], []
            for fid in self.geom_fids:
                oMf = cd.oMf[fid]
                pos.append(oMf.translation)
                rot.append(ca.reshape(oMf.rotation, 9, 1))
            # cd.J: the joint Jacobian in the WORLD frame, column k = dof k's motion axis as a spatial
            # velocity at the world origin [linear at origin; angular]. In MuJoCo's dof coordinates
            # (columns J T) and with the linear part moved to the base position p0.
            Jw = ca.mtimes(cd.J, T)
            p0 = qpos[0:3]
            lin = Jw[0:3, :] + ca.horzcat(*[ca.cross(Jw[3:6, k], p0) for k in range(nv)])
            cdof = ca.vertcat(lin, Jw[3:6, :])
            outs += [ca.horzcat(*pos), ca.horzcat(*rot), cdof]
            out_names += ["geom_pos", "geom_rot", "cdof"]
        outs = [ca.cse(o) for o in outs]
        self.M_sparsity = outs[0].sparsity()
        return ca.Function(name, [qpos, qvel, ctrl], outs, ["qpos", "qvel", "ctrl"], out_names)

    # --------------------------------------------------------------- integrate
    def qderiv_diag(self):
        """The diagonal of d qfrc_smooth / d qvel that MuJoCo's implicit integrators use: joint damping
        and the actuators' velocity bias (never clamped by actuatorfrcrange: as MuJoCo)."""
        d = -self.damping.copy()
        for a in self.actuators:
            d[a["dof"]] += a["bias_v"]
        return d

    def build_integrate(self, name="integrate"):
        """(after build_smooth, whose M sparsity this takes) integrate(qpos, qvel, M, qfrc_smooth, qfrc_constraint)
        -> [qpos', qvel', qacc]: MuJoCo's implicitfast (qH = M - dt D, D the velocity derivative above) or
        Euler advance, free-joint position in the world frame and quaternion by mju_quatIntegrate. `qacc`
        is the acceleration actually integrated (MjData.qacc for Euler; the implicit one for implicitfast)."""
        import casadi as ca
        nq, nv = self.nq, self.nv
        qpos, qvel = ca.SX.sym("qpos", nq), ca.SX.sym("qvel", nv)
        M = ca.SX.sym("M", self.M_sparsity)                     # the pattern smooth() writes
        qfs, qfc = ca.SX.sym("qfrc_smooth", nv), ca.SX.sym("qfrc_constraint", nv)
        dt = self.dt
        if self.integrator == "implicitfast":
            H = M - dt * ca.diag(ca.DM(self.qderiv_diag()))
        elif self.eulerdamp and np.any(self.damping):
            H = M + dt * ca.diag(ca.DM(self.damping))
        else:
            H = M
        qacc = _ldl_solve(H, qfs + qfc, nv, order=range(nv - 1, -1, -1))
        v1 = qvel + dt * qacc
        q1 = ca.vertcat(qpos[0:3] + dt * v1[0:3], _quat_integrate(qpos[3:7], v1[3:6], dt), qpos[7:] + dt * v1[6:])
        return ca.Function(name, [qpos, qvel, M, qfs, qfc], [ca.cse(q1), ca.cse(v1), ca.cse(qacc)],
                           ["qpos", "qvel", "M", "qfrc_smooth", "qfrc_constraint"], ["qpos_next", "qvel_next", "qacc"])


def _quat_to_mat_np(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
