"""frax side of the Menagerie campaign: the same robot as a frax.Robot, its
compatibility gate, and the three benchmark functions.

frax (github.com/danielpmorton/frax, arXiv 2604.04310) reads URDF only and knows
revolute and prismatic joints only, so the model comes from the Pinocchio model
the campaign already gated against MuJoCo (model.load_pinocchio), written out as
a URDF with one revolute or prismatic joint per degree of freedom:

    hinge / slide       one joint, Pinocchio's placement and axis
    free joint          x, y, z prismatic then x, y, z revolute, the way frax
                        itself models a floating base (frax.load_g1)
    ball joint          x, y, z revolute
    composite joint     its sub-joints in order (several MJCF joints in one body)

Every frax model therefore has nq = nv = MuJoCo's nv: a free joint's orientation
is three angles (R = Rx Ry Rz), not a quaternion. Massless links join the joints
of one expanded joint. The inertias are Pinocchio's; nothing here uses them.

Frame set: the campaign's (named MuJoCo bodies excluding 'world', body order).
A body is a fixed offset on the last joint that moves it, evaluated with frax's
own Robot._frame_transform / Robot._frame_jacobian; a body welded to the world
is a constant placement with a zero Jacobian. The functions of one configuration:

    fk    world position (3) and rotation (9) of every body
    jac   6 x nv Jacobian of every body, linear rows then angular rows, world
          aligned at the body origin, with respect to the joint rates
    hess  jax.jacfwd of jac: (6 nv n_bodies) x nv

run_gate checks fk against MuJoCo's kinematics at random configurations (through
the frax -> Pinocchio -> MuJoCo coordinate maps) and jac against forward-mode AD
of fk, both to model.FK_TOL.
"""
import os

import numpy as np

import model as _model

N_CHECK_CONFIGS = _model.N_CHECK_CONFIGS
_AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _f(x):
    """Shortest round-trip decimal of a scalar (URDF attribute text)."""
    return repr(float(x))


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    return {"X": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "Y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "Z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def _xyz_to_quat(a):
    """Angles (a, b, c) of R = Rx(a) Ry(b) Rz(c) -> Pinocchio quaternion (x, y, z, w)."""
    import pinocchio as pin
    return pin.Quaternion(_rot("X", a[0]) @ _rot("Y", a[1]) @ _rot("Z", a[2])).coeffs()


def _quat_to_xyz(xyzw):
    """Inverse of _xyz_to_quat on the branch |b| < pi/2."""
    import pinocchio as pin
    R = pin.Quaternion(np.asarray(xyzw, float).reshape(4, 1)).normalized().toRotationMatrix()
    return np.array([np.arctan2(-R[1, 2], R[2, 2]), np.arcsin(np.clip(R[0, 2], -1.0, 1.0)),
                     np.arctan2(-R[0, 1], R[0, 0])])


class FraxModel:
    """The Pinocchio model as one-dof joints: `joints` = [(kind 'R'|'P', axis,
    origin SE3, parent joint index or -1, carries pin joint id or None)],
    `segments` = the Pinocchio q layout as [('scalar' | 'free' | 'ball', first
    frax index)], `last` = {pin joint id: its last frax joint index}."""

    def __init__(self, pmodel, root_corr=None):
        import pinocchio as pin
        self.joints, self.segments, self.last = [], [], {0: -1}
        self.expanded = {"free": 0, "ball": 0, "composite": 0}
        corr = pin.SE3(np.asarray(root_corr[0]), np.asarray(root_corr[1])) if root_corr is not None else None
        for j in range(1, pmodel.njoints):
            parent = int(pmodel.parents[j])
            origin = pmodel.jointPlacements[j]
            if parent == 0 and corr is not None:
                origin = corr * origin
            self._expand(pmodel.joints[j], origin, self.last[parent])
            self.joints[-1] = self.joints[-1][:4] + (j,)     # the joint that carries pin joint j's body
            self.last[j] = len(self.joints) - 1
        self.n = len(self.joints)
        assert self.n == pmodel.nv
        self.pmodel, self.corr = pmodel, corr

    def _add(self, kind, axis, origin, parent):
        self.joints.append((kind, tuple(float(a) for a in axis), origin, parent, None))
        return len(self.joints) - 1

    def _chain(self, kinds, origin, parent):
        import pinocchio as pin
        for k in kinds:
            parent = self._add(k[0], _AXES[k[1]], origin, parent)
            origin = pin.SE3.Identity()
        return parent

    def _expand(self, jm, origin, parent):
        """Append jm's one-dof joints; returns the index of the last one."""
        name = jm.shortname().replace("JointModel", "")
        start = len(self.joints)
        if name in ("RX", "RY", "RZ", "PX", "PY", "PZ"):
            self.segments.append(("scalar", start))
            return self._add(name[0], _AXES[name[1]], origin, parent)
        if name in ("RevoluteUnaligned", "PrismaticUnaligned"):
            self.segments.append(("scalar", start))
            return self._add(name[0], np.asarray(jm.extract().axis).ravel(), origin, parent)
        if name == "FreeFlyer":
            self.segments.append(("free", start))
            self.expanded["free"] += 1
            return self._chain(("PX", "PY", "PZ", "RX", "RY", "RZ"), origin, parent)
        if name == "Spherical":
            self.segments.append(("ball", start))
            self.expanded["ball"] += 1
            return self._chain(("RX", "RY", "RZ"), origin, parent)
        if name == "Composite":                 # M = M_0 P_1 M_1 ... (P_0 is the identity here)
            self.expanded["composite"] += 1
            comp = jm.extract()
            for k, sub in enumerate(comp.joints):
                parent = self._expand(sub, origin * comp.jointPlacements[k], parent)
                origin = origin.Identity()
            return parent
        raise ValueError(f"joint type {name} has no revolute/prismatic expansion")

    # --- coordinates -------------------------------------------------------
    def to_pin(self, q):
        """frax q (nv) -> Pinocchio q (nq)."""
        out = []
        for kind, i in self.segments:
            if kind == "scalar":
                out.append([q[i]])
            elif kind == "free":
                out += [q[i:i + 3], _xyz_to_quat(q[i + 3:i + 6])]
            else:
                out.append(_xyz_to_quat(q[i:i + 3]))
        return np.concatenate(out)

    def from_pin(self, qp):
        """Pinocchio q (nq) -> frax q (nv), orientations on the branch |pitch| < pi/2."""
        q, pos = np.empty(self.n), 0
        for kind, i in self.segments:
            if kind == "scalar":
                q[i] = qp[pos]; pos += 1
            elif kind == "free":
                q[i:i + 3] = qp[pos:pos + 3]
                q[i + 3:i + 6] = _quat_to_xyz(qp[pos + 3:pos + 7]); pos += 7
            else:
                q[i:i + 3] = _quat_to_xyz(qp[pos:pos + 4]); pos += 4
        assert pos == self.pmodel.nq
        return q

    # --- URDF --------------------------------------------------------------
    def write_urdf(self, path):
        import pinocchio as pin

        def inertial(I=None):
            m, c, J = (I.mass, I.lever, I.inertia) if I is not None else (0.0, np.zeros(3), np.zeros((3, 3)))
            return (f'<inertial><origin xyz="{_f(c[0])} {_f(c[1])} {_f(c[2])}" rpy="0 0 0"/><mass value="{_f(m)}"/>'
                    f'<inertia ixx="{_f(J[0, 0])}" ixy="{_f(J[0, 1])}" ixz="{_f(J[0, 2])}" iyy="{_f(J[1, 1])}" '
                    f'iyz="{_f(J[1, 2])}" izz="{_f(J[2, 2])}"/></inertial>')

        lines = ['<?xml version="1.0"?>', '<robot name="menagerie">', f'<link name="world">{inertial()}</link>']
        for i, (kind, axis, origin, parent, pin_j) in enumerate(self.joints):
            t = origin.translation
            rpy = pin.rpy.matrixToRpy(origin.rotation)
            I = self.pmodel.inertias[pin_j] if pin_j is not None else None
            lines.append(f'<link name="l{i}">{inertial(I)}</link>')
            lines.append(
                f'<joint name="j{i}" type="{"revolute" if kind == "R" else "prismatic"}">'
                f'<parent link="{"world" if parent < 0 else f"l{parent}"}"/><child link="l{i}"/>'
                f'<origin xyz="{_f(t[0])} {_f(t[1])} {_f(t[2])}" rpy="{_f(rpy[0])} {_f(rpy[1])} {_f(rpy[2])}"/>'
                f'<axis xyz="{_f(axis[0])} {_f(axis[1])} {_f(axis[2])}"/>'
                f'<limit lower="-1e6" upper="1e6" effort="1e6" velocity="1e6"/></joint>')
        lines.append("</robot>")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return [f"j{i}" for i in range(self.n)]

    def frames(self, names):
        """Per body: (frax joint index or -1 for a body welded to the world, 4x4
        offset on that joint, or the constant world placement)."""
        import pinocchio as pin
        out = []
        for name in names:
            f = self.pmodel.frames[self.pmodel.getFrameId(name, pin.FrameType.BODY)]
            j, P = int(f.parentJoint), f.placement
            if j == 0 and self.corr is not None:
                P = self.corr * P
            out.append((self.last[j], np.asarray(P.homogeneous)))
        return out


def load(xml_path, urdf_path):
    """-> (MuJoCo model, FraxModel, frax.Robot, body frames, frax q at qpos0, qpos index map)."""
    import frax
    m = _model.load_mujoco(xml_path)
    pmodel = _model.load_pinocchio(xml_path, m=m)
    idx = _model.qpos_mj_to_pin_index(m, pmodel)
    fm = FraxModel(pmodel, _model.root_correction(m, pmodel, idx))
    robot = frax.Robot(urdf_path, joint_ordering=fm.write_urdf(urdf_path))
    assert robot.num_joints == fm.n
    assert list(robot.parent_idxs) == [e[3] for e in fm.joints]
    q0 = fm.from_pin(np.asarray(m.qpos0)[idx])
    return m, fm, robot, fm.frames(_model.body_names(m)), q0, idx


def build_functions(robot, frames):
    """(fk, jac, hess) of one configuration q (nv,), as jax functions."""
    import jax
    import jax.numpy as jnp
    n = robot.num_joints
    moving = [k for k, (i, _) in enumerate(frames) if i >= 0]
    static = [k for k, (i, _) in enumerate(frames) if i < 0]
    order = np.argsort(np.array(moving + static))            # (moving, static) stack -> body order
    parents = np.array([frames[k][0] for k in moving])
    offsets = np.stack([frames[k][1] for k in moving])
    chains = [np.nonzero(np.asarray(robot.ancestor_mask)[frames[k][0]])[0] for k in moving]
    world = np.stack([frames[k][1] for k in static]) if static else np.zeros((0, 4, 4))

    def fk(q):
        T = robot.joint_to_world_transforms(q)
        Tb = robot._frame_transform(T, jnp.asarray(offsets, T.dtype), parents)
        Tb = jnp.concatenate([Tb, jnp.asarray(world, T.dtype)])[order]
        return jnp.concatenate([Tb[:, :3, 3].reshape(-1), Tb[:, :3, :3].reshape(-1)])

    def jac(q):
        T = robot.joint_to_world_transforms(q)
        J = [robot._frame_jacobian(T, jnp.asarray(offsets[b], T.dtype), chains[b]) for b in range(len(moving))]
        J = jnp.concatenate([jnp.stack(J), jnp.zeros((len(static), 6, n), T.dtype)])[order]
        return J.reshape(-1)

    def hess(q):
        return jax.jacfwd(jac)(q).reshape(-1)

    return fk, jac, hess


def run_gate(xml_path, urdf_path):
    """frax compatibility gate, JSON-serializable; every failure is a recorded
    skip reason. fk_max_err is against MuJoCo; jac_max_err is frax's frame
    Jacobian against forward-mode AD of its own fk (linear rows: d position;
    angular rows: the axis of dR R^T)."""
    import mujoco
    out = {"status": "skip", "reason": None, "n_joints": None, "n_bodies": None, "n_bodies_static": None,
           "expanded": None, "pure_chain": None, "fk_max_err": None, "jac_max_err": None}
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import importlib.metadata
        out["frax"], out["jax"] = importlib.metadata.version("frax"), jax.__version__
        m, fm, robot, frames, q0, idx = load(xml_path, urdf_path)
    except Exception as e:
        out["reason"] = f"frax_load_failed: {e!r}"[:500]
        return out
    nb, n = len(frames), fm.n
    out.update(n_joints=n, n_bodies=nb, n_bodies_static=sum(i < 0 for i, _ in frames),
               expanded=fm.expanded, pure_chain=bool(robot.is_pure_kinematic_chain))
    try:
        fk, jac, _ = build_functions(robot, frames)
        fk_j, jac_j, dfk_j = jax.jit(fk), jax.jit(jac), jax.jit(jax.jacfwd(fk))
        body_ids = _model.named_body_ids(m)
        d = mujoco.MjData(m)
        rng = np.random.default_rng(0)
        fk_err = jac_err = 0.0
        for _ in range(N_CHECK_CONFIGS):
            q = q0 + rng.normal(0.0, 0.3, size=n)
            y = np.asarray(fk_j(q))
            pos, rot = y[:3 * nb].reshape(nb, 3), y[3 * nb:].reshape(nb, 3, 3)
            q_mj = np.empty(m.nq)
            q_mj[idx] = fm.to_pin(q)
            d.qpos[:] = q_mj
            mujoco.mj_kinematics(m, d)
            fk_err = max(fk_err, float(np.abs(pos - d.xpos[body_ids]).max()),
                         float(np.abs(rot - d.xmat[body_ids].reshape(nb, 3, 3)).max()))
            J = np.asarray(jac_j(q)).reshape(nb, 6, n)
            dy = np.asarray(dfk_j(q))
            dpos, drot = dy[:3 * nb].reshape(nb, 3, n), dy[3 * nb:].reshape(nb, 3, 3, n)
            W = np.einsum("bijk,blj->bilk", drot, rot)      # dR R^T per dof: skew of the angular column
            dw = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], axis=1)
            jac_err = max(jac_err, float(np.abs(J[:, :3] - dpos).max()), float(np.abs(J[:, 3:] - dw).max()))
        out["fk_max_err"], out["jac_max_err"] = fk_err, jac_err
        if fk_err > _model.FK_TOL:
            out["reason"] = f"fk_mismatch: max_err={fk_err:.3e} (tol {_model.FK_TOL:g})"
            return out
        if jac_err > _model.FK_TOL:
            out["reason"] = f"jac_mismatch: max_err={jac_err:.3e} (tol {_model.FK_TOL:g})"
            return out
    except Exception as e:
        out["reason"] = f"frax_check_failed: {e!r}"[:500]
        return out
    out["status"] = "pass"
    return out
