"""Rigid-body dynamics in plain JAX from a model file, for cusadi2.

    from cusadi2.jax import rbd
    model = rbd.load("robot.xml")            # MJCF or URDF, parsed by MuJoCo
    tau = model.inverse_dynamics(q, qd, qdd) # == mj_inverse (constraints and actuation disabled)
    qdd = model.forward_dynamics(q, qd, tau) # == mj_forward (smooth dynamics)
    xpos, xmat = model.forward_kinematics(q) # body frames in the world, like MjData.xpos / xmat
    q1 = model.integrate_position(q, qd, dt) # == mj_integratePos (quaternions on free / ball joints)

MuJoCo conventions throughout: qpos / qvel layouts, free-joint linear velocity in
the world frame and angular velocity in the body frame, ball-joint angular
velocity in the body frame, joint `ref` (qpos0) and `springref`. Covered: free,
ball, slide and hinge joints, several joints per body, bodies without joints,
armature, damping, joint stiffness on slide / hinge joints, gravity
compensation. Not covered (load raises): tendons, fluid forces, stiffness on
ball / free joints. Constraints (contacts, limits, friction loss, equality) are
never part of these functions: in an optimal-control transcription contact
forces enter as decision variables through the inverse dynamics.

Formulation: Featherstone's recursive Newton-Euler and articulated-body
algorithms in body coordinates, motion vectors ordered [angular; linear].
Every inertia is a constant 6x6 and every transform a 6x6 with one rotation,
which is what keeps the CasADi expression graphs that cusadi2.jax builds from
these functions (and their Jacobians and Hessians) small: this is the
formulation of Pinocchio, not of MuJoCo, whose subtree-centre-of-mass
quantities fill the symbolic sparsity. Each body is a chain of one node per
joint (massless intermediate nodes for several joints on one body), the world
is the root node with the gravity trick (a_world = -g).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

FREE, BALL, SLIDE, HINGE, NONE = 0, 1, 2, 3, -1


# ----------------------------------------------------------------------------- small algebra
def quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return jnp.array([aw * bw - ax * bx - ay * by - az * bz,
                      aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx,
                      aw * bz + ax * by - ay * bx + az * bw])


def quat_integrate(q, w, dt):
    """q * exp(w dt / 2): MuJoCo's mju_quatIntegrate (w in the local frame). Written in
    th^2 = dt^2 |w|^2 with a series branch near zero so that it and its derivatives are
    finite at w = 0 (a norm has no derivative there; an NLP initial guess often has v = 0)."""
    th2 = dt * dt * jnp.dot(w, w)
    small = th2 < 1e-8
    th2s = jnp.where(small, 1.0, th2)                       # keep the sqrt branch away from 0
    th = jnp.sqrt(th2s)
    cos_half = jnp.where(small, 1.0 - th2 / 8.0 + th2 * th2 / 384.0, jnp.cos(th / 2))
    sinc_half = jnp.where(small, 0.5 - th2 / 48.0 + th2 * th2 / 3840.0, jnp.sin(th / 2) / th)   # sin(th/2) / th
    dq = jnp.concatenate([cos_half[None], sinc_half * dt * w])
    return quat_mul(q, dq)


def skew(v):
    return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def axis_angle_to_mat(axis, th):
    K = skew(jnp.asarray(axis))
    return jnp.eye(3) + jnp.sin(th) * K + (1 - jnp.cos(th)) * (K @ K)


def xform(E, p):
    """Spatial transform A -> B: B's origin at p (A coords), E the A -> B rotation."""
    Z = jnp.zeros((3, 3))
    return jnp.block([[E, Z], [-E @ skew(p), E]])


def crm(v):
    """Motion cross product v x."""
    Z = jnp.zeros((3, 3))
    return jnp.block([[skew(v[:3]), Z], [skew(v[3:]), skew(v[:3])]])


def crf(v):
    """Force cross product v x* = -(v x)^T."""
    return -crm(v).T


def _np_quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def spatial_inertia(mass, inertia_principal, ipos, iquat) -> np.ndarray:
    """Constant 6x6 body-frame inertia about the body origin, centre of mass at ipos."""
    Rin = _np_quat_to_mat(iquat)
    Ic = Rin @ np.diag(inertia_principal) @ Rin.T
    C = np.array([[0, -ipos[2], ipos[1]], [ipos[2], 0, -ipos[0]], [-ipos[1], ipos[0], 0]])
    return np.block([[Ic + mass * C @ C.T, mass * C], [mass * C.T, mass * np.eye(3)]])


def _solve_spd(D, rhs):
    """D^-1 rhs for a small SPD D: scalar division, else Cholesky (both SX-interpretable)."""
    if D.shape[0] == 1:
        return rhs / D[0, 0]
    L = jnp.linalg.cholesky(D)
    return jax.scipy.linalg.cho_solve((L, True), rhs)


# ----------------------------------------------------------------------------- the model
@dataclass(frozen=True)
class Node:
    """One joint (or a jointless body) in the kinematic tree."""
    parent: int                 # node index, -1 = world
    body: int                   # MuJoCo body id
    R0: np.ndarray              # (3, 3) constant rotation of the pre-joint frame in the parent node frame
    p0: np.ndarray              # (3,) constant origin of the pre-joint frame in the parent node frame
    jtype: int                  # FREE, BALL, SLIDE, HINGE or NONE
    axis: np.ndarray            # (3,) slide / hinge axis in the node frame
    anchor: np.ndarray          # (3,) joint anchor in the node frame
    qadr: int
    dadr: int
    ndof: int
    qpos0: float                # `ref` of a slide / hinge joint
    inertia: Optional[np.ndarray]   # (6, 6) spatial inertia; None for a massless intermediate node
    mass: float
    ipos: np.ndarray


class RigidBodyModel:
    """A tree of Nodes with numpy constants; the dynamics are methods taking JAX arrays."""

    def __init__(self, nodes: List[Node], nq: int, nv: int, nbody: int, gravity, armature, damping,
                 stiffness, qpos_spring, gravcomp, body_names):
        self.nodes, self.nq, self.nv, self.nbody = nodes, int(nq), int(nv), int(nbody)
        self.gravity = np.asarray(gravity, float)
        self.armature, self.damping = np.asarray(armature, float), np.asarray(damping, float)
        self.stiffness, self.qpos_spring = np.asarray(stiffness, float), np.asarray(qpos_spring, float)
        self.gravcomp = np.asarray(gravcomp, float)          # (nbody,)
        self.body_names = list(body_names)
        self._has_gravcomp = bool(np.any(self.gravcomp != 0))
        self._has_stiffness = bool(np.any(self.stiffness != 0))
        self._unsupported_dynamics: List[str] = []
        self._last_node = {}                                 # body id -> its last node
        for i, n in enumerate(nodes):
            self._last_node[n.body] = i

    def _check_dynamics(self):
        if self._unsupported_dynamics:
            raise NotImplementedError("rbd dynamics: " + ", ".join(self._unsupported_dynamics))

    def _ancestors(self, i: int) -> List[int]:
        """Node i and its ancestors, root first."""
        chain = []
        while i >= 0:
            chain.append(i)
            i = self.nodes[i].parent
        return chain[::-1]

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_mujoco(cls, m) -> "RigidBodyModel":
        import mujoco
        unsupported = []
        if m.ntendon and (np.any(m.tendon_stiffness != 0) or np.any(m.tendon_damping != 0)):
            unsupported.append("tendon stiffness / damping")
        if m.opt.viscosity != 0 or m.opt.density != 0:
            unsupported.append("fluid forces (opt.viscosity / opt.density)")
        nodes: List[Node] = []
        last_node_of_body = {0: -1}
        stiffness, qpos_spring = np.zeros(m.nv), np.zeros(m.nv)
        for b in range(1, m.nbody):
            parent_node = last_node_of_body[int(m.body_parentid[b])]
            R0, p0 = _np_quat_to_mat(m.body_quat[b]), np.array(m.body_pos[b], float)
            I = spatial_inertia(m.body_mass[b], m.body_inertia[b], m.body_ipos[b], m.body_iquat[b])
            joints = list(range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]))
            if not joints:
                nodes.append(Node(parent_node, b, R0, p0, NONE, np.zeros(3), np.zeros(3), 0, 0, 0, 0.0, I,
                                  float(m.body_mass[b]), np.array(m.body_ipos[b], float)))
                last_node_of_body[b] = len(nodes) - 1
                continue
            for k, j in enumerate(joints):
                jt, qa, da = int(m.jnt_type[j]), int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
                ndof = {FREE: 6, BALL: 3, SLIDE: 1, HINGE: 1}[jt]
                if jt == FREE and (len(joints) != 1 or m.body_parentid[b] != 0):
                    raise ValueError("rbd: a free joint must be the only joint of a child of the world")
                if m.jnt_stiffness[j] != 0:
                    if jt in (FREE, BALL):
                        raise NotImplementedError("rbd: stiffness on a free / ball joint")
                    stiffness[da], qpos_spring[da] = m.jnt_stiffness[j], m.qpos_spring[qa]
                first, last = k == 0, k == len(joints) - 1
                nodes.append(Node(
                    parent=parent_node if first else len(nodes) - 1, body=b,
                    R0=(np.eye(3) if jt == FREE or not first else R0), p0=(np.zeros(3) if jt == FREE or not first else p0),
                    jtype=jt, axis=np.array(m.jnt_axis[j], float), anchor=np.array(m.jnt_pos[j], float),
                    qadr=qa, dadr=da, ndof=ndof, qpos0=float(m.qpos0[qa]) if jt in (SLIDE, HINGE) else 0.0,
                    inertia=I if last else None, mass=float(m.body_mass[b]) if last else 0.0,
                    ipos=np.array(m.body_ipos[b], float)))
            last_node_of_body[b] = len(nodes) - 1
        gravity = np.zeros(3) if (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY) else np.array(m.opt.gravity)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        model = cls(nodes, m.nq, m.nv, m.nbody, gravity, m.dof_armature, m.dof_damping, stiffness, qpos_spring,
                    m.body_gravcomp, names)
        model._unsupported_dynamics = unsupported      # kinematics still work; the dynamics raise
        return model

    # ------------------------------------------------------------------ per-node kinematics
    def _joint(self, n: Node, q):
        """(X: parent node -> node, S (6 x ndof) or None, R_J, o_J) with R_J / o_J the joint
        rotation and origin in the pre-joint frame."""
        X0 = xform(jnp.asarray(n.R0.T), jnp.asarray(n.p0))
        if n.jtype == NONE:
            return X0, None, jnp.eye(3), jnp.zeros(3)
        if n.jtype == FREE:
            R, o = quat_to_mat(q[n.qadr + 3:n.qadr + 7]), q[n.qadr:n.qadr + 3]
            return xform(R.T, o), None, R, o
        c = jnp.asarray(n.anchor)
        if n.jtype == SLIDE:
            a = jnp.asarray(n.axis)
            o = a * (q[n.qadr] - n.qpos0)
            return xform(jnp.eye(3), o) @ X0, jnp.concatenate([jnp.zeros(3), a])[:, None], jnp.eye(3), o
        if n.jtype == HINGE:
            a = jnp.asarray(n.axis)
            R = axis_angle_to_mat(a, q[n.qadr] - n.qpos0)
            o = c - R @ c
            return xform(R.T, o) @ X0, jnp.concatenate([a, jnp.cross(c, a)])[:, None], R, o
        R = quat_to_mat(q[n.qadr:n.qadr + 4])                              # BALL
        o = c - R @ c
        return xform(R.T, o) @ X0, jnp.concatenate([jnp.eye(3), skew(c)], axis=0), R, o

    def _free_motion(self, n: Node, q, qd, qdd=None):
        """Free joint: body-coordinate spatial velocity (and acceleration without the
        world's gravity term) from MuJoCo's world-linear / body-angular qvel."""
        R = quat_to_mat(q[n.qadr + 3:n.qadr + 7])
        w, vb = qd[n.dadr + 3:n.dadr + 6], R.T @ qd[n.dadr:n.dadr + 3]
        v = jnp.concatenate([w, vb])
        if qdd is None:
            return R, v, None
        a = jnp.concatenate([qdd[n.dadr + 3:n.dadr + 6], R.T @ qdd[n.dadr:n.dadr + 3] - jnp.cross(w, vb)])
        return R, v, a

    def _gravcomp_wrench(self, n: Node, Rw):
        """Spatial force of gravity compensation on the node's body, body coordinates."""
        F = Rw.T @ (-self.gravcomp[n.body] * n.mass * jnp.asarray(self.gravity))
        return jnp.concatenate([jnp.cross(jnp.asarray(n.ipos), F), F])

    def _passive(self, q, qd):
        """Generalized passive forces (damping, springs); MuJoCo's qfrc_passive without gravcomp."""
        f = -jnp.asarray(self.damping) * qd
        if self._has_stiffness:
            k = jnp.asarray(self.stiffness)
            qs = jnp.asarray(self.qpos_spring)
            # springs only on slide / hinge dofs, whose qpos index equals their own address map
            qd_like = jnp.zeros(self.nv)
            for n in self.nodes:
                if n.jtype in (SLIDE, HINGE) and self.stiffness[n.dadr] != 0:
                    qd_like = qd_like.at[n.dadr].set(q[n.qadr])
            f = f - k * (qd_like - qs)
        return f

    # ------------------------------------------------------------------ dynamics
    def inverse_dynamics(self, q, qd, qdd):
        """qfrc_inverse = M(q) qdd + c(q, qd) - qfrc_passive (recursive Newton-Euler)."""
        self._check_dynamics()
        q, qd, qdd = jnp.asarray(q), jnp.asarray(qd), jnp.asarray(qdd)
        N = len(self.nodes)
        a_world = jnp.concatenate([jnp.zeros(3), -jnp.asarray(self.gravity)])
        v, a, f, X, S, Rfree = [None] * N, [None] * N, [None] * N, [None] * N, [None] * N, [None] * N
        Rw = [None] * N
        for i, n in enumerate(self.nodes):
            X[i], S[i], RJ, _ = self._joint(n, q)
            vp = v[n.parent] if n.parent >= 0 else jnp.zeros(6)
            ap = a[n.parent] if n.parent >= 0 else a_world
            if n.jtype == FREE:
                Rfree[i], vJ, aJ = self._free_motion(n, q, qd, qdd)
                v[i] = vJ
                a[i] = X[i] @ ap + aJ
            elif n.jtype == NONE:
                v[i], a[i] = X[i] @ vp, X[i] @ ap
            else:
                d = slice(n.dadr, n.dadr + n.ndof)
                vJ = S[i] @ qd[d]
                v[i] = X[i] @ vp + vJ
                a[i] = X[i] @ ap + S[i] @ qdd[d] + crm(v[i]) @ vJ
            if n.inertia is None:
                f[i] = jnp.zeros(6)
            else:
                I = jnp.asarray(n.inertia)
                f[i] = I @ a[i] + crf(v[i]) @ (I @ v[i])
            if self._has_gravcomp:
                Rp = Rw[n.parent] if n.parent >= 0 else jnp.eye(3)
                Rw[i] = Rp @ jnp.asarray(n.R0) @ RJ
                if n.inertia is not None and self.gravcomp[n.body] != 0:
                    f[i] = f[i] - self._gravcomp_wrench(n, Rw[i])
        tau = jnp.zeros(self.nv)
        for i in range(N - 1, -1, -1):
            n = self.nodes[i]
            if n.jtype == FREE:
                tau = tau.at[n.dadr:n.dadr + 3].set(Rfree[i] @ f[i][3:])
                tau = tau.at[n.dadr + 3:n.dadr + 6].set(f[i][:3])
            elif n.jtype != NONE:
                tau = tau.at[n.dadr:n.dadr + n.ndof].set(S[i].T @ f[i])
            if n.parent >= 0:
                f[n.parent] = f[n.parent] + X[i].T @ f[i]
        return tau + jnp.asarray(self.armature) * qdd - self._passive(q, qd)

    def forward_dynamics(self, q, qd, tau):
        """qdd = M(q)^-1 (tau + qfrc_passive - c(q, qd)) by the articulated-body algorithm:
        O(n), with a 6x6 Cholesky solve per free joint and a 3x3 one per ball joint."""
        self._check_dynamics()
        q, qd, tau = jnp.asarray(q), jnp.asarray(qd), jnp.asarray(tau)
        N = len(self.nodes)
        g = jnp.asarray(self.gravity)
        arm = jnp.asarray(self.armature)
        tau = tau + self._passive(q, qd)
        a_world = jnp.concatenate([jnp.zeros(3), -g])
        v, c, IA, pA, X, S, Rfree, Rw = ([None] * N for _ in range(8))
        for i, n in enumerate(self.nodes):
            X[i], S[i], RJ, _ = self._joint(n, q)
            vp = v[n.parent] if n.parent >= 0 else jnp.zeros(6)
            if n.jtype == FREE:
                Rfree[i], v[i], _ = self._free_motion(n, q, qd)
                c[i] = jnp.zeros(6)
            elif n.jtype == NONE:
                v[i], c[i] = X[i] @ vp, jnp.zeros(6)
            else:
                vJ = S[i] @ qd[n.dadr:n.dadr + n.ndof]
                v[i] = X[i] @ vp + vJ
                c[i] = crm(v[i]) @ vJ
            IA[i] = jnp.asarray(n.inertia) if n.inertia is not None else jnp.zeros((6, 6))
            pA[i] = crf(v[i]) @ (IA[i] @ v[i])
            if self._has_gravcomp:
                Rp = Rw[n.parent] if n.parent >= 0 else jnp.eye(3)
                Rw[i] = Rp @ jnp.asarray(n.R0) @ RJ
                if n.inertia is not None and self.gravcomp[n.body] != 0:
                    pA[i] = pA[i] - self._gravcomp_wrench(n, Rw[i])
        U, D, u = [None] * N, [None] * N, [None] * N
        perm = np.array([3, 4, 5, 0, 1, 2])                 # MuJoCo free dofs [lin; ang] -> spatial [ang; lin]
        for i in range(N - 1, -1, -1):
            n = self.nodes[i]
            if n.jtype == NONE:
                Ia, pa = IA[i], pA[i]
            else:
                d = slice(n.dadr, n.dadr + n.ndof)
                if n.jtype == FREE:
                    Si = jnp.eye(6)
                    tau_i = jnp.concatenate([tau[n.dadr + 3:n.dadr + 6], Rfree[i].T @ tau[n.dadr:n.dadr + 3]])
                    arm_i = arm[d][perm]
                else:
                    Si, tau_i, arm_i = S[i], tau[d], arm[d]
                U[i] = IA[i] @ Si
                D[i] = Si.T @ U[i] + jnp.diag(arm_i)
                u[i] = tau_i - Si.T @ pA[i]
                Dinv_Ut = _solve_spd(D[i], U[i].T)
                Ia = IA[i] - U[i] @ Dinv_Ut
                pa = pA[i] + Ia @ c[i] + U[i] @ _solve_spd(D[i], u[i])
            if n.parent >= 0:
                IA[n.parent] = IA[n.parent] + X[i].T @ Ia @ X[i]
                pA[n.parent] = pA[n.parent] + X[i].T @ pa
        a = [None] * N
        qdd = jnp.zeros(self.nv)
        for i, n in enumerate(self.nodes):
            ap = a[n.parent] if n.parent >= 0 else a_world
            a_ = X[i] @ ap + c[i]
            if n.jtype == NONE:
                a[i] = a_
                continue
            qdd_i = _solve_spd(D[i], u[i] - U[i].T @ a_)
            if n.jtype == FREE:
                a[i] = a_ + qdd_i
                R, w, vb = Rfree[i], v[i][:3], v[i][3:]
                qdd = qdd.at[n.dadr + 3:n.dadr + 6].set(a[i][:3])
                qdd = qdd.at[n.dadr:n.dadr + 3].set(R @ (a[i][3:] + jnp.cross(w, vb)) + g)
            else:
                a[i] = a_ + S[i] @ qdd_i
                qdd = qdd.at[n.dadr:n.dadr + n.ndof].set(qdd_i)
        return qdd

    # ------------------------------------------------------------------ kinematics
    def _node_poses(self, q):
        """World rotation and origin of every node frame."""
        N = len(self.nodes)
        R, p = [None] * N, [None] * N
        for i, n in enumerate(self.nodes):
            Rp = R[n.parent] if n.parent >= 0 else jnp.eye(3)
            pp = p[n.parent] if n.parent >= 0 else jnp.zeros(3)
            _, _, RJ, oJ = self._joint(n, q)
            Rpre = Rp @ jnp.asarray(n.R0)
            ppre = pp + Rp @ jnp.asarray(n.p0)
            R[i], p[i] = Rpre @ RJ, ppre + Rpre @ oJ
        return R, p

    def forward_kinematics(self, q) -> Tuple[jax.Array, jax.Array]:
        """World position (nbody, 3) and rotation (nbody, 3, 3) of every body frame; row 0 is the world."""
        R, p = self._node_poses(jnp.asarray(q))
        xpos = [jnp.zeros(3)] + [p[self._last_node[b]] for b in range(1, self.nbody)]
        xmat = [jnp.eye(3)] + [R[self._last_node[b]] for b in range(1, self.nbody)]
        return jnp.stack(xpos), jnp.stack(xmat)

    def body_jacobian(self, q) -> jax.Array:
        """(nbody, 6, nv) Jacobian of every body frame, rows [linear; angular] of the frame
        origin in world axes with respect to qvel: MuJoCo's mj_jacBody (jacp, jacr) and
        Pinocchio's LOCAL_WORLD_ALIGNED frame Jacobian. Row 0 (the world) is zero. Only
        the columns of a body's ancestor joints are written, so the structural pattern is
        the tree's."""
        q = jnp.asarray(q)
        R, p = self._node_poses(q)
        out = [jnp.zeros((6, self.nv))]
        for b in range(1, self.nbody):
            i = self._last_node[b]
            J = jnp.zeros((6, self.nv))
            for k in self._ancestors(i):
                n = self.nodes[k]
                if n.jtype == NONE:
                    continue
                d = slice(n.dadr, n.dadr + n.ndof)
                if n.jtype == FREE:
                    w = R[k]                                             # angular dofs in the body frame
                    v = jnp.cross(w.T, p[i] - p[k]).T                    # columns w_j x (p_i - p_k)
                    J = J.at[:3, n.dadr:n.dadr + 3].set(jnp.eye(3))      # linear dofs in the world frame
                    J = J.at[:3, n.dadr + 3:n.dadr + 6].set(v)
                    J = J.at[3:, n.dadr + 3:n.dadr + 6].set(w)
                    continue
                anchor = p[k] + R[k] @ jnp.asarray(n.anchor)
                if n.jtype == SLIDE:
                    J = J.at[:3, d].set((R[k] @ jnp.asarray(n.axis))[:, None])
                elif n.jtype == HINGE:
                    w = R[k] @ jnp.asarray(n.axis)
                    J = J.at[:3, d].set(jnp.cross(w, p[i] - anchor)[:, None])
                    J = J.at[3:, d].set(w[:, None])
                else:                                                    # BALL: local angular velocity
                    w = R[k]
                    J = J.at[:3, d].set(jnp.cross(w.T, p[i] - anchor).T)
                    J = J.at[3:, d].set(w)
            out.append(J)
        return jnp.stack(out)

    def integrate_position(self, q, qd, dt):
        """MuJoCo's mj_integratePos: q + qd dt on slide / hinge, quaternion exponential on free / ball."""
        q, qd = jnp.asarray(q), jnp.asarray(qd)
        out = q
        for n in self.nodes:
            if n.jtype == FREE:
                out = out.at[n.qadr:n.qadr + 3].set(q[n.qadr:n.qadr + 3] + dt * qd[n.dadr:n.dadr + 3])
                out = out.at[n.qadr + 3:n.qadr + 7].set(quat_integrate(q[n.qadr + 3:n.qadr + 7], qd[n.dadr + 3:n.dadr + 6], dt))
            elif n.jtype == BALL:
                out = out.at[n.qadr:n.qadr + 4].set(quat_integrate(q[n.qadr:n.qadr + 4], qd[n.dadr:n.dadr + 3], dt))
            elif n.jtype != NONE:
                out = out.at[n.qadr].set(q[n.qadr] + dt * qd[n.dadr])
        return out


def load(path: str) -> RigidBodyModel:
    """Build the model from an MJCF or URDF file through MuJoCo's parser."""
    import mujoco
    return RigidBodyModel.from_mujoco(mujoco.MjModel.from_xml_path(str(path)))
