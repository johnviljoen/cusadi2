"""The MIT humanoid MPC stage (FORMULATION.md sections 2-6) written in JAX on top of MuJoCo MJX, as a baseline
for the cusadi2 assembly: MJX supplies the rigid-body algorithms (kinematics, RNE), JAX autodiff the
derivatives, and the result is laid out exactly like `mpc_assemble`'s output so the same CUDA QP solver
(mpc_qp_lib.cu) consumes it.

    asm = make_assemble(build_dir)                    # jitted
    blk, q, z_lb, z_ub = asm(x_pad, p_stage, p_shared)
      x_pad (B, N+2, 60), p_stage (B, N, 69), p_shared (B, 106)  ->  blk (B, N, 1068), q (B, N, 60), z_lb, z_ub (B, N, 74)

Conventions. The formulation is Pinocchio's: q = [p, quat xyzw, joints], v = [linear, angular] of the base
in the base frame, increments on SE(3) x R^18. MuJoCo's free joint has qpos = [p, quat wxyz, joints] and
qvel = [linear in the WORLD frame, angular in the base frame], so with R the base rotation
    qvel = [R v_lin, w, v_j],   qacc = [R (a_lin + w x v_lin), a_ang, a_j],   tau_base = [R^T f_lin, f_ang].
"""
import json, os
import jax, jax.numpy as jp, numpy as np, mujoco
from mujoco import mjx
jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
NQ, NV, NF, NS, NG = 25, 24, 12, 60, 74
DQ, V, F = slice(0, NV), slice(NV, 2 * NV), slice(2 * NV, 2 * NV + NF)
TOE, HEEL = (0.105, 0.0, -0.04), (-0.045, 0.0, -0.04)
SITES = ["right_toe", "left_toe", "right_heel", "left_heel"]          # contact order of the formulation

# ---------------------------------------------------------------- SE(3) x R^n, Pinocchio conventions
def _series(t, closed, c0, c1, c2):
    """closed(t) for t = theta^2 away from 0, its Taylor polynomial near 0 (both branches finite)."""
    small = t < 1e-4
    return jp.where(small, c0 + t * (c1 + t * c2), closed(jp.where(small, 1.0, t)))
def quat_mul(a, b):                                   # xyzw
    av, aw, bv, bw = a[:3], a[3], b[:3], b[3]
    return jp.concatenate([aw * bv + bw * av + jp.cross(av, bv), jp.array([aw * bw - av @ bv])])
def quat_rot(q):
    x, y, z, w = q
    return jp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def skew(w): return jp.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
def exp6(nu, w):
    """SE(3) exponential: (translation, quaternion) of exp([nu; w])."""
    t = w @ w
    half = _series(t, lambda s: jp.sin(jp.sqrt(s) / 2) / jp.sqrt(s), 0.5, -1 / 48, 1 / 3840)
    B = _series(t, lambda s: (1 - jp.cos(jp.sqrt(s))) / s, 0.5, -1 / 24, 1 / 720)
    C = _series(t, lambda s: (jp.sqrt(s) - jp.sin(jp.sqrt(s))) / (s * jp.sqrt(s)), 1 / 6, -1 / 120, 1 / 5040)
    W = skew(w)
    return nu + B * (W @ nu) + C * (W @ (W @ nu)), jp.concatenate([half * w, jp.array([jp.cos(jp.sqrt(t + 1e-300) / 2)])])
def log6(p, q):
    """SE(3) logarithm of (translation p, quaternion q): [nu; w]."""
    q = jp.where(q[3] < 0, -q, q)
    n2 = q[:3] @ q[:3]; small = n2 < 1e-8
    n = jp.sqrt(jp.where(small, 1.0, n2))
    scale = jp.where(small, (2 / q[3]) * (1 - n2 / (3 * q[3] ** 2) + n2 ** 2 / (5 * q[3] ** 4)), 2 * jp.arctan2(n, q[3]) / n)
    w = scale * q[:3]; t = w @ w
    D = _series(t, lambda s: (1 - jp.sqrt(s) * jp.sin(jp.sqrt(s)) / (2 * (1 - jp.cos(jp.sqrt(s))))) / s, 1 / 12, 1 / 720, 1 / 30240)
    W = skew(w)
    return jp.concatenate([p - 0.5 * (W @ p) + D * (W @ (W @ p)), w])
def integrate(q, dq):                                 # pin.integrate
    t, r = exp6(dq[:3], dq[3:6])
    return jp.concatenate([q[:3] + quat_rot(q[3:7]) @ t, quat_mul(q[3:7], r), q[7:] + dq[6:]])
def difference(q1, q2):                               # pin.difference(q1, q2): the tangent taking q1 to q2
    conj = jp.concatenate([-q1[3:7][:3], q1[3:7][3:]])
    return jp.concatenate([log6(quat_rot(q1[3:7]).T @ (q2[:3] - q1[:3]), quat_mul(conj, q2[3:7])), q2[7:] - q1[7:]])

# ---------------------------------------------------------------- MuJoCo model
def load_model(urdf=None, armature=None):
    urdf = urdf or os.path.join(os.environ.get("CUSADI_WIP", os.path.join(HERE, "vendor")), "mit_humanoid_urdf/humanoid_full_sf_toeheel_mj.urdf")
    spec = mujoco.MjSpec.from_file(urdf); spec.meshdir = os.path.dirname(urdf)
    root = spec.worldbody.bodies[0]; root.add_freejoint()
    for side in ("right", "left"):
        foot = next(b for b in spec.bodies if b.name == f"{side}_foot")
        foot.add_site(name=f"{side}_toe", pos=TOE); foot.add_site(name=f"{side}_heel", pos=HEEL)
    for g in spec.geoms: g.contype = g.conaffinity = 0            # no collisions: MJX rejects some geom pairs at put_model
    m = spec.compile()
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    if armature is not None: m.dof_armature[:] = armature
    assert m.nq == NQ and m.nv == NV, (m.nq, m.nv)
    return m

def make_stage(build_dir):
    """Returns stage(x_prev, x_cur, x_next, dt_cur, dt_next, is_first, has_next, p_stage, p_shared) -> (cost, g)."""
    meta = json.load(open(os.path.join(build_dir, "meta.json")))
    mj = load_model(armature=meta["armature"]); mx = mjx.put_model(mj); d0 = mjx.make_data(mj)
    site = np.array([mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, n) for n in SITES])
    hip = np.array([mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("right_hip_yaw", "left_hip_yaw")])
    site_body, root_of = mj.site_bodyid[site], mj.body_rootid
    cx, cy, cz = (np.array(meta["contact_idx"][a]) for a in "xyz")
    mu, weight, l_leg = meta["mu"], meta["bodyweight"], meta["l_leg_max"]
    lateral = meta["collision_radius"] ** 2 + meta["offset_lateral_stance"]
    sizes = dict(meta["p_stage"]); off = np.cumsum([0] + [n for _, n in meta["p_stage"]]); PS = {n: slice(off[i], off[i + 1]) for i, (n, _) in enumerate(meta["p_stage"])}
    off = np.cumsum([0] + [n for _, n in meta["p_shared"]]); PG = {n: slice(off[i], off[i + 1]) for i, (n, _) in enumerate(meta["p_shared"])}
    bez = np.array([0.0, 0.0, 0.02 / 4, 8 * (0.05 - 0.02 / 16 - 0.02 / 16) / 3, 0.02 / 4])       # v_TO 0.02, v_TD -0.02, h 0.05
    binom = np.array([1.0, 4.0, 6.0, 4.0, 1.0])

    def rigid_body(q, v, a):
        """Contact positions and velocities (world), hip positions, base rows of the inverse dynamics (Pinocchio frame)."""
        R = quat_rot(q[3:7]); w = v[3:6]
        d = d0.replace(qpos=jp.concatenate([q[:3], q[6:7], q[3:6], q[7:]]),
                       qvel=jp.concatenate([R @ v[:3], v[3:]]),
                       qacc=jp.concatenate([R @ (a[:3] + jp.cross(w, v[:3])), a[3:]]))
        d = mjx.rne(mx, mjx.com_vel(mx, mjx.com_pos(mx, mjx.kinematics(mx, d))), flg_acc=True)
        p = d.site_xpos[site]
        cvel = d.cvel[site_body]                                         # [angular, linear at the subtree com], world axes
        pdot = cvel[:, 3:] + jp.cross(cvel[:, :3], p - d.subtree_com[root_of[site_body]])
        tau = jp.concatenate([R.T @ d.qfrc_bias[:3], d.qfrc_bias[3:6]])
        return p, pdot, d.xpos[hip], tau, R

    def stage(x_prev, x_cur, x_next, dt_cur, dt_next, is_first, has_next, ps, pg):
        q_0, v_0 = pg[PG["q_0"]], pg[PG["v_0"]]
        Q_q = jp.concatenate([pg[PG["Q_fb"]], pg[PG["Q_leg"]], pg[PG["Q_leg"]], pg[PG["Q_arm"]], pg[PG["Q_arm"]]])
        Q_v = jp.concatenate([pg[PG["Qd_fb"]], pg[PG["Qd_leg"]], pg[PG["Qd_leg"]], pg[PG["Qd_arm"]], pg[PG["Qd_arm"]]])
        Q_a = jp.concatenate([pg[PG["Qa_fb"]], pg[PG["Qa_leg"]], pg[PG["Qa_leg"]], pg[PG["Qa_arm"]], pg[PG["Qa_arm"]]])
        q_des, v_des, F_des, phi, contact = (ps[PS[n]] for n in ("q_des", "v_des", "F_des", "phi_traj", "contact_traj"))
        v_k, F_k = x_cur[V], x_cur[F]
        q_k = integrate(q_0, x_cur[DQ])
        v_prev = jp.where(is_first > 0.5, v_0, x_prev[V])
        q_prev = jp.where(is_first > 0.5, q_0, integrate(q_0, x_prev[DQ]))
        a_k = (v_k - v_prev) / dt_cur
        # cost
        sign = jp.where(q_k[3:7] @ q_des[3:7] >= 0, 1.0, -1.0)
        e_q = difference(q_k, jp.concatenate([q_des[:3], sign * q_des[3:7], q_des[7:]]))
        e_v, e_F = v_des - v_k, F_des - F_k
        a_n = (x_next[V] - v_k) / dt_next
        cost = dt_cur * (e_q @ (Q_q * e_q) + e_v @ (Q_v * e_v) + a_k @ (Q_a * a_k) + e_F @ (pg[PG["R_F"]] * e_F)) \
            + has_next * dt_next * (a_n @ (Q_a * a_n))
        # constraints
        p, pdot, p_hip, tau, R = rigid_body(q_k, v_k, a_k)
        g_pos = difference(q_k, integrate(q_prev, (v_prev + a_k * dt_cur) * dt_cur))
        f = (F_k * weight).reshape(4, 3)
        r = (p - q_k[:3]) @ R                                             # contact points in the base frame
        fb = f @ R
        g_dyn = tau - jp.concatenate([fb.sum(0), jp.cross(r, fb).sum(0)])
        s = phi[:, None] ** np.arange(5) * (1 - phi[:, None]) ** np.arange(4, -1, -1)
        g_swing = p[:, 2] - s @ (binom * bez)
        Fx, Fy, Fz = F_k[cx], F_k[cy], F_k[cz]; lim = mu * Fz / np.sqrt(2)
        g_fric = jp.concatenate([-lim - Fx, Fx - lim, -lim - Fy, Fy - lim])
        g_cv = jp.concatenate([pdot[:, 0] * contact, pdot[:, 1] * contact])
        g_cF = (F_k.reshape(4, 3) * (1 - contact)[:, None]).reshape(-1)
        qx, qy, qz, qw = q_k[3], q_k[4], q_k[5], q_k[6]
        yaw = jp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)); c, sn = jp.cos(yaw), jp.sin(yaw)
        y_body = -sn * p[:, 0] + c * p[:, 1]                               # y in the yaw-aligned frame
        g_lat = jp.array([y_body[1] - y_body[0] - lateral, y_body[3] - y_body[2] - lateral])
        d_r, d_l = 0.5 * (p[0] + p[2]) - p_hip[0], 0.5 * (p[1] + p[3]) - p_hip[1]
        g_len = jp.array([d_r @ d_r - l_leg ** 2, d_l @ d_l - l_leg ** 2])
        return cost, jp.concatenate([g_pos, g_dyn, g_swing, g_fric, g_cv, g_cF, g_lat, g_len])
    return stage, meta

def make_assemble(build_dir, chunk=256):
    """Jitted assemble(x_pad, p_stage, p_shared) with the output layout of the cusadi2 plan `mpc_assemble`.
    Batches above `chunk` instances are evaluated `chunk` at a time (lax.map): forward-mode AD through MJX
    keeps 120 tangents of every intermediate, which exhausts a 16 GB device near B = 1024 otherwise."""
    stage, meta = make_stage(build_dir)
    N = meta["N"]; dt = np.array(meta["dt"]); dt_pad = jp.asarray(np.concatenate([dt, [1.0]]))
    is_first = jp.asarray(np.arange(N) == 0, jp.float64); has_next = jp.asarray(np.arange(N) < N - 1, jp.float64)
    eq = np.zeros(NG, bool); eq[meta["eq_rows"]] = True
    lbg, ubg = jp.asarray(np.where(eq, 0.0, np.array(meta["lbg"], float))), jp.asarray(np.where(eq, 0.0, np.array(meta["ubg"], float)))
    pat = {k: (np.array(v[0]), np.array(v[1])) for k, v in meta["pattern"].items()}

    def one(xp, xc, xn, dtc, dtn, first, nxt, ps, pg):
        cost = lambda a, b: stage(a, b, xn, dtc, dtn, first, nxt, ps, pg)[0]
        cons = lambda z: stage(z[:NS], z[NS:], xn, dtc, dtn, first, nxt, ps, pg)[1]
        grad_cur = jax.grad(cost, argnums=1)
        P, c = jax.jacfwd(lambda b: grad_cur(xp, b))(xc), grad_cur(xp, xc)
        P_up = jax.jacfwd(lambda b: jax.grad(cost, argnums=0)(xp, b))(xc)                 # rows x_{k-1}, cols x_k
        z = jp.concatenate([xp, xc]); g = cons(z); J = jax.jacfwd(cons)(z)
        A_prev, A = J[:, :NS], J[:, NS:]
        blk = jp.concatenate([P[pat["Ptriu"]], P_up[pat["Pup"]], A[pat["A"]], A_prev[pat["Aprev"]]])
        return blk, c, lbg - g, ubg - g
    over_stages = jax.vmap(one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None))
    def instance(x_pad, ps, pg):
        return over_stages(x_pad[:N], x_pad[1:N + 1], x_pad[2:], dt_pad[:N], dt_pad[1:], is_first, has_next, ps, pg)
    @jax.jit
    def assemble(x_pad, ps, pg):
        if x_pad.shape[0] <= chunk: return jax.vmap(instance)(x_pad, ps, pg)
        return jax.lax.map(lambda a: instance(*a), (x_pad, ps, pg), batch_size=chunk)
    return assemble, meta
