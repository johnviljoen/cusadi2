"""The MPPI rollout function: one control-rate-independent physics step of the Unitree Go2 (Menagerie
go2.xml through Pinocchio) on the course, with PD joint control, penalty contact and the running
cost, as a single CasADi SX function. See FORMULATION.md.

    step(x, knots, rp, par) -> x_next

    x      [q(19) v(18) t tau cost]   Pinocchio q = [p, quat xyzw, joints], v = [v_body, w_body, dq];
                                      t is the gait clock, tau the time since the plan started
    knots  (N_KNOT, 12) row-major     joint-target offsets from HOME at tau = j * KNOT_DT, linear in between
    rp     [goal_x goal_y goal_yaw gait_phase]
    par    PARAMS, in that order      gains, contact and cost parameters (inputs, so tuning needs no rebuild)

`n_sub` fuses that many steps of DT into one function.
"""
import os
import sys

import casadi as ca
import numpy as np

import course

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "menagerie_bench"))
XML = os.path.join(HERE, "..", "mujoco_menagerie", "unitree_go2", "go2.xml")

NQ, NV, NJ, NX = 19, 18, 12, 40
IT, ITAU, ICOST = 37, 38, 39
N_KNOT = 5
HOME = np.array([0.0, 0.9, -1.8] * 4)                 # the Menagerie "home" keyframe
HOME_Z = 0.27
TAU_MAX = np.array([23.7, 23.7, 45.43] * 4)           # go2.xml motor ctrlrange
BODY_LEN = 0.3868                                     # hip to hip, go2.xml
BRIDGE = (0.2, 0.4)                                   # offsets of the gap-bridging terrain samples [m]
FOOT_R = 0.022                                        # go2.xml foot sphere

# Contact points: (parent body, position in that body, radius, is_foot).
LEGS = ["FL", "FR", "RL", "RR"]
CONTACTS = ([(f"{l}_calf", (-0.002, 0.0, -0.213), FOOT_R, True) for l in LEGS]
            + [(f"{l}_calf", (0.0, 0.0, 0.0), 0.03, False) for l in LEGS]                 # knees
            + [("base", (sx * 0.24, sy * 0.05, -0.05), 0.01, False) for sx in (1, -1) for sy in (1, -1)]
            + [("base", (0.0, 0.0, -0.057), 0.0, False)])                                # belly
TROT = np.array([0.0, 0.5, 0.5, 0.0])                 # phase offset per leg (FL FR RL RR)

PARAMS = [                      # name, default
    ("kp", 50.0), ("kd", 0.5),                        # go2_mjx.xml position actuators
    ("k_n", 15000.0), ("a_n", 1.0),                   # normal force k d (1 - a_n * v_n)
    ("mu", 0.8), ("v_eps", 0.1),                      # friction -mu f_n v_t / sqrt(|v_t|^2 + v_eps^2)
    ("v_max", 0.6), ("d_slow", 0.5),                  # speed command toward the goal, ramped down inside d_slow
    ("h0", 0.29), ("gait_hz", 2.0), ("duty", 0.45), ("swing", 0.15),
    ("look", 0.2),                                   # a swing foot clears the terrain this far ahead of it as well
    ("gait_clip", 1.0),                               # the foot-height error saturates here (1 = never)
    ("body_look", 0.0),                               # 1: the height and pitch references also look `look` ahead of the feet
    ("pitch_k", 1.0), ("h_rise", 0.0),                # body pitch follows this fraction of the slope under the feet; extra height per rise
    ("w_vel", 3.0), ("w_yaw", 0.5), ("w_ang", 0.02), ("w_up", 10.0), ("w_h", 100.0), ("w_gait", 150.0),
    ("w_ctrl", 0.1), ("w_tau", 1e-5), ("w_body", 10.0), ("w_slip", 0.1), ("w_pos", 0.2), ("w_jv", 1e-4),
    ("w_foot", 5.0),                                 # feet stay under their hips (horizontal distance)
]
PAR = {n: i for i, (n, _) in enumerate(PARAMS)}
PAR_DEFAULT = np.array([v for _, v in PARAMS])


class CA:
    tanh, fmin, fmax = staticmethod(ca.tanh), staticmethod(ca.fmin), staticmethod(ca.fmax)


def terrain_fn():
    x, y = ca.SX.sym("x"), ca.SX.sym("y")
    h = course.height(x, y, CA)
    return ca.Function("terrain", [x, y], [h, ca.jacobian(h, x), ca.jacobian(h, y)])


def load(dt):
    """Pinocchio model with the contact frames added and the joint damping folded into the armature
    (implicit in dt: (M + dt D) a = tau - D v - h), and the MuJoCo model it came from."""
    import model as menagerie
    import pinocchio as pin
    m = menagerie.load_mujoco(XML)
    pm = menagerie.load_pinocchio(XML, m=m)
    damping = np.asarray(m.dof_damping).copy()
    pm.armature = np.asarray(m.dof_armature) + dt * damping
    fids = []
    for i, (body, pos, _, _) in enumerate(CONTACTS):
        b = pm.frames[pm.getFrameId(body, pin.FrameType.BODY)]
        fr = pin.Frame(f"contact{i}", b.parentJoint, b.placement * pin.SE3(np.eye(3), np.array(pos)), pin.FrameType.OP_FRAME)
        fids.append(pm.addFrame(fr))
    return m, pm, damping, fids


def build_step(dt, n_sub=1, name="go2_step"):
    import pinocchio as pin
    import pinocchio.casadi as cpin
    m, pm, damping, fids = load(dt)
    cm = cpin.Model(pm)
    terr = terrain_fn()
    x0 = ca.SX.sym("x", NX)
    knots = ca.SX.sym("knots", N_KNOT * NJ)
    rp = ca.SX.sym("rp", 4)
    par = ca.SX.sym("par", len(PARAMS))
    P = lambda n: par[PAR[n]]
    lo, hi = pm.lowerPositionLimit[7:], pm.upperPositionLimit[7:]
    knot_dt = ca.SX.sym("knot_dt")        # placeholder, substituted by the caller's constant below

    def one(x):
        q, v, t, tau_c, cost = x[:NQ], x[NQ:NQ + NV], x[IT], x[ITAU], x[ICOST]
        cd = cm.createData()
        cpin.forwardKinematics(cm, cd, q, v)
        cpin.updateFramePlacements(cm, cd)
        R = cd.oMi[1].rotation
        p = q[:3]
        vw = ca.mtimes(R, v[:3])
        # ---- control: linear interpolation of the knots, PD, torque limits
        s = tau_c / knot_dt
        a = 0
        for j in range(N_KNOT):
            wj = ca.fmax(0, 1 - ca.fabs(s - j)) if j < N_KNOT - 1 else ca.fmin(1, ca.fmax(0, s - (j - 1)))
            a = a + wj * knots[j * NJ:(j + 1) * NJ]
        q_des = ca.fmin(ca.fmax(HOME + a, lo), hi)
        qj, dq = q[7:], v[6:]
        trq = ca.fmin(ca.fmax(P("kp") * (q_des - qj) - P("kd") * dq, -TAU_MAX), TAU_MAX)
        # ---- penalty contact at every point
        fext = [cpin.Force.Zero() for _ in range(cm.njoints)]
        foot_h, foot_z, foot_xy, foot_slip, body_f = [], [], [], 0, 0
        for fid, (_, _, rad, is_foot) in zip(fids, CONTACTS):
            fr = pm.frames[fid]
            pc = cd.oMf[fid].translation
            vc = cpin.getFrameVelocity(cm, cd, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
            h, hx, hy = terr(pc[0], pc[1])
            nz = 1 / ca.sqrt(1 + hx * hx + hy * hy)
            n = ca.vertcat(-hx * nz, -hy * nz, nz)
            d = rad + (h - pc[2]) * nz                                   # penetration along the normal
            vn = ca.dot(vc, n)
            fn = ca.if_else(d > 0, ca.fmax(0, P("k_n") * d * (1 - P("a_n") * vn)), 0)
            vt = vc - vn * n
            f = fn * n - P("mu") * fn * vt / ca.sqrt(ca.dot(vt, vt) + P("v_eps") ** 2)
            Rj = cd.oMi[fr.parentJoint].rotation
            fl = ca.mtimes(Rj.T, f)
            r = fr.placement.translation
            fext[fr.parentJoint] = fext[fr.parentJoint] + cpin.Force(fl, ca.cross(ca.DM(r), fl))
            if is_foot:
                foot_h.append(h); foot_z.append(pc[2]); foot_xy.append(pc[:2])
                foot_slip = foot_slip + ca.if_else(d > 0, ca.dot(vt, vt), 0)
            else:
                body_f = body_f + fn
        # ---- running cost at the pre-step state
        e = rp[:2] - p[:2]
        dist = ca.sqrt(ca.dot(e, e) + 1e-8)
        v_cmd = P("v_max") * e / ca.fmax(dist, P("d_slow"))
        yaw = ca.atan2(R[1, 0], R[0, 0])
        yaw_err = ca.atan2(ca.sin(yaw - rp[2]), ca.cos(yaw - rp[2]))
        move = ca.fmin(1, (ca.sqrt(ca.dot(v_cmd, v_cmd) + 1e-8) + 0.3 * ca.fabs(yaw_err)) / 0.15)
        # support height under each foot: the terrain with gaps bridged along the direction of travel
        # (a hole narrower than about 2 * BRIDGE[0] counts as level with its rims)
        u = e / dist
        sup = []
        for i in range(4):
            rim = [ca.fmax(terr(foot_xy[i][0] + s * BRIDGE[0] * u[0], foot_xy[i][1] + s * BRIDGE[0] * u[1])[0],
                           terr(foot_xy[i][0] + s * BRIDGE[1] * u[0], foot_xy[i][1] + s * BRIDGE[1] * u[1])[0]) for s in (-1, 1)]
            sup.append(ca.fmax(foot_h[i], ca.fmin(rim[0], rim[1])))
        # terrain `look` ahead of each foot: a swing foot clears it; optionally the body prepares for it too
        ahead_h = [terr(foot_xy[i][0] + P("look") * u[0], foot_xy[i][1] + P("look") * u[1])[0] for i in range(4)]
        sup_look = [ca.fmax(sup[i], P("body_look") * ahead_h[i]) for i in range(4)]
        h_feet = sum(sup_look) / 4
        c_gait, c_foot = 0, 0
        for i in range(4):
            ph = P("gait_hz") * t + TROT[i] + rp[3]
            ph = ph - ca.floor(ph)
            sw = (ph - P("duty")) / (1 - P("duty"))
            z_ref = move * P("swing") * ca.if_else(sw > 0, ca.sin(np.pi * sw), 0)
            z_tar = ca.fmax(sup[i], ca.if_else(sw > 0, ahead_h[i], 0)) + z_ref
            err = ca.fmin(ca.fabs(foot_z[i] - FOOT_R - z_tar), P("gait_clip"))    # saturated: an unreachable target is not a wall
            c_gait = c_gait + err ** 2
            hip = cd.oMi[cm.getJointId(LEGS[i] + "_thigh_joint")].translation[:2]
            c_foot = c_foot + ca.dot(foot_xy[i] - hip, foot_xy[i] - hip)
        dh = (sup_look[0] + sup_look[1] - sup_look[2] - sup_look[3]) / 2             # the body pitches with the terrain under its feet
        slope = P("pitch_k") * dh / ca.sqrt(dh * dh + BODY_LEN ** 2)
        dv = vw[:2] - v_cmd
        terms = {"vel": P("w_vel") * ca.dot(dv, dv), "yaw": P("w_yaw") * yaw_err ** 2, "ang": P("w_ang") * ca.dot(v[3:6], v[3:6]),
                 "up": P("w_up") * ((R[2, 0] - slope) ** 2 + R[2, 1] ** 2),
                 "h": P("w_h") * (p[2] - h_feet - P("h0") - P("h_rise") * ca.fabs(dh)) ** 2,
                 "gait": P("w_gait") * c_gait, "ctrl": P("w_ctrl") * ca.dot(q_des - HOME, q_des - HOME), "tau": P("w_tau") * ca.dot(trq, trq),
                 "body": P("w_body") * body_f / 100, "slip": P("w_slip") * foot_slip, "pos": P("w_pos") * dist, "jv": P("w_jv") * ca.dot(dq, dq),
                 "foot": P("w_foot") * c_foot}
        c = sum(terms.values())
        # ---- semi-implicit Euler, joint damping implicit through the armature
        gen = ca.vertcat(ca.SX.zeros(6), trq - damping[6:] * dq)
        acc = cpin.aba(cm, cd, q, v, gen, fext)
        v1 = v + dt * acc
        q1 = cpin.integrate(cm, q, v1 * dt)
        return ca.vertcat(q1, v1, t + dt, tau_c + dt, cost + dt * c), terms

    x = x0
    for _ in range(n_sub):
        x, terms = one(x)
    return x0, knots, rp, par, knot_dt, x, name, terms


def make(dt, knot_dt, n_sub=1, name="go2_step"):
    x0, knots, rp, par, kd_sym, x, name, _ = build_step(dt, n_sub, name)
    x = ca.substitute(x, kd_sym, knot_dt)
    return ca.Function(name, [x0, knots, rp, par], [ca.cse(x)], ["x", "knots", "rp", "par"], ["x_next"])


def make_terms(dt, knot_dt):
    """The running-cost terms at a state (for diagnostics): terms(x, knots, rp, par) -> vector, names TERMS."""
    x0, knots, rp, par, kd_sym, _, _, terms = build_step(dt, 1, "terms")
    out = ca.substitute(ca.vertcat(*terms.values()), kd_sym, knot_dt)
    return ca.Function("terms", [x0, knots, rp, par], [out]), list(terms)


def initial_state(x, y, yaw, z=None):
    s = np.zeros(NX)
    s[:3] = (x, y, (HOME_Z + 0.02 if z is None else z) + float(course.height(x, y)))
    s[3:7] = (0, 0, np.sin(yaw / 2), np.cos(yaw / 2))
    s[7:19] = HOME
    return s
