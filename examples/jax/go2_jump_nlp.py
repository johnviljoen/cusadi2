#!/usr/bin/env python
"""Go2 jump as an inverse-dynamics NLP in JAX, exported to cuipm and solved with IPOPT.

The first whole-body problem of the Locomanipulation_Optimization_Proliferation
project in the cusadi2 / cuipm pipeline: the core of the hopscotch MPC
formulation (vault: Hopscotch_Jump_MPC_Formulation) in the lifted
inverse-dynamics transcription (vault: Inverse_Dynamics_Transcription_For_Sparse_KKT):

    variables   q_k (19), v_k (18) at knots k = 0..N;  a_k (18), tau_k (12), lam_k (4 feet x 3,
                world frame, at the foot bottom) over intervals k = 0..N-1
    parameters  measured state (q, v), landing footholds (4 x 3)
    equalities  q_0 = q_hat, v_0 = v_hat;  v_{k+1} = v_k + Ts a_k;  q_{k+1} = integrate(q_k, v_{k+1}, Ts)
                ID(q_k, v_k, a_k) - S^T tau_k - sum_i J_i(q_k)^T lam_{i,k} = 0
                feet in contact over interval k: J_i(q_{k+1}) v_{k+1} = 0 (no slip)
                touchdown knots: foot bottom = foothold
    inequalities contact intervals: friction pyramid mu lam_z >= |lam_x|, |lam_y|;
                swing intervals: foot bottom height >= clearance
    bounds      joint limits, torque limits, lam_z in [0, f_max] in contact, lam = 0 in swing
    cost        the formulation's regularisers (effort, posture, level base, rest at the end)

Two routes to the nine cuipm plans, chosen per experiment by `"staged"`: false traces
the whole horizon (`Go2Jump`, `export_cuipm`); true traces one stage (`Go2JumpStaged`,
the same problem as per-stage functions of q_k, v_k, q_{k+1}, v_{k+1}, a_k, tau_k,
tau_{k+1}, lam_k with the schedule as stage constants) and composes it
(`export_cuipm_staged`): the plans then run the stage graph per (instance, stage)
behind the same whole-horizon signatures, on the same x and p.

No command-line arguments: `go2_jump_nlp.json` next to this file lists the experiments;
every setting is explicit. Each experiment writes `runs/go2_jump_nlp/<name>/` with
config.json (settings, commit, dirty flag, versions), the cuipm plans + problem.json,
ipopt.json (reference solve), solution.npz and gpu_timing.json (the nine cusadi2 plans
of the same NLP, the exported Functions uploaded as plans, evaluated on the GPU).

    uv run --extra examples examples/jax/go2_jump_nlp.py
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CONFIG = os.path.join(HERE, "go2_jump_nlp.json")
RUNS = os.path.join(ROOT, "runs", "go2_jump_nlp")

REQUIRED = {
    "xml": str, "feet": list, "foot_offset": list, "foot_radius": float, "N": int, "Ts": float,
    "stance_until": int, "flight_until": int, "jump_dx": float, "jump_dy": float, "mu": float, "f_max": float,
    "clearance": float, "v_max": list, "rest_from": int, "weights": dict, "ipopt": dict, "gpu_batches": list,
    "run": bool, "staged": bool,
}
WEIGHTS = ("tau", "lam", "vj", "qj_home", "quat", "dtau", "rest_vb", "rest_vj", "rest_quat", "rest_qj",
           "terminal_quat", "terminal_vb")


def load_config(path):
    with open(path) as fh:
        cfg = json.load(fh)
    for name, exp in cfg.items():
        for k, t in REQUIRED.items():
            if k not in exp:
                raise KeyError(f"{name}: missing key {k!r}")
            if t is float and isinstance(exp[k], int):
                exp[k] = float(exp[k])
            if not isinstance(exp[k], t):
                raise TypeError(f"{name}: key {k!r} must be {t.__name__}")
        unknown = set(exp) - set(REQUIRED)
        if unknown:
            raise KeyError(f"{name}: unknown keys {sorted(unknown)}")
        missing = set(WEIGHTS) - set(exp["weights"])
        if missing or set(exp["weights"]) - set(WEIGHTS):
            raise KeyError(f"{name}: weights must be exactly {WEIGHTS}")
    return cfg


# ----------------------------------------------------------------------------- the NLP
class Go2Jump:
    def __init__(self, exp):
        import mujoco
        from cusadi2.jax import rbd
        self.exp = exp
        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, exp["xml"]))
        self.model = rbd.RigidBodyModel.from_mujoco(self.m)
        self.nq, self.nv, self.nj = self.model.nq, self.model.nv, self.model.nv - 6
        self.N, self.Ts = exp["N"], exp["Ts"]
        self.feet = [self.model.body_names.index(b) for b in exp["feet"]]
        self.off = np.asarray(exp["foot_offset"], float)
        self.R = exp["foot_radius"]
        N = self.N
        # contact schedule c[k][i]: stance - flight - stance, all feet together (the jump gait)
        self.contact = np.array([[not (exp["stance_until"] <= k < exp["flight_until"])] * 4 for k in range(N)])
        self.touchdown = [k for k in range(1, N + 1)
                          if k == exp["flight_until"]]        # knot where all feet land
        # layout of z
        n_knot, n_int = (N + 1), N
        self.o_q, self.o_v = 0, n_knot * self.nq
        self.o_a = self.o_v + n_knot * self.nv
        self.o_tau = self.o_a + n_int * self.nv
        self.o_lam = self.o_tau + n_int * self.nj
        self.nx = self.o_lam + n_int * 12
        self.np = self.nq + self.nv + 12
        key = [k for k in range(self.m.nkey) if mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_KEY, k) == "home"]
        self.q_home = np.array(self.m.key_qpos[key[0]], float)
        self.qj_home = self.q_home[7:]
        # put the feet on the floor: shift the base height so the lowest foot bottom is at z = 0
        xpos, xmat = self.model.forward_kinematics(jnp.asarray(self.q_home))
        bottoms = np.asarray(self.foot_bottoms(jnp.asarray(self.q_home)))
        self.q_home[2] -= bottoms[:, 2].min()
        self.q_hat = self.q_home.copy()
        self.v_hat = np.zeros(self.nv)
        b0 = np.asarray(self.foot_bottoms(jnp.asarray(self.q_hat)))
        self.footholds = b0 + np.array([exp["jump_dx"], exp["jump_dy"], 0.0])
        self.footholds[:, 2] = 0.0

    # ---- kinematics helpers (JAX)
    def foot_bottoms(self, q):
        xpos, xmat = self.model.forward_kinematics(q)
        return jnp.stack([xpos[b] + xmat[b] @ self.off - jnp.array([0.0, 0.0, self.R]) for b in self.feet])

    def foot_jacobians(self, q):
        """(4, 3, nv): velocity of each foot's bottom point in world axes w.r.t. qvel."""
        from cusadi2.jax.rbd import skew
        xpos, xmat = self.model.forward_kinematics(q)
        J = self.model.body_jacobian(q)
        ez = jnp.array([0.0, 0.0, 1.0])
        out = []
        for b in self.feet:
            J_center = J[b, :3] - skew(xmat[b] @ self.off) @ J[b, 3:]
            out.append(J_center + self.R * skew(ez) @ J[b, 3:])
        return jnp.stack(out)

    # ---- z layout
    def unpack(self, z):
        N, nq, nv, nj = self.N, self.nq, self.nv, self.nj
        q = z[self.o_q:self.o_v].reshape(N + 1, nq)
        v = z[self.o_v:self.o_a].reshape(N + 1, nv)
        a = z[self.o_a:self.o_tau].reshape(N, nv)
        tau = z[self.o_tau:self.o_lam].reshape(N, nj)
        lam = z[self.o_lam:].reshape(N, 4, 3)
        return q, v, a, tau, lam

    def pack(self, q, v, a, tau, lam):
        return np.concatenate([np.asarray(q).ravel(), np.asarray(v).ravel(), np.asarray(a).ravel(),
                               np.asarray(tau).ravel(), np.asarray(lam).ravel()])

    def params(self):
        return np.concatenate([self.q_hat, self.v_hat, self.footholds.ravel()])

    # ---- f, c, d
    def f(self, z, p):
        w = self.exp["weights"]
        q, v, a, tau, lam = self.unpack(z)
        J = 0.0
        for k in range(self.N):
            qn, vn = q[k + 1], v[k + 1]
            quat_xyz, vj, qj = qn[4:7], vn[6:], qn[7:]
            J = J + w["tau"] * jnp.sum(tau[k] ** 2) + w["lam"] * jnp.sum(lam[k] ** 2) + w["vj"] * jnp.sum(vj ** 2) \
                + w["qj_home"] * jnp.sum((qj - self.qj_home) ** 2) + w["quat"] * jnp.sum(quat_xyz ** 2)
            if k >= self.exp["rest_from"]:
                J = J + w["rest_vb"] * jnp.sum(vn[:6] ** 2) + w["rest_vj"] * jnp.sum(vj ** 2) \
                    + w["rest_quat"] * jnp.sum(quat_xyz ** 2) + w["rest_qj"] * jnp.sum((qj - self.qj_home) ** 2)
            if k + 1 < self.N:
                J = J + w["dtau"] * jnp.sum((tau[k + 1] - tau[k]) ** 2)
        J = J + w["terminal_quat"] * jnp.sum(q[self.N][4:7] ** 2) + w["terminal_vb"] * jnp.sum(v[self.N][:6] ** 2)
        return J

    def c(self, z, p):
        q, v, a, tau, lam = self.unpack(z)
        q_hat, v_hat = p[:self.nq], p[self.nq:self.nq + self.nv]
        footholds = p[self.nq + self.nv:].reshape(4, 3)
        eqs = [q[0] - q_hat, v[0] - v_hat]
        for k in range(self.N):
            eqs.append(v[k + 1] - v[k] - self.Ts * a[k])
            eqs.append(q[k + 1] - self.model.integrate_position(q[k], v[k + 1], self.Ts))
            Jk = self.foot_jacobians(q[k])
            gen_contact = sum((Jk[i].T @ lam[k, i] for i in range(4)), jnp.zeros(self.nv))
            eqs.append(self.model.inverse_dynamics(q[k], v[k], a[k]) - jnp.concatenate([jnp.zeros(6), tau[k]]) - gen_contact)
            if self.contact[k].any():
                Jn = self.foot_jacobians(q[k + 1])
                for i in range(4):
                    if self.contact[k][i]:
                        eqs.append(Jn[i] @ v[k + 1])
        for k in self.touchdown:
            b = self.foot_bottoms(q[k])
            for i in range(4):
                eqs.append(b[i] - footholds[i])
        return jnp.concatenate(eqs)

    def d(self, z, p):
        q, v, a, tau, lam = self.unpack(z)
        mu = self.exp["mu"]
        ineqs = []
        for k in range(self.N):
            swing = [i for i in range(4) if not self.contact[k][i]]
            if swing:
                b = self.foot_bottoms(q[k + 1])
                for i in swing:
                    ineqs.append(b[i, 2:3] - self.exp["clearance"])
            for i in range(4):
                if self.contact[k][i]:
                    lx, ly, lz = lam[k, i, 0], lam[k, i, 1], lam[k, i, 2]
                    ineqs.append(jnp.stack([mu * lz - lx, mu * lz + lx, mu * lz - ly, mu * lz + ly]))
        return jnp.concatenate(ineqs)

    # ---- bounds and initial guess
    def bounds(self):
        # no bound = inf (a finite +-1e20 was a real bound to cuipm and MadNLP: slacks of ~1e20 whose
        # multipliers swamped the complementarity; IPOPT alone dropped |b| >= 1e19)
        big = np.inf
        N, nq, nv, nj = self.N, self.nq, self.nv, self.nj
        q_L, q_U = np.full(nq, -big), np.full(nq, big)
        q_L[3:7], q_U[3:7] = -1.0, 1.0
        q_L[7:], q_U[7:] = self.m.jnt_range[1:, 0], self.m.jnt_range[1:, 1]
        v_L, v_U = np.full(nv, -big), np.full(nv, big)
        v_L[6:], v_U[6:] = -np.asarray(self.exp["v_max"] * 4), np.asarray(self.exp["v_max"] * 4)
        tau_max = np.asarray(self.m.actuator_ctrlrange[:, 1])
        lam_L, lam_U = np.zeros((N, 4, 3)), np.zeros((N, 4, 3))
        for k in range(N):
            for i in range(4):
                if self.contact[k][i]:
                    lam_L[k, i] = [-self.exp["f_max"], -self.exp["f_max"], 0.0]
                    lam_U[k, i] = [self.exp["f_max"], self.exp["f_max"], self.exp["f_max"]]
        x_L = self.pack(np.tile(q_L, (N + 1, 1)), np.tile(v_L, (N + 1, 1)), np.full((N, nv), -big), np.tile(-tau_max, (N, 1)), lam_L)
        x_U = self.pack(np.tile(q_U, (N + 1, 1)), np.tile(v_U, (N + 1, 1)), np.full((N, nv), big), np.tile(tau_max, (N, 1)), lam_U)
        return x_L, x_U

    def initial_guess(self):
        N = self.N
        q = np.tile(self.q_hat, (N + 1, 1))
        v = np.zeros((N + 1, self.nv))
        a = np.zeros((N, self.nv))
        tau = np.zeros((N, self.nj))
        lam = np.zeros((N, 4, 3))
        weight = float(self.m.body_subtreemass[1]) * 9.81
        for k in range(N):
            n_c = int(self.contact[k].sum())
            for i in range(4):
                if self.contact[k][i]:
                    lam[k, i, 2] = weight / n_c
        return self.pack(q, v, a, tau, lam)


class Go2JumpStaged(Go2Jump):
    """The same NLP as per-stage functions (`cusadi2.jax.ipm.StagedNLP`). Stage k holds the
    interval k: knots k (q_k, v_k) and k + 1 (q_k1, v_k1, read from the same knot buffers),
    a_k, tau_k, tau_{k+1} (for the torque-rate term; zero and gated off at the last stage),
    lam_k, the parameters p and the per-stage constants w_k = [torque-rate weight, rest flag,
    terminal flag]. Rows that exist only at some stages (initial state, no-slip and
    friction on stance intervals, touchdown, clearance on swing) are blocks with an activity
    mask; the global x and p are those of `Go2Jump`."""

    def __init__(self, exp):
        super().__init__(exp)
        N, w = self.N, exp["weights"]
        self.w_k = np.stack([np.where(np.arange(N) + 1 < N, w["dtau"], 0.0),
                             (np.arange(N) >= exp["rest_from"]).astype(float),
                             (np.arange(N) == N - 1).astype(float)], axis=1)
        self.specs = [dict(name="q_k", size=self.nq, mode="coupled", offset=0, span=1),
                      dict(name="v_k", size=self.nv, mode="coupled", offset=0, span=1),
                      dict(name="q_k1", size=self.nq, mode="coupled", offset=1, span=1, alias="q_k"),
                      dict(name="v_k1", size=self.nv, mode="coupled", offset=1, span=1, alias="v_k"),
                      dict(name="a_k", size=self.nv, mode="coupled", offset=0, span=0),
                      dict(name="tau_k", size=self.nj, mode="coupled", offset=0, span=0),
                      dict(name="tau_k1", size=self.nj, mode="coupled", offset=1, span=0, alias="tau_k"),
                      dict(name="lam_k", size=12, mode="coupled", offset=0, span=0),
                      dict(name="p", size=self.np, mode="shared", param=True),
                      dict(name="w_k", size=3, mode="stage", const=self.w_k)]
        from cusadi2.jax.ipm import rows_active
        stance = [self.contact[:, i] for i in range(4)]
        swing = [~self.contact[:, i] for i in range(4)]
        touchdown = np.array([k + 1 in self.touchdown for k in range(N)])
        self.c_active = rows_active(N, [(self.nq + self.nv, "first"), (self.nv + self.nq + self.nv, "all")]
                                    + [(3, stance[i]) for i in range(4)] + [(3, touchdown) for _ in range(4)])
        self.d_active = rows_active(N, [(1, swing[i]) for i in range(4)] + [(4, stance[i]) for i in range(4)])

    def stage_f(self, q_k, v_k, q_k1, v_k1, a_k, tau_k, tau_k1, lam_k, p, w_k):
        w = self.exp["weights"]
        quat_xyz, vj, qj = q_k1[4:7], v_k1[6:], q_k1[7:]
        J = w["tau"] * jnp.sum(tau_k ** 2) + w["lam"] * jnp.sum(lam_k ** 2) + w["vj"] * jnp.sum(vj ** 2) \
            + w["qj_home"] * jnp.sum((qj - self.qj_home) ** 2) + w["quat"] * jnp.sum(quat_xyz ** 2)
        J = J + w_k[1] * (w["rest_vb"] * jnp.sum(v_k1[:6] ** 2) + w["rest_vj"] * jnp.sum(vj ** 2)
                          + w["rest_quat"] * jnp.sum(quat_xyz ** 2) + w["rest_qj"] * jnp.sum((qj - self.qj_home) ** 2))
        J = J + w_k[0] * jnp.sum((tau_k1 - tau_k) ** 2)
        J = J + w_k[2] * (w["terminal_quat"] * jnp.sum(quat_xyz ** 2) + w["terminal_vb"] * jnp.sum(v_k1[:6] ** 2))
        return J

    def stage_c(self, q_k, v_k, q_k1, v_k1, a_k, tau_k, tau_k1, lam_k, p, w_k):
        q_hat, v_hat = p[:self.nq], p[self.nq:self.nq + self.nv]
        footholds = p[self.nq + self.nv:].reshape(4, 3)
        lam = lam_k.reshape(4, 3)
        Jk = self.foot_jacobians(q_k)
        gen_contact = sum((Jk[i].T @ lam[i] for i in range(4)), jnp.zeros(self.nv))
        Jn = self.foot_jacobians(q_k1)
        b = self.foot_bottoms(q_k1)
        return jnp.concatenate(
            [q_k - q_hat, v_k - v_hat,
             v_k1 - v_k - self.Ts * a_k,
             q_k1 - self.model.integrate_position(q_k, v_k1, self.Ts),
             self.model.inverse_dynamics(q_k, v_k, a_k) - jnp.concatenate([jnp.zeros(6), tau_k]) - gen_contact]
            + [Jn[i] @ v_k1 for i in range(4)]
            + [b[i] - footholds[i] for i in range(4)])

    def stage_d(self, q_k, v_k, q_k1, v_k1, a_k, tau_k, tau_k1, lam_k, p, w_k):
        mu = self.exp["mu"]
        lam = lam_k.reshape(4, 3)
        b = self.foot_bottoms(q_k1)
        return jnp.concatenate(
            [b[i, 2:3] - self.exp["clearance"] for i in range(4)]
            + [jnp.stack([mu * lam[i, 2] - lam[i, 0], mu * lam[i, 2] + lam[i, 0], mu * lam[i, 2] - lam[i, 1], mu * lam[i, 2] + lam[i, 1]])
               for i in range(4)])

    def staged_nlp(self):
        from cusadi2.jax.ipm import StagedNLP
        nlp = StagedNLP(self.stage_f, self.stage_c, self.stage_d, self.N, self.specs,
                        c_active=self.c_active, d_active=self.d_active)
        assert (nlp.nx, nlp.np) == (self.nx, self.np), "the staged layout must be Go2Jump's z and p"
        return nlp


def jump_contact(N, stance_until, flight_until, before, after):
    """(N, 4) contact over the intervals of one jump transition: the feet `before` (4 bools) down until
    stance_until, all four up in flight, the feet `after` down from flight_until."""
    c = np.zeros((N, 4), bool)
    c[:stance_until] = np.asarray(before, bool)
    c[flight_until:] = np.asarray(after, bool)
    return c


class Go2JumpSchedule(Go2JumpStaged):
    """Go2JumpStaged with a per-foot schedule: one jump transition (the jump timing of `exp`) from the
    stance `before` to the stance `after` (4 bools each, FL FR RL RR; a foot not down dangles), so
    windows with 4-, 3- or 2-foot stances. The feet of `after` are pinned to their footholds in p at
    the touchdown knot flight_until; p is Go2Jump's (measured state, one foothold per foot).

    union=False: the schedule is structure (activity masks, lam bounds), as Go2JumpStaged.
    union=True: every row block at every stage (no-slip, touchdown pin, clearance and friction cone for
    all four feet) and the stance lam bounds on every interval, so one export (one sparsity pattern)
    holds every schedule; a problem's schedule enters only through its own bounds
    (`schedule_bounds`): rows it does not have are masked (+-inf; relaxed-equality rows then become
    free rows), the lam of feet in the air are fixed at 0 (x_L = x_U = 0; cuipm_jax
    fixed_variables="parameter"). Intended for the relaxed export (StagedNLP(relax_equality=True))."""

    def __init__(self, exp, before=(True,) * 4, after=(True,) * 4, union=False):
        super().__init__(exp)
        from cusadi2.jax.ipm import rows_active
        N = self.N
        self.union = bool(union)
        self.set_schedule(before, after)
        on = np.ones(N, bool)
        stance = [on] * 4 if union else [self.contact[:, i] for i in range(4)]
        swing = [on] * 4 if union else [~self.contact[:, i] for i in range(4)]
        pin = [on] * 4 if union else [self.pin_stages(self.land)[:, i] for i in range(4)]
        self.c_active = rows_active(N, [(self.nq + self.nv, "first"), (self.nv + self.nq + self.nv, "all")]
                                    + [(3, stance[i]) for i in range(4)] + [(3, pin[i]) for i in range(4)])
        self.d_active = rows_active(N, [(1, swing[i]) for i in range(4)] + [(4, stance[i]) for i in range(4)])

    def set_schedule(self, before, after):
        """The schedule used by initial_guess() and (union=False) bounds(); for union=True the activity
        masks and bounds() stay the union's."""
        self.before, self.after = np.asarray(before, bool).copy(), np.asarray(after, bool).copy()
        self.contact = jump_contact(self.N, self.exp["stance_until"], self.exp["flight_until"], self.before, self.after)
        self.land = self.after.copy()

    def pin_stages(self, land):
        """(N, 4): stage k pins foot i (its bottom at knot k + 1) iff k + 1 is the touchdown knot and i lands."""
        td = np.arange(self.N) + 1 == self.exp["flight_until"]
        return td[:, None] & np.asarray(land, bool)[None, :]

    def bounds(self):
        if not self.union:
            return super().bounds()
        c = self.contact
        self.contact = np.ones_like(c)
        try:
            return super().bounds()
        finally:
            self.contact = c

    def schedule_bounds(self, nlp, d_L, d_U, before, after):
        """Per-problem bounds (x_L, x_U, d_L, d_U) on the union's relaxed export `nlp` (its d_L, d_U) for
        the schedule before -> after: lam of feet in the air x_L = x_U = 0, rows the schedule does not
        have masked (+-inf)."""
        if not (self.union and nlp.relax_equality):
            raise ValueError("schedule_bounds: needs the union class and its relaxed export")
        contact = jump_contact(self.N, self.exp["stance_until"], self.exp["flight_until"], before, after)
        x_L, x_U = (v.copy() for v in self.bounds())
        lam_air = np.zeros((self.N, 4, 3), bool)
        lam_air[~contact] = True
        sel = self.pack(np.zeros((self.N + 1, self.nq)), np.zeros((self.N + 1, self.nv)), np.zeros((self.N, self.nv)),
                        np.zeros((self.N, self.nj)), lam_air).astype(bool)
        x_L[sel], x_U[sel] = 0.0, 0.0
        d_L, d_U = np.asarray(d_L, float).copy(), np.asarray(d_U, float).copy()
        nc = self.c_active.shape[1]
        ns0 = (self.nq + self.nv) + (self.nv + self.nq + self.nv)
        pin = self.pin_stages(after)
        off = {"noslip": ns0, "pin": ns0 + 12, "clearance": nc, "cone": nc + 4}
        size = {"noslip": 3, "pin": 3, "clearance": 1, "cone": 4}
        drop = {"noslip": ~contact, "pin": ~pin, "clearance": contact, "cone": ~contact}   # (N, 4) masked stage/foot
        for kind, m in drop.items():
            for i in range(4):
                a = off[kind] + i * size[kind]
                r = nlp.d_rows[m[:, i], a:a + size[kind]]
                r = r[r >= 0]
                d_L[r], d_U[r] = -np.inf, np.inf
        return x_L, x_U, d_L, d_U


def course_contact(phase_knots, contacts):
    """(N, 4) contact over the intervals of a phase list: phase j covers phase_knots[j] intervals with
    the feet contacts[j] (4 bools) down."""
    return np.concatenate([np.repeat(np.asarray(c, bool)[None], n, axis=0) for n, c in zip(phase_knots, contacts)])


class Go2Course(Go2JumpStaged):
    """A whole hopscotch course as one NLP: phases (knot counts `phase_knots`, feet down `contacts`, the
    same timing for every problem), N = sum(phase_knots) must equal exp["N"]. Landing events are the
    phases j >= 1 with feet down (`events`, E of them); event j is at knot K_j = sum(phase_knots[:j]). A foot
    lands at event j iff it is down in phase j and not in phase j - 1; its bottom is pinned to its
    target for that event, p = [q_hat, v_hat, targets (E, 4, 3)]. The event of a stage comes from the
    per-stage constant w_k = [torque-rate weight, rest flag, terminal flag, one-hot event (E)]: stage k
    pins its knot k + 1 to sum_j w_k[3 + j] target_j. Fixed timing makes the event knots the same for
    every problem, so only which feet land differs (masks).

    union=False: the schedule as structure. union=True: no-slip, pin, clearance and cone for all feet at
    every stage, stance lam bounds everywhere; a problem's schedule only through `schedule_bounds`.
    With phase_knots [stance_until, flight_until - stance_until, N - flight_until] and four feet down
    before and after, it is Go2JumpStaged's NLP (same x, p)."""

    def __init__(self, exp, phase_knots, contacts, union=False):
        Go2JumpStaged.__init__(self, exp)
        from cusadi2.jax.ipm import rows_active
        N, w = self.N, exp["weights"]
        self.phase_knots = [int(n) for n in phase_knots]
        if sum(self.phase_knots) != N:
            raise ValueError(f"Go2Course: phase knots sum to {sum(self.phase_knots)}, N = {N}")
        self.starts = np.cumsum([0] + self.phase_knots[:-1])
        # landing events: the phases j >= 1 with feet down (the phase types are the same for every
        # problem of a union: flights stay flights)
        self.events = [j for j in range(1, len(self.phase_knots)) if np.asarray(contacts[j], bool).any()]
        self.E = len(self.events)
        self.union = bool(union)
        self.set_schedule(contacts)
        self.np = self.nq + self.nv + 12 * self.E
        self.targets = np.zeros((self.E, 4, 3))
        onehot = np.zeros((N, self.E))
        for e, j in enumerate(self.events):
            onehot[self.starts[j] - 1, e] = 1.0                         # stage K_j - 1 holds knot K_j
        self.w_k = np.concatenate([np.stack([np.where(np.arange(N) + 1 < N, w["dtau"], 0.0),
                                             (np.arange(N) >= exp["rest_from"]).astype(float),
                                             (np.arange(N) == N - 1).astype(float)], axis=1), onehot], axis=1)
        self.specs = [dict(sp) for sp in self.specs]
        for sp in self.specs:
            if sp["name"] == "p":
                sp["size"] = self.np
            if sp["name"] == "w_k":
                sp["size"], sp["const"] = 3 + self.E, self.w_k
        on = np.ones(N, bool)
        stance = [on] * 4 if union else [self.contact[:, i] for i in range(4)]
        swing = [on] * 4 if union else [~self.contact[:, i] for i in range(4)]
        pin = [on] * 4 if union else [self.pin_stages(self.land)[:, i] for i in range(4)]
        self.c_active = rows_active(N, [(self.nq + self.nv, "first"), (self.nv + self.nq + self.nv, "all")]
                                    + [(3, stance[i]) for i in range(4)] + [(3, pin[i]) for i in range(4)])
        self.d_active = rows_active(N, [(1, swing[i]) for i in range(4)] + [(4, stance[i]) for i in range(4)])

    def landings(self, contacts):
        """(E, 4) bool: foot i lands at event e (phase events[e])."""
        c = np.asarray(contacts, bool)
        return np.stack([c[j] & ~c[j - 1] for j in self.events])

    def set_schedule(self, contacts):
        self.contacts = np.asarray(contacts, bool).copy()
        if self.contacts.shape != (len(self.phase_knots), 4):
            raise ValueError(f"Go2Course: contacts must be ({len(self.phase_knots)}, 4)")
        if any(not self.contacts[j].any() for j in self.events):
            raise ValueError("Go2Course: a landing-event phase has no foot down")
        self.contact = course_contact(self.phase_knots, self.contacts)
        self.land = self.landings(self.contacts)

    def pin_stages(self, land):
        """(N, 4): stage K_j - 1 pins foot i iff it lands at event j."""
        out = np.zeros((self.N, 4), bool)
        for e, j in enumerate(self.events):
            out[self.starts[j] - 1] = np.asarray(land, bool)[e]
        return out

    def params(self):
        return np.concatenate([self.q_hat, self.v_hat, self.targets.ravel()])

    def bounds(self):
        if not self.union:
            return Go2Jump.bounds(self)
        c = self.contact
        self.contact = np.ones_like(c)
        try:
            return Go2Jump.bounds(self)
        finally:
            self.contact = c

    def stage_c(self, q_k, v_k, q_k1, v_k1, a_k, tau_k, tau_k1, lam_k, p, w_k):
        q_hat, v_hat = p[:self.nq], p[self.nq:self.nq + self.nv]
        T = p[self.nq + self.nv:].reshape(self.E, 12)
        target = (w_k[3:] @ T).reshape(4, 3)
        lam = lam_k.reshape(4, 3)
        Jk = self.foot_jacobians(q_k)
        gen_contact = sum((Jk[i].T @ lam[i] for i in range(4)), jnp.zeros(self.nv))
        Jn = self.foot_jacobians(q_k1)
        b = self.foot_bottoms(q_k1)
        return jnp.concatenate(
            [q_k - q_hat, v_k - v_hat,
             v_k1 - v_k - self.Ts * a_k,
             q_k1 - self.model.integrate_position(q_k, v_k1, self.Ts),
             self.model.inverse_dynamics(q_k, v_k, a_k) - jnp.concatenate([jnp.zeros(6), tau_k]) - gen_contact]
            + [Jn[i] @ v_k1 for i in range(4)]
            + [b[i] - target[i] for i in range(4)])

    def schedule_bounds(self, nlp, d_L, d_U, contacts):
        """Per-problem bounds (x_L, x_U, d_L, d_U) on the union's relaxed export for the phase contacts
        (P, 4): lam of feet in the air fixed at 0, rows the schedule does not have masked (+-inf)."""
        if not (self.union and nlp.relax_equality):
            raise ValueError("schedule_bounds: needs the union class and its relaxed export")
        contact = course_contact(self.phase_knots, contacts)
        x_L, x_U = (v.copy() for v in self.bounds())
        lam_air = np.zeros((self.N, 4, 3), bool)
        lam_air[~contact] = True
        sel = self.pack(np.zeros((self.N + 1, self.nq)), np.zeros((self.N + 1, self.nv)), np.zeros((self.N, self.nv)),
                        np.zeros((self.N, self.nj)), lam_air).astype(bool)
        x_L[sel], x_U[sel] = 0.0, 0.0
        d_L, d_U = np.asarray(d_L, float).copy(), np.asarray(d_U, float).copy()
        nc = self.c_active.shape[1]
        ns0 = (self.nq + self.nv) + (self.nv + self.nq + self.nv)
        pin = self.pin_stages(self.landings(contacts))
        off = {"noslip": ns0, "pin": ns0 + 12, "clearance": nc, "cone": nc + 4}
        size = {"noslip": 3, "pin": 3, "clearance": 1, "cone": 4}
        drop = {"noslip": ~contact, "pin": ~pin, "clearance": contact, "cone": ~contact}
        for kind, m in drop.items():
            for i in range(4):
                a = off[kind] + i * size[kind]
                r = nlp.d_rows[m[:, i], a:a + size[kind]]
                r = r[r >= 0]
                d_L[r], d_U[r] = -np.inf, np.inf
        return x_L, x_U, d_L, d_U


# ----------------------------------------------------------------------------- experiment
def git_info():
    def run(*args):
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def run_experiment(name, exp):
    import casadi
    import mujoco
    from cusadi2.jax.ipm import export_cuipm, export_cuipm_staged, gpu_functions, solve_ipopt, trace_nlp
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    start = datetime.now().isoformat(timespec="seconds")
    prob = Go2JumpStaged(exp) if exp["staged"] else Go2Jump(exp)
    z0, p0 = prob.initial_guess(), prob.params()
    x_L, x_U = prob.bounds()
    print(f"[{name}] nx={prob.nx} np={prob.np} N={prob.N} contact intervals={int(prob.contact.all(1).sum())} "
          f"flight={int((~prob.contact.any(1)).sum())} staged={exp['staged']}", flush=True)

    t = time.perf_counter()
    if exp["staged"]:
        staged = prob.staged_nlp()                      # one stage traced; the whole horizon by symbolic stage calls
        nlp = staged.whole_horizon()
        nyc, nyd = staged.nyc, staged.nyd
    else:
        nlp = trace_nlp(prob.f, prob.c, prob.d, prob.nx, prob.np)
        nyc, nyd = int(nlp[3].numel()), int(nlp[4].numel())
    t_trace = time.perf_counter() - t
    print(f"[{name}] traced to SX in {t_trace:.1f}s: nyc={nyc} nyd={nyd}", flush=True)

    t = time.perf_counter()
    if exp["staged"]:
        info = export_cuipm_staged(staged, z0, p0, os.path.join(out, "plans"), x_L=x_L, x_U=x_U,
                                   d_L=np.zeros(nyd), d_U=np.full(nyd, np.inf))
    else:
        info = export_cuipm(prob.f, prob.c, prob.d, z0, p0, os.path.join(out, "plans"), x_L=x_L, x_U=x_U,
                            d_L=np.zeros(nyd), d_U=np.full(nyd, np.inf), nlp=nlp)
    t_export = time.perf_counter() - t
    plans = {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in info["plans"].items()}
    print(f"[{name}] cuipm plans written in {t_export:.1f}s: " +
          ", ".join(f"{k} nnz={v['nnz']} live={v['peak_live_slots']} lvl={v['n_levels']}" for k, v in plans.items()), flush=True)

    res = solve_ipopt(None, None, None, z0, p0, x_L=x_L, x_U=x_U, d_L=np.zeros(nyd), d_U=np.full(nyd, np.inf),
                      options=exp["ipopt"], nlp=nlp)
    q, v, a, tau, lam = prob.unpack(res["x"])
    c_max = float(np.max(np.abs(res["g"][:nyc]))) if nyc else 0.0
    d_min = float(np.min(res["g"][nyc:])) if nyd else 0.0
    ipopt = {"status": res["status"], "success": res["success"], "iterations": res["iterations"], "wall_s": res["wall_s"],
             "f": res["f"], "max_abs_c": c_max, "min_d": d_min,
             "base_z_min": float(np.min(q[:, 2])), "base_z_max": float(np.max(q[:, 2])),
             "base_x_final": float(q[-1, 0]), "max_abs_tau": float(np.max(np.abs(tau))), "max_lam_z": float(np.max(lam[:, :, 2])),
             "foot_bottoms_final": np.asarray(prob.foot_bottoms(jnp.asarray(q[-1]))).tolist()}
    print(f"[{name}] IPOPT: {res['status']} in {res['iterations']} iterations, {res['wall_s']:.2f}s; f={res['f']:.4g} "
          f"max|c|={c_max:.1e} min d={d_min:.1e}; base z {ipopt['base_z_min']:.3f}..{ipopt['base_z_max']:.3f}, "
          f"final x {ipopt['base_x_final']:.3f}, max|tau| {ipopt['max_abs_tau']:.1f}, max lam_z {ipopt['max_lam_z']:.1f}", flush=True)
    np.savez(os.path.join(out, "solution.npz"), x=res["x"], lam_c=res["lam_c"], lam_d=res["lam_d"], q=q, v=v, a=a, tau=tau,
             lam=lam, p=p0, x_L=x_L, x_U=x_U, z0=z0)
    with open(os.path.join(out, "ipopt.json"), "w") as fh:
        json.dump(ipopt, fh, indent=1)

    # the same nine functions as cusadi2 plans on the GPU (the exported Functions, uploaded as plans)
    gpu = {}
    if exp["gpu_batches"] and any(d.platform == "gpu" for d in jax.devices()):
        t = time.perf_counter()
        calls = gpu_functions(info["functions"])
        zj, pj, eps = jnp.asarray(res["x"]), jnp.asarray(p0), jnp.zeros(1)
        lam_c, lam_d = jnp.asarray(res["lam_c"]), jnp.asarray(res["lam_d"])
        for B in exp["gpu_batches"]:
            Z = jnp.tile(zj[None], (B, 1)); P = jnp.tile(pj[None], (B, 1)); E = jnp.tile(eps[None], (B, 1))
            Lc, Ld = jnp.tile(lam_c[None], (B, 1)), jnp.tile(lam_d[None], (B, 1))
            timings = {}
            for k, call in calls.items():
                g = jax.jit(jax.vmap(call))
                args = {"hess_c": (Z, P, Lc), "hess_d": (Z, P, Ld)}.get(k, (Z, P, E) if k.endswith("f") else (Z, P))
                jax.block_until_ready(g(*args))
                ts = []
                for _ in range(10):
                    t0 = time.perf_counter(); jax.block_until_ready(g(*args)); ts.append(time.perf_counter() - t0)
                timings[k] = float(np.median(ts)) * 1e6
            timings["all_nine"] = float(sum(timings.values()))
            gpu[str(B)] = timings
            print(f"[{name}] GPU batch {B}: " + ", ".join(f"{k} {v:.0f} us" for k, v in timings.items()), flush=True)
        gpu["plan_build_s"] = time.perf_counter() - t
        gpu["serial_fallback"] = sorted(k for k, g in calls.items() if g.serial)
        with open(os.path.join(out, "gpu_timing.json"), "w") as fh:
            json.dump(gpu, fh, indent=1)

    config = {"experiment": name, "settings": exp, "start": start, "git": git_info(),
              "versions": {"jax": jax.__version__, "casadi": casadi.__version__, "mujoco": mujoco.__version__,
                           "python": sys.version.split()[0]}, "host": os.uname().nodename,
              "dims": {"nx": prob.nx, "np": prob.np, "nyc": nyc, "nyd": nyd}, "trace_s": t_trace, "export_s": t_export,
              "plans": plans}
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)
    return config, ipopt, gpu


def main():
    cfg = load_config(CONFIG)
    for name, exp in cfg.items():
        if exp["run"]:
            run_experiment(name, exp)


if __name__ == "__main__":
    main()
