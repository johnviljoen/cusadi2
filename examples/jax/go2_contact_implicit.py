#!/usr/bin/env python
"""Go2 contact-implicit NLP in JAX: no contact schedule, exact Coulomb contact relaxed along the
central path (vault: Central_Path_Contact_Implicit_Formulation, Contact_Implicit_Go2_Discovery).

The lifted inverse-dynamics transcription of go2_jump_nlp.py with the four feet as contact pairs
against a terrain height h(x) (flat ground with one pit, tanh edges), vertical normal:

    variables   q_k (19), v_k (18) at knots k = 0..N;  a_k (18), tau_k (12), lam_k (4 feet x 3, world
                frame = contact frame: x, y tangential, z normal), gap_k (4), psi_k (4) over intervals
    parameters  measured state (q, v), goal base position (x, y), pit (x0, x1, depth),
                sink (4): per-foot lowering of the terrain, gap_i = phi_i + sink_i (0 unless set),
                ref (3): start x, y and the time of knot 0 of the base reference (cost_form "reference")
    equalities  q_0 = q_hat, v_0 = v_hat;  v_{k+1} = v_k + Ts a_k;  q_{k+1} = integrate(q_k, v_{k+1}, Ts)
                ID(q_k, v_k, a_k) - S^T tau_k - sum_i J_i(q_k)^T lam_{i,k} = 0
                gap_{i,k} = phi_i(q_{k+1}) = foot bottom z - h(foot bottom x)          (position level)
                v_t,i(q_{k+1}, v_{k+1}) + 2 psi_{i,k} lam_t,i,k = 0                    (maximum dissipation)
    inequalities knee height >= knee_min;  cone margin Phi_i = mu^2 lam_n^2 - |lam_t|^2 >= 0;
                friction margin (cone_margin mu lam_n)^2 - |lam_t|^2 >= 0: the plan uses at most the
                fraction cone_margin of the cone (the friction law keeps the true mu), which bounds the
                creep 2 kappa_t F / (r^2 - F^2) of a sticking foot
    homotopy rows  gap_i lam_n,i  and  psi_i Phi_i: the two complementarity products per foot. Their
                bounds carry the relaxation: [kappa, kappa] is the central path, [0, kappa] the
                one-sided relaxation, kappa_n = kappa_hat m_i g l_ref and kappa_t = kappa_hat m_i g v_ref
                with m_i g the foot's standing load (`homotopy_rows` gives rows and weights).
    bounds      joint, joint-speed and torque limits, base height >= base_z_min,
                gap >= 0, psi >= 0, 0 <= lam_n <= f_max, |lam_t| <= f_max
    cost        go2_jump_nlp's regularisers, a rest phase, a terminal base-position cost, and per foot
                and interval the friction dissipation Ts w_diss 2 psi |lam_t|^2 (= -lam_t . v_t) and a slip
                term Ts w_slip exp(-gap / slip_gap) |v_t|^2; slip_form "psi" writes v_t as -2 psi lam_t
                (the dissipation row: no kinematics in the cost, but quartic in psi, lam_t), "kinematic"
                takes the foot's tangential velocity J(q_{k+1}) v_{k+1}

No command-line arguments: `go2_contact_implicit.json` next to this file lists the experiments, every
setting explicit. An experiment checks the model (staged composition against a direct JAX
evaluation, Jacobians against central differences) and solves its fixed-kappa problem with IPOPT;
it writes `runs/go2_contact_implicit/<name>/` with config.json, checks.json, ipopt.json, solution.npz.

    uv run --extra examples examples/jax/go2_contact_implicit.py
"""
import json
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from go2_jump_nlp import ROOT, Go2Jump, git_info  # noqa: E402

CONFIG = os.path.join(HERE, "go2_contact_implicit.json")
RUNS = os.path.join(ROOT, "runs", "go2_contact_implicit")

MODEL = {
    "xml": str, "feet": list, "foot_offset": list, "foot_radius": float, "N": int, "Ts": float, "mu": float,
    "f_max": float, "v_max": list, "rest_from": int, "base_z_min": float, "knee_min": float, "pit_edge": float,
    "l_ref": float, "v_ref": float, "slip_gap": float, "slip_form": str, "cone_margin": float, "cost_form": str, "v_cmd": float, "scale_force_rows": bool, "weights": dict,
}
REQUIRED = {"run": bool, "model": dict, "goal": list, "pit": list, "kappa_hat": float, "bounds": str, "ipopt": dict,
            "check_seed": int}
WEIGHTS = ("tau", "lam", "vj", "qj_home", "quat", "dtau", "rest_vb", "rest_vj", "rest_quat", "rest_qj",
           "terminal_quat", "terminal_vb", "goal_pos", "diss", "slip", "dlam")
G = 9.81


def check_keys(d, keys, what):
    for k, t in keys.items():
        if k not in d:
            raise KeyError(f"{what}: missing key {k!r}")
        if t is float and isinstance(d[k], int) and not isinstance(d[k], bool):
            d[k] = float(d[k])
        if not isinstance(d[k], t):
            raise TypeError(f"{what}: key {k!r} must be {t.__name__}")
    if set(d) - set(keys):
        raise KeyError(f"{what}: unknown keys {sorted(set(d) - set(keys))}")


def check_model(m, what):
    check_keys(m, MODEL, what)
    if set(m["weights"]) != set(WEIGHTS):
        raise KeyError(f"{what}: weights must be exactly {WEIGHTS}")
    if m["cost_form"] not in ("fixed_end", "invariant", "reference"):
        raise ValueError(f"{what}: cost_form is 'fixed_end', 'invariant' or 'reference'")
    if m["slip_form"] not in ("psi", "kinematic"):
        raise ValueError(f"{what}: slip_form is 'psi' or 'kinematic'")


def load_config(path):
    with open(path) as fh:
        cfg = json.load(fh)
    for name, exp in cfg.items():
        check_keys(exp, REQUIRED, name)
        check_model(exp["model"], f"{name}.model")
        if exp["bounds"] not in ("central", "relaxed"):
            raise ValueError(f"{name}: bounds is 'central' ([kappa, kappa]) or 'relaxed' ([0, kappa])")
    return cfg


class Go2ContactImplicit(Go2Jump):
    """The NLP of the module docstring as per-stage functions (`cusadi2.jax.ipm.StagedNLP`). Stage k
    holds interval k: knots k and k + 1, a_k, tau_k, tau_{k+1} (torque-rate term), lam_k, gap_k, psi_k,
    p and the stage constants w_k = [torque-rate weight, rest flag, terminal flag, time of knot k + 1 after knot 0, force-rate weight]."""

    def __init__(self, model):
        import mujoco
        from cusadi2.jax import rbd
        from cusadi2.jax.ipm import rows_active
        self.exp = model
        self.m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, model["xml"]))
        self.model = rbd.RigidBodyModel.from_mujoco(self.m)
        self.nq, self.nv, self.nj = self.model.nq, self.model.nv, self.model.nv - 6
        self.N, self.Ts = model["N"], model["Ts"]
        self.feet = [self.model.body_names.index(b) for b in model["feet"]]
        self.off = np.asarray(model["foot_offset"], float)
        self.R = model["foot_radius"]
        N, nq, nv, nj = self.N, self.nq, self.nv, self.nj
        sizes = [(N + 1) * nq, (N + 1) * nv, N * nv, N * nj, N * 12, N * 4, N * 4]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.nx = int(self.offsets[-1])
        self.np = nq + nv + 2 + 3 + 4 + 3
        key = [k for k in range(self.m.nkey) if mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_KEY, k) == "home"]
        self.q_home = np.array(self.m.key_qpos[key[0]], float)
        self.qj_home = self.q_home[7:]
        self.q_home[2] -= np.asarray(self.foot_bottoms(jnp.asarray(self.q_home)))[:, 2].min()     # feet on z = 0
        self.q_hat, self.v_hat = self.q_home.copy(), np.zeros(nv)
        self.feet0 = np.asarray(self.foot_bottoms(jnp.asarray(self.q_hat)))
        self.load, self.tau_stand = self.standing_load()
        w = model["weights"]
        # cost_form "fixed_end": rest costs from stage rest_from, goal and terminal costs on the last knot (a
        # point-to-point problem with a fixed end time). "invariant": both on every stage, so the problem
        # one interval later is the same problem (receding horizon); rest_from is unused. "reference": the
        # goal and terminal terms on every stage against a base reference in absolute time (from the start
        # position toward the goal at v_cmd, then held; velocity reference to match), the rest terms from the
        # reference's arrival on; the problem one interval later only gains a last stage. rest_from is unused.
        every = model["cost_form"] in ("invariant", "reference")
        self.w_k = np.stack([np.where(np.arange(N) + 1 < N, w["dtau"], 0.0),
                             np.ones(N) if every else (np.arange(N) >= model["rest_from"]).astype(float),
                             np.ones(N) if every else (np.arange(N) == N - 1).astype(float),
                             (np.arange(N) + 1) * self.Ts,
                             np.where(np.arange(N) + 1 < N, w["dlam"], 0.0)], axis=1)
        self.specs = [dict(name="q_k", size=nq, mode="coupled", offset=0, span=1),
                      dict(name="v_k", size=nv, mode="coupled", offset=0, span=1),
                      dict(name="q_k1", size=nq, mode="coupled", offset=1, span=1, alias="q_k"),
                      dict(name="v_k1", size=nv, mode="coupled", offset=1, span=1, alias="v_k"),
                      dict(name="a_k", size=nv, mode="coupled", offset=0, span=0),
                      dict(name="tau_k", size=nj, mode="coupled", offset=0, span=0),
                      dict(name="tau_k1", size=nj, mode="coupled", offset=1, span=0, alias="tau_k"),
                      dict(name="lam_k", size=12, mode="coupled", offset=0, span=0),
                      dict(name="lam_k1", size=12, mode="coupled", offset=1, span=0, alias="lam_k"),
                      dict(name="gap_k", size=4, mode="coupled", offset=0, span=0),
                      dict(name="psi_k", size=4, mode="coupled", offset=0, span=0),
                      dict(name="p", size=self.np, mode="shared", param=True),
                      dict(name="w_k", size=5, mode="stage", const=self.w_k)]
        self.n_dyn = nv + nq + nv
        self.c_active = rows_active(N, [(nq + nv, "first"), (self.n_dyn, "all"), (4, "all"), (8, "all")])
        self.d_active = rows_active(N, [(4, "all"), (4, "all"), (4, "all"), (4, "all"), (4, "all")])

    # ---- statics of the measured state
    def standing_load(self):
        """(normal force per foot (4,), joint torques (12,)) holding q_hat still: the base rows of
        ID(q_hat, 0, 0) balanced by vertical foot forces (least squares; 4 unknowns, 6 rows)."""
        q = jnp.asarray(self.q_hat)
        rhs = np.asarray(self.model.inverse_dynamics(q, jnp.zeros(self.nv), jnp.zeros(self.nv)))
        J = np.asarray(self.foot_jacobians(q))                                 # (4, 3, nv)
        A = J[:, 2, :].T                                                       # (nv, 4): J_i^T e_z
        n = np.linalg.lstsq(A[:6], rhs[:6], rcond=None)[0]
        return n, rhs[6:] - (A @ n)[6:]

    # ---- terrain and contact kinematics (JAX)
    def terrain(self, x, pit):
        """Ground height under x: 0, and -depth inside the pit [x0, x1], tanh edges of width pit_edge."""
        s = lambda u: 0.5 * (1.0 + jnp.tanh(u / self.exp["pit_edge"]))
        return -pit[2] * s(x - pit[0]) * s(pit[1] - x)

    def gaps(self, q, pit):
        b = self.foot_bottoms(q)
        return b[:, 2] - self.terrain(b[:, 0], pit)

    def knees(self, q):
        xpos, _ = self.model.forward_kinematics(q)
        return jnp.stack([xpos[b][2] for b in self.feet])                      # the calf body's origin is the knee

    def cone(self, lam):
        mu = self.exp["mu"]
        return mu * mu * lam[:, 2] ** 2 - lam[:, 0] ** 2 - lam[:, 1] ** 2

    # ---- z layout: q, v (knots), a, tau, lam, gap, psi (intervals), each knot-major
    def unpack(self, z):
        N, o = self.N, self.offsets
        shapes = [(N + 1, self.nq), (N + 1, self.nv), (N, self.nv), (N, self.nj), (N, 4, 3), (N, 4), (N, 4)]
        return tuple(z[o[i]:o[i + 1]].reshape(s) for i, s in enumerate(shapes))

    def pack(self, q, v, a, tau, lam, gap, psi):
        return np.concatenate([np.asarray(t, float).ravel() for t in (q, v, a, tau, lam, gap, psi)])

    def params(self, goal, pit, sink=(0.0, 0.0, 0.0, 0.0), ref=(0.0, 0.0, 0.0)):
        """p = (q_hat, v_hat, goal, pit, sink, ref): sink (4,) lowers the terrain under each foot (the
        depth a foot rests at in a compliant plant), the same at every knot. ref = (start x, start y,
        t0): where the base reference of cost_form "reference" starts and the time of knot 0 since then."""
        return np.concatenate([self.q_hat, self.v_hat, np.asarray(goal, float), np.asarray(pit, float), np.asarray(sink, float),
                               np.asarray(ref, float)])

    def reference(self, goal, ref, t):
        """(base xy reference, base xy velocity reference, arrived flag) at absolute time t."""
        d = goal - ref[:2]
        dist = jnp.sqrt(jnp.sum(d ** 2) + 1e-12)
        arrived = jnp.where(t * self.exp["v_cmd"] >= dist, 1.0, 0.0)
        return ref[:2] + d / dist * jnp.minimum(self.exp["v_cmd"] * t, dist), d / dist * self.exp["v_cmd"] * (1.0 - arrived), arrived

    def split_p(self, p):
        nq, nv = self.nq, self.nv
        return p[:nq], p[nq:nq + nv], p[nq + nv:nq + nv + 2], p[nq + nv + 2:nq + nv + 5], p[nq + nv + 5:nq + nv + 9], p[nq + nv + 9:]

    # ---- stage functions
    def stage_f(self, q_k, v_k, q_k1, v_k1, a_k, tau_k, tau_k1, lam_k, lam_k1, gap_k, psi_k, p, w_k):
        w = self.exp["weights"]
        goal, ref = self.split_p(p)[2], self.split_p(p)[5]
        if self.exp["cost_form"] == "reference":
            p_ref, v_ref, rest = self.reference(goal, ref, ref[2] + w_k[3])
        else:
            p_ref, v_ref, rest = goal, jnp.zeros(2), w_k[1]
        vb = v_k1[:6] - jnp.concatenate([v_ref, jnp.zeros(4)])
        quat_xyz, vj, qj = q_k1[4:7], v_k1[6:], q_k1[7:]
        J = w["tau"] * jnp.sum(tau_k ** 2) + w["lam"] * jnp.sum(lam_k ** 2) + w["vj"] * jnp.sum(vj ** 2) \
            + w["qj_home"] * jnp.sum((qj - self.qj_home) ** 2) + w["quat"] * jnp.sum(quat_xyz ** 2)
        J = J + rest * (w["rest_vb"] * jnp.sum(vb ** 2) + w["rest_vj"] * jnp.sum(vj ** 2)
                          + w["rest_quat"] * jnp.sum(quat_xyz ** 2) + w["rest_qj"] * jnp.sum((qj - self.qj_home) ** 2))
        J = J + w_k[0] * jnp.sum((tau_k1 - tau_k) ** 2) + w_k[4] * jnp.sum((lam_k1 - lam_k) ** 2)
        lt2 = jnp.sum(lam_k.reshape(4, 3)[:, :2] ** 2, axis=1)
        if self.exp["slip_form"] == "psi":
            vt2 = 4.0 * psi_k ** 2 * lt2
        else:
            Jn = self.foot_jacobians(q_k1)
            vt2 = jnp.stack([jnp.sum((Jn[i] @ v_k1)[:2] ** 2) for i in range(4)])
        J = J + self.Ts * jnp.sum(w["diss"] * 2.0 * psi_k * lt2 + w["slip"] * jnp.exp(-gap_k / self.exp["slip_gap"]) * vt2)
        J = J + w_k[2] * (w["terminal_quat"] * jnp.sum(quat_xyz ** 2) + w["terminal_vb"] * jnp.sum(vb ** 2)
                          + w["goal_pos"] * jnp.sum((q_k1[:2] - p_ref) ** 2))
        return J

    def stage_c(self, q_k, v_k, q_k1, v_k1, a_k, tau_k, tau_k1, lam_k, lam_k1, gap_k, psi_k, p, w_k):
        q_hat, v_hat, _, pit, sink, _ = self.split_p(p)
        lam = lam_k.reshape(4, 3)
        Jk = self.foot_jacobians(q_k)
        gen_contact = sum((Jk[i].T @ lam[i] for i in range(4)), jnp.zeros(self.nv))
        Jn = self.foot_jacobians(q_k1)
        v_t = jnp.stack([(Jn[i] @ v_k1)[:2] for i in range(4)])
        return jnp.concatenate(
            [q_k - q_hat, v_k - v_hat,
             v_k1 - v_k - self.Ts * a_k,
             q_k1 - self.model.integrate_position(q_k, v_k1, self.Ts),
             self.model.inverse_dynamics(q_k, v_k, a_k) - jnp.concatenate([jnp.zeros(6), tau_k]) - gen_contact,
             gap_k - self.gaps(q_k1, pit) - sink,
             (v_t + 2.0 * psi_k[:, None] * lam[:, :2]).ravel()])

    def stage_d(self, q_k, v_k, q_k1, v_k1, a_k, tau_k, tau_k1, lam_k, lam_k1, gap_k, psi_k, p, w_k):
        lam = lam_k.reshape(4, 3)
        Phi = self.cone(lam)
        margin = (self.exp["cone_margin"] * self.exp["mu"] * lam[:, 2]) ** 2 - lam[:, 0] ** 2 - lam[:, 1] ** 2
        # scale_force_rows: the cone and margin rows per unit of (mu * standing load)^2, so that their residual
        # is a relative force error and not N^2 (the same feasible set; psi Phi keeps its units and its kappa_t)
        c = 1.0 / (self.exp["mu"] * jnp.asarray(self.load)) ** 2 if self.exp["scale_force_rows"] else 1.0
        return jnp.concatenate([self.knees(q_k1) - self.exp["knee_min"], c * Phi, gap_k * lam[:, 2], psi_k * Phi, c * margin])

    def staged_nlp(self, relax_equality):
        from cusadi2.jax.ipm import StagedNLP
        nlp = StagedNLP(self.stage_f, self.stage_c, self.stage_d, self.N, self.specs, c_active=self.c_active,
                        d_active=self.d_active, relax_equality=relax_equality)
        assert (nlp.nx, nlp.np) == (self.nx, self.np), "the staged layout must be this class's z and p"
        return nlp

    # ---- homotopy rows, bounds, initial guess
    def homotopy_rows(self, nlp):
        """(rows, weights) of the complementarity products in nlp's d numbering: per stage and foot
        gap lam_n (weight m_i g l_ref) and psi Phi (weight m_i g v_ref); the row's bounds are
        kappa_hat * weight."""
        off = (int(self.c_active.shape[1]) if nlp.relax_equality else 0) + 8
        rows = np.concatenate([nlp.d_rows[:, off:off + 4].ravel(), nlp.d_rows[:, off + 4:off + 8].ravel()])
        weights = np.concatenate([np.tile(self.load * self.exp["l_ref"], self.N), np.tile(self.load * self.exp["v_ref"], self.N)])
        return rows, weights

    def d_bounds(self, nlp, kappa_hat, kind):
        """(d_L, d_U) at a fixed kappa_hat: `central` [kappa, kappa] or `relaxed` [0, kappa] on the
        homotopy rows; 0 on equality rows of a relaxed export; [0, inf) on the inequalities."""
        d_L = np.zeros(nlp.nyd)
        d_U = np.where(nlp.relaxed, 0.0, np.inf) if nlp.relax_equality else np.full(nlp.nyd, np.inf)
        rows, weights = self.homotopy_rows(nlp)
        d_U[rows] = kappa_hat * weights
        d_L[rows] = kappa_hat * weights if kind == "central" else 0.0
        return d_L, d_U

    def bounds(self):
        big = np.inf
        N, nq, nv = self.N, self.nq, self.nv
        f_max = self.exp["f_max"]
        q_L, q_U = np.full(nq, -big), np.full(nq, big)
        q_L[2] = self.exp["base_z_min"]
        q_L[3:7], q_U[3:7] = -1.0, 1.0
        q_L[7:], q_U[7:] = self.m.jnt_range[1:, 0], self.m.jnt_range[1:, 1]
        v_L, v_U = np.full(nv, -big), np.full(nv, big)
        v_L[6:], v_U[6:] = -np.asarray(self.exp["v_max"] * 4), np.asarray(self.exp["v_max"] * 4)
        tau_max = np.asarray(self.m.actuator_ctrlrange[:, 1])
        lam_L = np.tile([-f_max, -f_max, 0.0], (N, 4, 1))
        lam_U = np.tile([f_max, f_max, f_max], (N, 4, 1))
        x_L = self.pack(np.tile(q_L, (N + 1, 1)), np.tile(v_L, (N + 1, 1)), np.full((N, nv), -big), np.tile(-tau_max, (N, 1)),
                        lam_L, np.zeros((N, 4)), np.zeros((N, 4)))
        x_U = self.pack(np.tile(q_U, (N + 1, 1)), np.tile(v_U, (N + 1, 1)), np.full((N, nv), big), np.tile(tau_max, (N, 1)),
                        lam_U, np.full((N, 4), big), np.full((N, 4), big))
        return x_L, x_U

    def initial_guess(self, kappa_hat, dq=None):
        """Standing still on the central path of kappa_hat: every foot carries its standing load at
        the gap kappa_n / load = kappa_hat l_ref (knots >= 1 are lifted by that much; knot 0 is the
        measured state), no tangential force, psi = kappa_t / Phi. `dq` (12,) offsets the joints of
        the knots >= 1."""
        N, mu = self.N, self.exp["mu"]
        hover = kappa_hat * self.exp["l_ref"]
        q = np.tile(self.q_hat, (N + 1, 1))
        q[1:, 2] += hover
        if dq is not None:
            q[1:, 7:] += np.asarray(dq, float)
        lam = np.zeros((N, 4, 3))
        lam[:, :, 2] = self.load
        psi = kappa_hat * self.load * self.exp["v_ref"] / (mu * self.load) ** 2
        return self.pack(q, np.zeros((N + 1, self.nv)), np.zeros((N, self.nv)), np.tile(self.tau_stand, (N, 1)), lam,
                         np.full((N, 4), hover), np.tile(psi, (N, 1)))

    # ---- the residuals stage by stage, straight from JAX (the check of the staged composition)
    def direct(self, z, p):
        q, v, a, tau, lam, gap, psi = self.unpack(jnp.asarray(z))
        p = jnp.asarray(p)
        f, cs, ds = 0.0, [], []
        for k in range(self.N):
            tau1 = tau[k + 1] if k + 1 < self.N else jnp.zeros(self.nj)
            lam1 = lam[k + 1].ravel() if k + 1 < self.N else jnp.zeros(12)
            args = (q[k], v[k], q[k + 1], v[k + 1], a[k], tau[k], tau1, lam[k].ravel(), lam1, gap[k], psi[k], p, jnp.asarray(self.w_k[k]))
            f = f + self.stage_f(*args)
            cs.append(np.asarray(self.stage_c(*args))[self.c_active[k]])
            ds.append(np.asarray(self.stage_d(*args))[self.d_active[k]])
        return float(f), np.concatenate(cs), np.concatenate(ds)


# ----------------------------------------------------------------------------- checks and reference solve
def model_checks(prob, nlp, z, p, seed):
    """Staged whole-horizon f, c, d against the direct JAX evaluation, and their Jacobians against
    central differences along random directions, at the point z."""
    import casadi as ca
    X, P, F, Cx, Dx = nlp.whole_horizon()
    g = ca.vertcat(F, Cx, Dx)
    val = ca.Function("val_g", [X, P], [g])
    jac = ca.Function("jac_g", [X, P], [ca.jacobian(g, X)])
    f, c, d = prob.direct(z, p)
    ref = np.concatenate([[f], c, d])
    got = np.asarray(val(z, p)).ravel()
    rng = np.random.default_rng(seed)
    Jz = jac(z, p)
    fd_err, eps = 0.0, 1e-6
    for _ in range(5):
        u = rng.standard_normal(prob.nx)
        num = (np.asarray(val(z + eps * u, p)).ravel() - np.asarray(val(z - eps * u, p)).ravel()) / (2 * eps)
        ana = np.asarray(ca.mtimes(Jz, ca.DM(u))).ravel()
        fd_err = max(fd_err, float(np.max(np.abs(num - ana) / (1.0 + np.abs(ana)))))
    return {"value_max_abs_diff": float(np.max(np.abs(ref - got))), "jacobian_fd_max_rel_err": fd_err,
            "nnz_jacobian": int(Jz.nnz()), "rows": int(g.numel())}


def run_experiment(name, exp):
    import casadi
    import mujoco
    from cusadi2.jax.ipm import solve_ipopt
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    start = datetime.now().isoformat(timespec="seconds")
    prob = Go2ContactImplicit(exp["model"])
    t = time.perf_counter()
    nlp = prob.staged_nlp(relax_equality=False)
    t_trace = time.perf_counter() - t
    p0 = prob.params(exp["goal"], exp["pit"])
    z0 = prob.initial_guess(exp["kappa_hat"])
    x_L, x_U = prob.bounds()
    d_L, d_U = prob.d_bounds(nlp, exp["kappa_hat"], exp["bounds"])
    print(f"[{name}] nx={prob.nx} np={prob.np} nyc={nlp.nyc} nyd={nlp.nyd} traced in {t_trace:.1f}s; standing load "
          f"{np.round(prob.load, 2).tolist()} N, feet x {np.round(prob.feet0[:, 0], 3).tolist()}", flush=True)

    rng = np.random.default_rng(exp["check_seed"])
    z_chk = z0 + 0.05 * rng.standard_normal(prob.nx)
    checks = model_checks(prob, nlp, z_chk, p0, exp["check_seed"])
    f0, c0, d0 = prob.direct(z0, p0)
    checks["guess_max_abs_c"] = float(np.max(np.abs(c0)))
    checks["guess_homotopy_rel_err"] = float(np.max(np.abs(d0[prob.homotopy_rows(nlp)[0]] / prob.homotopy_rows(nlp)[1]
                                                           - exp["kappa_hat"]))) / exp["kappa_hat"]
    print(f"[{name}] checks: {checks}", flush=True)
    with open(os.path.join(out, "checks.json"), "w") as fh:
        json.dump(checks, fh, indent=1)

    res = solve_ipopt(None, None, None, z0, p0, x_L=x_L, x_U=x_U, d_L=d_L, d_U=d_U, options=exp["ipopt"],
                      nlp=nlp.whole_horizon())
    q, v, a, tau, lam, gap, psi = prob.unpack(res["x"])
    ipopt = {"status": res["status"], "success": res["success"], "iterations": res["iterations"], "wall_s": res["wall_s"],
             "f": res["f"], "max_abs_c": float(np.max(np.abs(res["g"][:nlp.nyc]))),
             "base_xy_final": q[-1, :2].tolist(), "base_z_min": float(q[:, 2].min()), "base_z_max": float(q[:, 2].max()),
             "gap_min": float(gap.min()), "gap_max": float(gap.max()), "lam_n_max": float(lam[:, :, 2].max()),
             "max_abs_tau": float(np.max(np.abs(tau)))}
    print(f"[{name}] IPOPT: {ipopt}", flush=True)
    np.savez(os.path.join(out, "solution.npz"), x=res["x"], q=q, v=v, a=a, tau=tau, lam=lam, gap=gap, psi=psi, p=p0, z0=z0)
    with open(os.path.join(out, "ipopt.json"), "w") as fh:
        json.dump(ipopt, fh, indent=1)
    config = {"experiment": name, "settings": exp, "start": start, "git": git_info(),
              "versions": {"jax": jax.__version__, "casadi": casadi.__version__, "mujoco": mujoco.__version__,
                           "python": sys.version.split()[0]}, "host": os.uname().nodename,
              "dims": {"nx": prob.nx, "np": prob.np, "nyc": nlp.nyc, "nyd": nlp.nyd},
              "standing_load_N": prob.load.tolist(), "feet0": prob.feet0.tolist(), "trace_s": t_trace}
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)


def main():
    cfg = load_config(CONFIG)
    for name, exp in cfg.items():
        if exp["run"]:
            run_experiment(name, exp)


if __name__ == "__main__":
    main()
