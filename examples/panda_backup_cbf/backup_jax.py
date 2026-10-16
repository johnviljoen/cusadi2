"""frax / cbfpy side of the Panda backup-trajectory CBF filter (JAX).

Same equations as backup_casadi.py, written against the unmodified frax robot and
cbfpy config of profiling/panda_problem.py:

  * nominal controller, P, q and the k = 0 rows are cbfpy's own (`cbf.qp_data`);
  * the rollout under the backup policy and the predictive rows are written here in JAX
    with frax's mass_matrix / gravity_vector / centrifugal_coriolis_vector and the demo
    config's h_2; Lie derivatives by forward mode with the 8 seeds [f, g] (cbfpy's own
    construction for its rows).

Four implementation variants, so the frax column can be reported at its fastest:
    loop  = "scan" | "unrolled"      lax.scan over the horizon vs a Python loop XLA unrolls
    deriv = "jvp"  | "jacfwd"        vmapped jvp over the 8 seeds vs jacfwd(H)(z) @ [f, g]
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
PROFILING = HERE.parent.parent / "profiling"
sys.path.insert(0, str(PROFILING))
from panda_problem import PandaProblem  # noqa: E402
from backup_casadi import DT, TAU, ALPHA  # noqa: E402


class BackupProblem:
    """PandaProblem("qpax") plus the predictive rows."""

    def __init__(self, N, dt=DT, tau=TAU, loop="scan", deriv="jvp"):
        assert loop in ("scan", "unrolled") and deriv in ("jvp", "jacfwd")
        self.base = PandaProblem("qpax")
        self.robot, self.cfg, self.cbf = self.base.robot, self.base.cfg, self.base.cbf
        self.N, self.dt, self.tau, self.loop, self.deriv = N, dt, tau, loop, deriv
        self.nv = self.base.nv
        self.m = self.base.m
        self.u_max = jnp.asarray(self.robot.actuated_joint_max_forces)
        # cbfpy's penalties per row: 1e3 on every CBF row (k = 0 and predictive), 1e5 on the torque bounds
        self.penalty = np.concatenate([np.full(self.base.num_cbf * (N + 1), self.base.penalty[0]), self.base.penalty[self.base.num_cbf:]])
        self.solver_tol = self.base.solver_tol
        self.n_cbf_per_step = self.base.num_cbf
        self.n_rows = 2 * self.m + self.n_cbf_per_step * (N + 1)
        self.desired_from_q = self.base.desired_from_q
        self.nominal = self.base.nominal

    # ---- backup policy and one semi-implicit Euler step ----
    def backup_torque(self, q, v):
        g = self.robot.gravity_vector(q)
        M = self.robot.mass_matrix(q)
        return jnp.clip(g - M @ v / self.tau, -self.u_max, self.u_max)

    def step(self, z):
        nv = self.nv
        q, v = z[:nv], z[nv:]
        M = self.robot.mass_matrix(q)
        g = self.robot.gravity_vector(q)
        c = self.robot.centrifugal_coriolis_vector(q, v)
        u_b = jnp.clip(g - M @ v / self.tau, -self.u_max, self.u_max)
        a = jnp.linalg.solve(M, u_b - g - c)
        v1 = v + self.dt * a
        q1 = q + self.dt * v1
        return jnp.concatenate([q1, v1])

    def predicted_barriers(self, z):
        """H(z) = [h2(phi_1(z)); ...; h2(phi_N(z))], shape (64 N,)."""
        if self.N == 0:
            return jnp.zeros(0)
        if self.loop == "scan":
            def body(zk, _):
                z1 = self.step(zk)
                return z1, self.cfg.h_2(z1)
            _, Hs = jax.lax.scan(body, z, None, length=self.N)
            return Hs.reshape(-1)
        Hs, zk = [], z
        for _ in range(self.N):
            zk = self.step(zk)
            Hs.append(self.cfg.h_2(zk))
        return jnp.concatenate(Hs)

    def predictive_rows(self, z):
        """(G_pred (64N, 7), h_pred (64N,)) : -LgH u <= alpha H + LfH."""
        if self.N == 0:
            return jnp.zeros((0, self.m)), jnp.zeros(0)
        f = self.cfg.f(z)
        g = self.cfg.g(z)                                   # (14, 7)
        seeds = jnp.concatenate([f[:, None], g], axis=1)    # (14, 8)
        if self.deriv == "jvp":
            def one(s):
                return jax.jvp(self.predicted_barriers, (z,), (s,))
            H, L = jax.vmap(one, in_axes=(1,), out_axes=(None, 1))(seeds)  # H (64N,), L (64N, 8)
        else:
            H = self.predicted_barriers(z)
            L = jax.jacfwd(self.predicted_barriers)(z) @ seeds
        LfH, LgH = L[:, 0], L[:, 1:]
        return -LgH, ALPHA * H + LfH

    # ---- full QP data: cbfpy rows + predictive rows, same row order as the CasADi function ----
    def qp_data(self, z, u_des):
        P, q, _, _, G0, h0 = self.cbf.qp_data(z, u_des)
        n0 = self.n_cbf_per_step
        Gp, hp = self.predictive_rows(z)
        G = jnp.concatenate([G0[:n0], Gp, G0[n0:]])
        h = jnp.concatenate([h0[:n0], hp, h0[n0:]])
        return P, q, G, h

    def model_part(self, z, z_des):
        u = self.nominal(z, z_des)
        P, q, G, h = self.qp_data(z, u)
        return u, P, q, G, h

    def solve(self, P, q, G, h):
        import qpax
        return qpax.solve_qp_elastic_primal(P, q, G, h, penalty=jnp.asarray(self.penalty), solver_tol=self.solver_tol)[: self.m]

    def end_to_end(self, z, z_des):
        u, P, q, G, h = self.model_part(z, z_des)
        return self.solve(P, q, G, h)

    def rollout_states(self, z):
        """phi_0..phi_N, shape (N+1, 14) (for the video traces and the stopping check)."""
        zs = [z]
        for _ in range(self.N):
            zs.append(self.step(zs[-1]))
        return jnp.stack(zs)
