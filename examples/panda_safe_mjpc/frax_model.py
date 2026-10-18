"""The frax / cbfpy twin of model.py's step function (the frax evaluator), and its export for the driver.

Same equations, written against the unmodified frax robot and cbfpy config of profiling/panda_problem.py:
frax's mass_matrix / gravity_vector / centrifugal_coriolis_vector / ee_transform for the dynamics and the
cost, cbfpy's G_qp / h_qp for the 64 barrier rows (the same rows cbfpy's own filter solves), and the
exact small-active-set projection written in JAX. `rollout(x0, Y, rp, par)` chains T steps (lax.scan or unrolled).

`export_programs(T, dtype, out_dir, batches)` serialises (jax.export, StableHLO bytecode, concrete batch)
    safe_model_T{T}_B{B}_{dtype}.mlir : (x[B,16], Y[B,49], rp[B,3], par[B,6]) -> (u_nom[B,7], G[B,64,7], h[B,64])
    safe_integrate_B{B}_{dtype}.mlir  : (x[B,16], u[B,7], rp[B,3], par[B,6])  -> x1[B,16]
for the driver to compile and run through the PJRT C API.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common  # noqa: E402

if "jax" not in sys.modules:                                  # standalone use; scripts call configure() themselves
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    common.configure("gpu", x64=os.environ.get("SAFE_MPPI_X64", "1") == "1")
common.ensure_frax()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from panda_problem import PandaProblem  # noqa: E402
from model import DT, N_KNOT, NV, NX, ND, NPAR, PROJ_ROUNDS, DES_ROT, PARAMS, knot_dt, interp_weights  # noqa: E402


class FraxStep:
    def __init__(self, T, dt=DT, rounds=PROJ_ROUNDS):
        self.base = PandaProblem("qpax")
        self.robot, self.cfg, self.cbf = self.base.robot, self.base.cfg, self.base.cbf
        self.T, self.dt, self.rounds = T, dt, rounds
        self.kdt = knot_dt(T)
        self.u_max = jnp.asarray(self.robot.actuated_joint_max_forces)
        self.des_rot = jnp.asarray(DES_ROT)

    def weights(self, tau):
        return interp_weights(tau, self.kdt, jnp)            # MJPC's cubic spline, the same weights as model.py

    def project(self, u, A, b):
        """Exact small-active-set projection, identical to model.project (same candidate order and tolerances)."""
        from model import BIG, PROJ_TOL, TIE_EPS, DET_REL, TRUST
        inv_norm = 1.0 / jnp.sqrt(jnp.sum(A * A, axis=1) + 1e-12)
        for _ in range(self.rounds):
            r = (A @ u - b) * inv_norm
            rt = r - TIE_EPS * jnp.arange(r.shape[0], dtype=r.dtype) * (1.0 + jnp.abs(r))
            sels, ms, rr = [], [], rt
            for _k in range(3):
                m = jnp.max(rr); sel = (rr >= m).astype(u.dtype)
                sels.append(sel); ms.append(m); rr = rr - BIG * sel
            a = jnp.stack([sels[k] @ A for k in range(3)])          # (3, 7)
            bb = jnp.stack([sels[k] @ b for k in range(3)])
            g = a @ a.T; c = a @ u - bb
            def cand(S):
                S = list(S)
                if len(S) == 1:
                    k = S[0]; det = g[k, k]; scale = g[k, k]; lam = jnp.array([c[k] / det])
                elif len(S) == 2:
                    k, l = S; det = g[k, k] * g[l, l] - g[k, l] * g[k, l]; scale = g[k, k] * g[l, l]
                    lam = jnp.array([(g[l, l] * c[k] - g[k, l] * c[l]) / det, (-g[k, l] * c[k] + g[k, k] * c[l]) / det])
                else:
                    k, l, m_ = S
                    G = jnp.array([[g[k, k], g[k, l], g[k, m_]], [g[l, k], g[l, l], g[l, m_]], [g[m_, k], g[m_, l], g[m_, m_]]])
                    det = (G[0, 0] * (G[1, 1] * G[2, 2] - G[1, 2] * G[2, 1]) - G[0, 1] * (G[1, 0] * G[2, 2] - G[1, 2] * G[2, 0])
                           + G[0, 2] * (G[1, 0] * G[2, 1] - G[1, 1] * G[2, 0]))
                    adj = jnp.array([[G[1, 1] * G[2, 2] - G[1, 2] * G[2, 1], G[0, 2] * G[2, 1] - G[0, 1] * G[2, 2], G[0, 1] * G[1, 2] - G[0, 2] * G[1, 1]],
                                     [G[1, 2] * G[2, 0] - G[1, 0] * G[2, 2], G[0, 0] * G[2, 2] - G[0, 2] * G[2, 0], G[0, 2] * G[1, 0] - G[0, 0] * G[1, 2]],
                                     [G[1, 0] * G[2, 1] - G[1, 1] * G[2, 0], G[0, 1] * G[2, 0] - G[0, 0] * G[2, 1], G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]]])
                    lam = adj @ jnp.array([c[k], c[l], c[m_]]) / det; scale = g[k, k] * g[l, l] * g[m_, m_]
                uS = u - sum(lam[p] * a[S[p]] for p in range(len(S)))
                ok = det > DET_REL * scale
                move2 = jnp.sum((uS - u) ** 2); ref2 = sum(jnp.maximum(0.0, c[k]) ** 2 / g[k, k] for k in S)
                ok = ok & (move2 <= TRUST * ref2 + 1e-12)
                ok = ok & jnp.all(lam >= -PROJ_TOL)
                for k in range(3):
                    if k not in S:
                        ok = ok & ((a[k] @ uS - bb[k]) / jnp.sqrt(g[k, k] + 1e-12) <= PROJ_TOL)
                return uS, ok
            u_new = u - jnp.maximum(0.0, c[0]) / g[0, 0] * a[0]
            none_viol = jnp.max(r) <= PROJ_TOL
            u_new = jnp.where(none_viol, u, u_new)
            for S in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]:
                uS, ok = cand(S)
                u_new = jnp.where(ok & ~none_viol, uS, u_new)
            u = u_new
        return u

    def step(self, x, Y, rp, par):
        nv = NV
        q, v, tau, c = x[:nv], x[nv:2 * nv], x[2 * nv], x[2 * nv + 1]
        z = x[:2 * nv]
        P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
        M = self.robot.mass_matrix(q); g = self.robot.gravity_vector(q); cor = self.robot.centrifugal_coriolis_vector(q, v)
        w = self.weights(tau)
        q_des = sum(w[j] * Y[j * nv:(j + 1) * nv] for j in range(N_KNOT))
        u_nom = M @ (P["kp"] * (q_des - q) - P["kd"] * v) + g + cor
        u_nom = jnp.clip(u_nom, -self.u_max, self.u_max)
        n = self.cfg.num_cbf
        A = self.cbf.G_qp(z, u_nom); b = self.cbf.h_qp(z, u_nom)          # [G; I; -I], [alpha h + Lf h; u_max; u_max]
        u = self.project(u_nom, A, b)
        u = jnp.clip(u, -self.u_max, self.u_max)
        a = jnp.linalg.solve(M, u - g - cor)
        v1 = v + self.dt * a; q1 = q + self.dt * v1
        Tee = self.robot.ee_transform(q); pos, rot = Tee[:3, 3], Tee[:3, :3]
        e_rot = -0.5 * (jnp.cross(rot[:, 0], self.des_rot[:, 0]) + jnp.cross(rot[:, 1], self.des_rot[:, 1]) + jnp.cross(rot[:, 2], self.des_rot[:, 2]))
        ell = P["w_pos"] * jnp.sum((pos - rp) ** 2) + P["w_rot"] * jnp.sum(e_rot ** 2) + P["w_v"] * jnp.sum(v ** 2) + P["w_u"] * jnp.sum(u ** 2)
        x1 = jnp.concatenate([q1, v1, jnp.array([tau + self.dt, c + self.dt * ell])])
        return x1, u

    def model(self, x, Y, rp, par):
        """(u_nom, G[64,7], h[64]): the nominal torque and cbfpy's rows; the projection is the MPPI's shared kernel."""
        nv = NV
        q, v, tau = x[:nv], x[nv:2 * nv], x[2 * nv]
        z = x[:2 * nv]
        P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
        M = self.robot.mass_matrix(q); g = self.robot.gravity_vector(q); cor = self.robot.centrifugal_coriolis_vector(q, v)
        w = self.weights(tau)
        q_des = sum(w[j] * Y[j * nv:(j + 1) * nv] for j in range(N_KNOT))
        u_nom = jnp.clip(M @ (P["kp"] * (q_des - q) - P["kd"] * v) + g + cor, -self.u_max, self.u_max)
        n = self.cfg.num_cbf
        return u_nom, self.cbf.G_qp(z, u_nom)[:n], self.cbf.h_qp(z, u_nom)[:n]

    def integrate(self, x, u, rp, par):
        nv = NV
        q, v, tau, c = x[:nv], x[nv:2 * nv], x[2 * nv], x[2 * nv + 1]
        P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
        M = self.robot.mass_matrix(q); g = self.robot.gravity_vector(q); cor = self.robot.centrifugal_coriolis_vector(q, v)
        a = jnp.linalg.solve(M, u - g - cor)
        v1 = v + self.dt * a; q1 = q + self.dt * v1
        Tee = self.robot.ee_transform(q); pos, rot = Tee[:3, 3], Tee[:3, :3]
        e_rot = -0.5 * (jnp.cross(rot[:, 0], self.des_rot[:, 0]) + jnp.cross(rot[:, 1], self.des_rot[:, 1]) + jnp.cross(rot[:, 2], self.des_rot[:, 2]))
        ell = P["w_pos"] * jnp.sum((pos - rp) ** 2) + P["w_rot"] * jnp.sum(e_rot ** 2) + P["w_v"] * jnp.sum(v ** 2) + P["w_u"] * jnp.sum(u ** 2)
        return jnp.concatenate([q1, v1, jnp.array([tau + self.dt, c + self.dt * ell])])

    def rollout(self, x0, Y, rp, par, loop="scan"):
        if loop == "scan":
            def body(x, _):
                return self.step(x, Y, rp, par)[0], None
            return jax.lax.scan(body, x0, None, length=self.T)[0]
        x = x0
        for _ in range(self.T):
            x = self.step(x, Y, rp, par)[0]
        return x


def export_programs(T, dtype, out_dir, batches):
    """Write safe_model_T{T}_B{B}_{dtype}.mlir and safe_integrate_B{B}_{dtype}.mlir for the driver (jax.export,
    StableHLO bytecode, concrete batch). dtype 'float' | 'double'."""
    from jax import export
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jd = jnp.float32 if dtype == "float" else jnp.float64
    assert dtype == "float" or jax.config.jax_enable_x64, "double export needs x64"
    fs = FraxStep(T)
    paths = {}
    for B in batches:
        spec = lambda n: jax.ShapeDtypeStruct((B, n), jd)  # noqa: E731
        p = out_dir / f"safe_model_T{T}_B{B}_{dtype}.mlir"
        exp = export.export(jax.jit(jax.vmap(fs.model), keep_unused=True), platforms=["cuda"])(spec(NX), spec(ND), spec(3), spec(NPAR))
        p.write_bytes(exp.mlir_module_serialized); paths[f"model_{B}"] = p
        p = out_dir / f"safe_integrate_B{B}_{dtype}.mlir"
        exp = export.export(jax.jit(jax.vmap(fs.integrate), keep_unused=True), platforms=["cuda"])(spec(NX), spec(NV), spec(3), spec(NPAR))
        p.write_bytes(exp.mlir_module_serialized); paths[f"integrate_{B}"] = p
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=20); ap.add_argument("--dtype", default="double")
    ap.add_argument("--out", default=str(HERE / "build" / "programs")); ap.add_argument("--batches", type=int, nargs="+", default=[1, 256])
    a = ap.parse_args()
    print(export_programs(a.T, a.dtype, a.out, a.batches))
