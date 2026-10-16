"""The frax / cbfpy twin of model.py's step function (the frax evaluator), and its export for the driver.

Same equations, written against the unmodified frax robot and cbfpy config of profiling/panda_problem.py:
frax's mass_matrix / gravity_vector / centrifugal_coriolis_vector / ee_transform for the dynamics and the
cost, cbfpy's G_qp / h_qp for the cupboard scene's barrier rows (scene.barriers_jnp on frax's kinematics; the
demo's own joint-limit, singularity and self-collision barriers).

`export_programs(T, dtype, out_dir)` serialises (jax.export, StableHLO bytecode, batch dimension symbolic)
    safe_rollout_{loop}_T{T}_{dtype}.mlir : (x0[B,16], Y[B,28], rp[B,3], par[B,6]) -> xT[B,16]
    safe_step_{dtype}.mlir                : (x[B,16],  Y, rp, par)                  -> (x1[B,16], u[B,7])
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
from cbfpy import CBF  # noqa: E402
from oscbf_demo import DemoOSCBFConfig  # noqa: E402
from panda_problem import PandaProblem  # noqa: E402
from model import DT, N_KNOT, NV, NX, ND, NPAR, DES_ROT, PARAMS, knot_dt  # noqa: E402
import scene  # noqa: E402


@jax.tree_util.register_static
class CupboardOSCBFConfig(DemoOSCBFConfig):
    """The demo's CBF config (same alpha, alpha_2, u bounds, joint-limit, singularity and self-collision barriers) with
    the obstacle sphere replaced by the cupboard scene (scene.barriers_jnp): frax's kinematics, cbfpy's rows."""

    def __init__(self, robot):
        super().__init__(robot, np.zeros((1, 3)), np.zeros(1))     # the demo's obstacle arguments are unused

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])
        pos, rad = self.robot.link_collision_data(q)
        h_self = self.robot._self_collision_distances_from_link_data(pos, rad)
        Tee = self.robot.ee_transform(q)
        return scene.barriers_jnp(self.q_min, self.q_max, q, h_singularity, h_self, pos, rad, Tee[:3, 3], Tee[:3, :3])


class FraxStep:
    def __init__(self, T, dt=DT):
        self.base = PandaProblem("qpax")                 # the frax robot (and the dataset sampler's problem)
        self.robot = self.base.robot
        self.cfg = CupboardOSCBFConfig(self.robot); self.cfg.backend = "qpax"
        self.cbf = CBF.from_config(self.cfg)
        assert self.cfg.num_cbf == scene.N_ROWS, (self.cfg.num_cbf, scene.N_ROWS)
        self.T, self.dt = T, dt
        self.kdt = knot_dt(T)
        self.u_max = jnp.asarray(self.robot.actuated_joint_max_forces)
        self.des_rot = jnp.asarray(DES_ROT)

    def weights(self, tau):
        s = jnp.clip(tau / self.kdt, 0.0, N_KNOT - 1)
        return [jnp.maximum(0.0, 1.0 - jnp.abs(s - j)) for j in range(N_KNOT)]

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
        axis = jnp.asarray(scene.AXIS_YZ); wrist = pos - scene.WRIST_BACK * rot[:, 2]       # cost shaping, as in model.py
        s_tcp = 1.0 / (1.0 + jnp.exp(-(pos[0] - scene.FUNNEL_X_ON) / scene.FUNNEL_W)); s_wr = 1.0 / (1.0 + jnp.exp(-(wrist[0] - scene.FUNNEL_X_ON) / scene.FUNNEL_W))
        ell = ell + P["w_fun"] * s_tcp * jnp.sum((pos[1:3] - axis) ** 2) + P["w_wrist"] * s_wr * jnp.sum((wrist[1:3] - axis) ** 2)
        ell = ell + P["w_q"] * jnp.sum((q - jnp.asarray(scene.Q_PREGRASP)) ** 2)
        return jnp.concatenate([q1, v1, jnp.array([tau + self.dt, c + self.dt * ell])])


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
