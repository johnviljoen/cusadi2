"""The MuJoCo XLA (MJX) + cbfpy twin of model.py's two model functions (the "MuJoCo XLA" baseline of the paper).

Same equations as frax_model.py, with the rigid-body terms and the kinematics taken from MJX instead of frax:
mjx.crb / mjx.full_m for the mass matrix, mjx.rne for gravity + Coriolis, mjx.kinematics for the body poses (the
end-effector pose, the collision-sphere centres) and mjx.jac for the end-effector Jacobian of the singularity barrier.
cbfpy builds the 64 one-step CBF rows (G_qp / h_qp) from a config whose f, g and h_2 call MJX, exactly as the frax
config does with frax, so the row algebra and the barrier definitions are cbfpy's and the demo's.

MuJoCo's URDF importer drops the rotational inertia of frax's massless `panda_link8` (0.1 kg m^2 on the diagonal),
so the MJCF is generated from the Pinocchio model of the same URDF (composite joint inertias in principal form, joint
placements and limits): the MJX model is then the same rigid-body model as frax's and CasADi's to machine precision.

`export_programs(T, dtype, out_dir, batches)` writes mjx_model_T{T}_B{B}_{dtype}.mlir and mjx_integrate_B{B}_{dtype}.mlir
for the driver's PJRT evaluator.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common  # noqa: E402

if "jax" not in sys.modules:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    common.configure("gpu", x64=os.environ.get("SAFE_MPPI_X64", "1") == "1")
common.ensure_frax()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from cbfpy import CBF  # noqa: E402
from mujoco import mjx  # noqa: E402
from oscbf_demo import DemoOSCBFConfig  # noqa: E402
from panda_problem import PandaProblem  # noqa: E402
from model import DT, N_KNOT, NV, NX, ND, NPAR, DES_ROT, PARAMS, knot_dt  # noqa: E402

URDF = common.FRAX_DIR / "frax" / "assets" / "franka_panda" / "panda.urdf"


def mjcf_from_pinocchio(urdf=URDF):
    """MJCF of the Pinocchio model of `urdf`: one body per joint at the joint placement (so MuJoCo's body frames are
    Pinocchio's joint frames and frax's joint_to_world_transforms), the composite inertia of every joint's subtree of
    fixed links as the body's inertial element, hinge joints about z with the URDF limits. No geoms: collisions are
    the demo's analytic spheres, evaluated in h_2."""
    pm = pin.buildModelFromUrdf(str(urdf))
    r = lambda v: repr(float(v))  # noqa: E731
    L = ['<mujoco model="panda_from_pinocchio">', '  <compiler angle="radian" inertiafromgeom="false" balanceinertia="false"/>',
         '  <option gravity="0 0 -9.81"/>', '  <worldbody>']
    for j in range(1, pm.njoints):
        assert pm.joints[j].shortname() == "JointModelRZ", pm.joints[j].shortname()
        pl, I = pm.jointPlacements[j], pm.inertias[j]
        qx, qy, qz, qw = pin.Quaternion(pl.rotation).coeffs()
        # principal inertia and frame from numpy's eigendecomposition: MuJoCo's own `fullinertia` path goes through its
        # iterative 3x3 eigensolver, which leaves 4e-9 in the mass matrix against frax / Pinocchio
        lam, Rp = np.linalg.eigh(I.inertia)              # inertia about the centre of mass, in the body frame
        if np.linalg.det(Rp) < 0:
            Rp[:, 2] *= -1
        px, py, pz, pw = pin.Quaternion(np.ascontiguousarray(Rp)).coeffs()
        ind = "  " * (j + 1)
        L.append(f'{ind}<body name="link{j}" pos="{r(pl.translation[0])} {r(pl.translation[1])} {r(pl.translation[2])}" quat="{r(qw)} {r(qx)} {r(qy)} {r(qz)}">')
        L.append(f'{ind}  <inertial pos="{r(I.lever[0])} {r(I.lever[1])} {r(I.lever[2])}" quat="{r(pw)} {r(px)} {r(py)} {r(pz)}" mass="{r(I.mass)}" '
                 f'diaginertia="{r(lam[0])} {r(lam[1])} {r(lam[2])}"/>')
        L.append(f'{ind}  <joint name="joint{j}" type="hinge" axis="0 0 1" limited="true" range="{r(pm.lowerPositionLimit[j - 1])} {r(pm.upperPositionLimit[j - 1])}"/>')
    for j in range(pm.njoints - 1, 0, -1):
        L.append("  " * (j + 1) + "</body>")
    L += ["  </worldbody>", "</mujoco>"]
    return "\n".join(L)


@jax.tree_util.register_static
class MjxOSCBFConfig(DemoOSCBFConfig):
    """The demo's CBF config (same barriers, same alpha, same u bounds) with the model calls routed to MJX."""

    def __init__(self, robot, collision_positions, collision_radii, mj):
        self.mj = mj
        super().__init__(robot, collision_positions, collision_radii)

    def f(self, z, *args, **kwargs):
        q, v = z[: self.num_joints], z[self.num_joints:]
        M, bias, _ = self.mj.dyn(q, v)
        return jnp.concatenate([v, -jnp.linalg.inv(M) @ bias])

    def g(self, z, *args, **kwargs):
        q, v = z[: self.num_joints], z[self.num_joints:]
        M, _, _ = self.mj.dyn(q, v)
        return jnp.block([[jnp.zeros((self.num_joints, self.m))], [jnp.linalg.inv(M)]])

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        d = self.mj.fwd(q, jnp.zeros_like(q))
        h_joint_limits = jnp.concatenate([jnp.asarray(self.q_max) - q, q - jnp.asarray(self.q_min)])
        sigmas = jax.lax.linalg.svd(self.mj.ee_jacobian(d), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])
        pos, rad = self.mj.sphere_positions(d), self.robot.flat_collision_radii
        center_deltas = (pos[:, None, :] - self.collision_positions[None, :, :]).reshape(-1, 3)
        radii_sums = (rad[:, None] + self.collision_radii[None, :]).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums
        h_floor = pos[:, 2] - rad - 0.01
        h_self_collision = self.robot._self_collision_distances_from_link_data(pos, rad)
        return jnp.concatenate([h_joint_limits, h_singularity, h_collision, h_self_collision, h_floor])


class MjxStep:
    def __init__(self, T, dt=DT):
        self.base = PandaProblem("qpax")
        self.robot = self.base.robot
        self.T, self.dt, self.kdt = T, dt, knot_dt(T)
        self.xml = mjcf_from_pinocchio()
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.mx = mjx.put_model(self.m, impl="jax")
        self.dx = mjx.make_data(self.m, impl="jax")
        self.ee_body = self.m.nbody - 1
        self.ee_R, self.ee_t = jnp.asarray(self.robot.ee_offset[:3, :3]), jnp.asarray(self.robot.ee_offset[:3, 3])
        self.pad = jnp.asarray(self.robot.padded_collision_positions)         # (nv, max_spheres, 3) in joint frames
        self.slice = np.asarray(self.robot.collision_slice_indices)
        self.u_max = jnp.asarray(self.robot.actuated_joint_max_forces)
        self.des_rot = jnp.asarray(DES_ROT)
        cfg = MjxOSCBFConfig(self.robot, np.array([[0.5, 0.5, 0.5]]), np.array([0.3]), self)
        cfg.backend = "qpax"
        self.cfg, self.cbf = cfg, CBF.from_config(cfg)

    # ---- MJX pipeline: the subset of mjx.forward that the model needs (kinematics, CRB mass matrix, RNE bias) ----
    def fwd(self, q, v):
        d = self.dx.replace(qpos=q, qvel=v)
        d = mjx.kinematics(self.mx, d); d = mjx.com_pos(self.mx, d); d = mjx.crb(self.mx, d)
        d = mjx.com_vel(self.mx, d); d = mjx.rne(self.mx, d)
        return d

    def dyn(self, q, v):
        d = self.fwd(q, v)
        return mjx.full_m(self.mx, d), d.qfrc_bias, d

    def ee(self, d):
        R, p = d.xmat[self.ee_body].reshape(3, 3), d.xpos[self.ee_body]
        return p + R @ self.ee_t, R @ self.ee_R

    def ee_jacobian(self, d):
        pos, _ = self.ee(d)
        jp, jr = mjx.jac(self.mx, d, pos, self.ee_body)
        return jnp.concatenate([jp.T, jr.T])

    def sphere_positions(self, d):
        R, p = d.xmat[1:].reshape(-1, 3, 3), d.xpos[1:]
        pts = p[:, None, :] + jnp.einsum("nij,nsj->nsi", R, self.pad)
        return pts.reshape(-1, 3)[self.slice]

    # ---- the two model functions ----
    def weights(self, tau):
        s = jnp.clip(tau / self.kdt, 0.0, N_KNOT - 1)
        return [jnp.maximum(0.0, 1.0 - jnp.abs(s - j)) for j in range(N_KNOT)]

    def model(self, x, Y, rp, par):
        nv = NV
        q, v, tau = x[:nv], x[nv:2 * nv], x[2 * nv]
        z = x[:2 * nv]
        P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
        M, bias, _ = self.dyn(q, v)
        w = self.weights(tau)
        q_des = sum(w[j] * Y[j * nv:(j + 1) * nv] for j in range(N_KNOT))
        u_nom = jnp.clip(M @ (P["kp"] * (q_des - q) - P["kd"] * v) + bias, -self.u_max, self.u_max)
        n = self.cfg.num_cbf
        return u_nom, self.cbf.G_qp(z, u_nom)[:n], self.cbf.h_qp(z, u_nom)[:n]

    def integrate(self, x, u, rp, par):
        nv = NV
        q, v, tau, c = x[:nv], x[nv:2 * nv], x[2 * nv], x[2 * nv + 1]
        P = {k: par[i] for i, (k, _) in enumerate(PARAMS)}
        M, bias, d = self.dyn(q, v)
        a = jnp.linalg.solve(M, u - bias)
        v1 = v + self.dt * a; q1 = q + self.dt * v1
        pos, rot = self.ee(d)
        e_rot = -0.5 * (jnp.cross(rot[:, 0], self.des_rot[:, 0]) + jnp.cross(rot[:, 1], self.des_rot[:, 1]) + jnp.cross(rot[:, 2], self.des_rot[:, 2]))
        ell = P["w_pos"] * jnp.sum((pos - rp) ** 2) + P["w_rot"] * jnp.sum(e_rot ** 2) + P["w_v"] * jnp.sum(v ** 2) + P["w_u"] * jnp.sum(u ** 2)
        return jnp.concatenate([q1, v1, jnp.array([tau + self.dt, c + self.dt * ell])])


def export_programs(T, dtype, out_dir, batches):
    from jax import export
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jd = jnp.float32 if dtype == "float" else jnp.float64
    assert dtype == "float" or jax.config.jax_enable_x64, "double export needs x64"
    ms = MjxStep(T)
    paths = {}
    for B in batches:
        spec = lambda n: jax.ShapeDtypeStruct((B, n), jd)  # noqa: E731
        p = out_dir / f"mjx_model_T{T}_B{B}_{dtype}.mlir"
        exp = export.export(jax.jit(jax.vmap(ms.model), keep_unused=True), platforms=["cuda"])(spec(NX), spec(ND), spec(3), spec(NPAR))
        p.write_bytes(exp.mlir_module_serialized); paths[f"model_{B}"] = p
        p = out_dir / f"mjx_integrate_B{B}_{dtype}.mlir"
        exp = export.export(jax.jit(jax.vmap(ms.integrate), keep_unused=True), platforms=["cuda"])(spec(NX), spec(NV), spec(3), spec(NPAR))
        p.write_bytes(exp.mlir_module_serialized); paths[f"integrate_{B}"] = p
    return paths


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=100); ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--export", action="store_true"); ap.add_argument("--dtype", default="double")
    ap.add_argument("--out", default=str(HERE / "build" / "programs")); ap.add_argument("--batches", type=int, nargs="+", default=[256])
    a = ap.parse_args()
    from frax_model import FraxStep
    from dataset import make_dataset
    t0 = time.perf_counter(); ms = MjxStep(a.T); fs = FraxStep(a.T)
    ds = make_dataset(fs.base, a.n, 0, a.T)
    X, Y, RP, PAR = [jnp.asarray(ds[k]) for k in ("x", "Y", "rp", "par")]
    um, Gm, hm = [np.asarray(v) for v in jax.jit(jax.vmap(ms.model))(X, Y, RP, PAR)]
    uf, Gf, hf = [np.asarray(v) for v in jax.jit(jax.vmap(fs.model))(X, Y, RP, PAR)]
    rel = lambda p, q: float(np.abs(p - q).max() / max(1.0, np.abs(q).max()))  # noqa: E731
    print(f"model  mjx vs frax: u_nom {rel(um, uf):.1e}  G {rel(Gm, Gf):.1e}  h {rel(hm, hf):.1e}")
    U = jnp.asarray(uf)
    x1m = np.asarray(jax.jit(jax.vmap(ms.integrate))(X, U, RP, PAR)); x1f = np.asarray(jax.jit(jax.vmap(fs.integrate))(X, U, RP, PAR))
    print(f"integrate mjx vs frax: x1 {rel(x1m, x1f):.1e}   ({time.perf_counter() - t0:.1f} s)")
    if a.export:
        print(export_programs(a.T, a.dtype, a.out, a.batches))
