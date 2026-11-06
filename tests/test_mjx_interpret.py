"""MuJoCo MJX smooth dynamics (forward / inverse) interpret into CasADi SX: values match
MJX, Jacobians match jax.jacfwd, and the structural pattern covers the numerical one.
CPU only (no cusadi2 plans); skipped without mujoco-mjx."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
mujoco = pytest.importorskip("mujoco")
mjx = pytest.importorskip("mujoco.mjx")
import casadi as ca  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cusadi2.jax import trace_to_sx  # noqa: E402

jax.config.update("jax_enable_x64", True)

MODEL = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="capsule" size="0.05 0.1" mass="1"/>
      <body name="l1" pos="0 0 -0.2">
        <joint type="hinge" axis="0 1 0" armature="0.01" damping="0.1"/>
        <geom type="capsule" size="0.03 0.15" mass="0.5"/>
        <body name="l2" pos="0 0 -0.3">
          <joint type="hinge" axis="1 0 0" armature="0.01"/>
          <geom type="capsule" size="0.03 0.15" mass="0.3"/>
        </body>
      </body>
      <body name="arm" pos="0.1 0 0">
        <joint type="slide" axis="1 0 0"/>
        <geom type="box" size="0.05 0.05 0.05" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model():
    m = mujoco.MjModel.from_xml_string(MODEL)
    m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_CONTACT
                           | mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    mx = mjx.put_model(m, impl="jax")
    return m, mx, mjx.make_data(mx)


@pytest.mark.parametrize("kind", ["inverse", "forward"])
def test_mjx_smooth_dynamics_to_sx(model, kind):
    m, mx, dx = model
    if kind == "inverse":
        f = lambda q, qd, u: mjx.inverse(mx, dx.replace(qpos=q, qvel=qd, qacc=u)).qfrc_inverse
    else:
        f = lambda q, qd, u: mjx.forward(mx, dx.replace(qpos=q, qvel=qd, qfrc_applied=u)).qacc
    rng = np.random.default_rng(0)
    q = np.asarray(m.qpos0) + 0.3 * rng.standard_normal(m.nq)
    q[3:7] /= np.linalg.norm(q[3:7])
    qd, u = rng.standard_normal(m.nv), rng.standard_normal(m.nv)
    syms, (y,), (shape,), _ = trace_to_sx(f, [(m.nq,), (m.nv,), (m.nv,)], [np.float64] * 3, ["q", "qd", "u"])
    assert shape == (m.nv,)
    x = ca.vertcat(*syms)
    xv = np.concatenate([q, qd, u])
    args = [jnp.asarray(a) for a in (q, qd, u)]
    np.testing.assert_allclose(np.asarray(ca.Function("f", [x], [y])(xv)).ravel(), f(*args), rtol=1e-12, atol=1e-12)
    J = ca.jacobian(y, x)
    J_num = np.asarray(ca.Function("J", [x], [ca.densify(J)])(xv))
    J_ref = np.hstack([np.asarray(b) for b in jax.jacfwd(f, argnums=(0, 1, 2))(*args)])
    np.testing.assert_allclose(J_num, J_ref, rtol=1e-10, atol=1e-10)
    structural = np.zeros(J_num.shape, bool)
    structural[J.sparsity().get_triplet()[0], J.sparsity().get_triplet()[1]] = True
    assert np.all(structural[np.abs(J_ref) > 1e-12]), "structural pattern misses a numerical nonzero"
    if kind == "inverse":
        assert J.nnz() < J.numel()          # the inverse dynamics keep tree sparsity through MJX
