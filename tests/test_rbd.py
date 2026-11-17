"""cusadi2.jax.rbd against MuJoCo: inverse / forward dynamics, kinematics and position
integration on a model with every supported feature, and on the Unitree G1 when the
Menagerie file is present. Plus: the inverse dynamics interpret into SX with a
structural pattern that covers the numerical one."""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
mujoco = pytest.importorskip("mujoco")
import casadi as ca  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cusadi2.jax import rbd  # noqa: E402
from cusadi2.jax import trace_to_sx  # noqa: E402

jax.config.update("jax_enable_x64", True)

MODEL = """
<mujoco>
  <option gravity="0.1 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 1" quat="0.9 0.1 0.2 0" gravcomp="0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="2"/>
      <body name="composite" pos="0.2 0 0" quat="0.8 0 0.6 0">
        <joint type="slide" axis="0 1 0" ref="0.1" stiffness="3" springref="0.2" damping="0.5"/>
        <joint type="hinge" axis="1 0 1" pos="0.05 0 0" ref="0.3" armature="0.02" damping="0.1" stiffness="2" springref="-0.1"/>
        <geom type="capsule" size="0.03 0.1" mass="0.7" pos="0 0 -0.1" quat="0.7 0.7 0 0"/>
        <body name="ball" pos="0 0 -0.3">
          <joint type="ball" pos="0.02 0 0" armature="0.01" damping="0.3"/>
          <geom type="sphere" size="0.05" mass="0.4" pos="0 0.03 0"/>
          <body name="welded" pos="0.1 0 0" quat="0 1 0 0">
            <geom type="box" size="0.02 0.02 0.02" mass="0.3"/>
            <body name="leaf" pos="0 0 -0.1" gravcomp="1">
              <joint type="hinge" axis="0 0 1"/>
              <geom type="cylinder" size="0.02 0.05" mass="0.2" quat="0.7 0.7 0 0"/>
            </body>
          </body>
        </body>
      </body>
      <body name="arm2" pos="-0.2 0 0">
        <joint type="hinge" axis="0 1 0" damping="0.2"/>
        <geom type="capsule" size="0.02 0.1" mass="0.5"/>
      </body>
    </body>
    <body name="fixed_arm" pos="1 0 0">
      <joint type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body pos="0.1 0 0">
        <joint type="slide" axis="1 0 0" damping="0.1"/>
        <geom type="sphere" size="0.03" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
G1 = os.path.join(os.path.dirname(__file__), "..", "examples", "mujoco_menagerie", "unitree_g1", "g1.xml")


def _models():
    out = [("features", mujoco.MjModel.from_xml_string(MODEL))]
    if os.path.exists(G1):
        out.append(("g1", mujoco.MjModel.from_xml_path(G1)))
    return out


def random_state(m, rng, scale=0.5):
    q = np.asarray(m.qpos0) + scale * rng.standard_normal(m.nq)
    mujoco.mj_normalizeQuat(m, q)
    return q, rng.standard_normal(m.nv), 3.0 * rng.standard_normal(m.nv)


def reference(m, q, qd, u, kind):
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_ACTUATION
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = q, qd
    if kind == "inverse":
        d.qacc[:] = u
        mujoco.mj_inverse(m, d)
        return d.qfrc_inverse.copy()
    d.qfrc_applied[:] = u
    mujoco.mj_forward(m, d)
    return d.qacc.copy()


@pytest.mark.parametrize("name,m", _models(), ids=lambda x: x if isinstance(x, str) else "")
def test_dynamics_match_mujoco(name, m):
    model = rbd.RigidBodyModel.from_mujoco(m)
    assert (model.nq, model.nv, model.nbody) == (m.nq, m.nv, m.nbody)
    rng = np.random.default_rng(0)
    inv = jax.jit(model.inverse_dynamics)
    fwd = jax.jit(model.forward_dynamics)
    for _ in range(5):
        q, qd, u = random_state(m, rng)
        tau_ref = reference(m, q, qd, u, "inverse")
        np.testing.assert_allclose(inv(q, qd, u), tau_ref, rtol=1e-11, atol=1e-11 * np.abs(tau_ref).max())
        qdd_ref = reference(m, q, qd, u, "forward")
        np.testing.assert_allclose(fwd(q, qd, u), qdd_ref, rtol=1e-10, atol=1e-10 * np.abs(qdd_ref).max())
        # forward and inverse are inverses of each other
        np.testing.assert_allclose(inv(q, qd, fwd(q, qd, u)), u, rtol=1e-10, atol=1e-10 * np.abs(u).max())


@pytest.mark.parametrize("name,m", _models(), ids=lambda x: x if isinstance(x, str) else "")
def test_kinematics_and_integration_match_mujoco(name, m):
    model = rbd.RigidBodyModel.from_mujoco(m)
    rng = np.random.default_rng(1)
    d = mujoco.MjData(m)
    for _ in range(3):
        q, qd, _ = random_state(m, rng)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        xpos, xmat = model.forward_kinematics(q)
        np.testing.assert_allclose(xpos, d.xpos, atol=1e-12)
        np.testing.assert_allclose(xmat, d.xmat.reshape(-1, 3, 3), atol=1e-12)
        q1 = q.copy()
        mujoco.mj_integratePos(m, q1, qd, 0.01)
        np.testing.assert_allclose(model.integrate_position(q, qd, 0.01), q1, atol=1e-12)


def test_integrate_position_is_smooth_at_zero_velocity():
    m = mujoco.MjModel.from_xml_string(MODEL)
    model = rbd.RigidBodyModel.from_mujoco(m)
    q = np.asarray(m.qpos0)
    for qd in (np.zeros(m.nv), 1e-6 * np.ones(m.nv)):
        J = np.asarray(jax.jacfwd(lambda v: model.integrate_position(q, v, 0.01))(jnp.asarray(qd)))
        assert np.all(np.isfinite(J))
        q1 = q.copy()
        mujoco.mj_integratePos(m, q1, qd, 0.01)
        np.testing.assert_allclose(model.integrate_position(q, qd, 0.01), q1, atol=1e-12)
    # the series branch and the exact branch agree at the switch (th2 = 1e-8 -> |w| dt = 1e-4)
    w = np.array([0.6, -0.8, 0.0]) * 1e-4 / 0.01
    a = rbd.quat_integrate(jnp.array([1.0, 0, 0, 0]), jnp.asarray(w * 0.999), 0.01)
    b = rbd.quat_integrate(jnp.array([1.0, 0, 0, 0]), jnp.asarray(w * 1.001), 0.01)
    assert np.linalg.norm(np.asarray(a) - np.asarray(b)) < 1e-6


def test_inverse_dynamics_to_sx_pattern():
    m = mujoco.MjModel.from_xml_string(MODEL)
    model = rbd.RigidBodyModel.from_mujoco(m)
    rng = np.random.default_rng(2)
    f = model.inverse_dynamics
    syms, (y,), _, _ = trace_to_sx(f, [(m.nq,), (m.nv,), (m.nv,)], [np.float64] * 3, ["q", "qd", "qdd"])
    x = ca.vertcat(*syms)
    J = ca.jacobian(y, x)
    structural = np.zeros((m.nv, m.nq + 2 * m.nv), bool)
    structural[J.sparsity().get_triplet()[0], J.sparsity().get_triplet()[1]] = True
    Jf = ca.Function("J", [x], [ca.densify(J)])
    for _ in range(3):
        q, qd, u = random_state(m, rng)
        args = [jnp.asarray(a) for a in (q, qd, u)]
        J_ref = np.hstack([np.asarray(b) for b in jax.jacfwd(f, argnums=(0, 1, 2))(*args)])
        np.testing.assert_allclose(np.asarray(Jf(np.concatenate([q, qd, u]))), J_ref, rtol=1e-10, atol=1e-10)
        assert np.all(structural[np.abs(J_ref) > 1e-12])
    assert J.nnz() < 0.8 * J.numel()           # tree sparsity survives


@pytest.mark.parametrize("name,m", _models(), ids=lambda x: x if isinstance(x, str) else "")
def test_body_jacobian_matches_mujoco(name, m):
    model = rbd.RigidBodyModel.from_mujoco(m)
    rng = np.random.default_rng(4)
    d = mujoco.MjData(m)
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(3):
        q, _, _ = random_state(m, rng)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        J = np.asarray(model.body_jacobian(q))
        for b in range(m.nbody):
            mujoco.mj_jacBody(m, d, jacp, jacr, b)
            np.testing.assert_allclose(J[b, :3], jacp, atol=1e-12)
            np.testing.assert_allclose(J[b, 3:], jacr, atol=1e-12)


@pytest.mark.skipif(not os.path.exists(G1), reason="Menagerie G1 not present")
def test_g1_kinematics_graphs_no_larger_than_the_paper():
    """Guard against graph bloat: fk / fkjac / fkhess of the G1 built as the cusadi2 paper's
    Menagerie benchmark did (stacked body placements, stacked sparsified 6 x nv body
    Jacobians, their d/dq), counted in CasADi instructions; the paper's pinocchio.casadi
    numbers were 2027 / 6341 / 68368, rbd measured 1685 / 4763 / 48625 (2026-11-06)."""
    m = mujoco.MjModel.from_xml_path(G1)
    model = rbd.RigidBodyModel.from_mujoco(m)

    def fk(q):
        xpos, xmat = model.forward_kinematics(q)
        return jnp.concatenate([jnp.concatenate([xpos[b], xmat[b].T.reshape(-1)]) for b in range(1, m.nbody)])

    def fkjac(q):
        J = model.body_jacobian(q)
        return jnp.concatenate([J[b].T.reshape(-1) for b in range(1, m.nbody)])

    (q,), (y,), _, _ = trace_to_sx(fk, [(m.nq,)], [np.float64], ["q"])
    assert ca.Function("fk", [q], [y]).n_instructions() <= 2027
    (q,), (y,), _, _ = trace_to_sx(fkjac, [(m.nq,)], [np.float64], ["q"])
    ys = ca.sparsify(y)
    assert ca.Function("fkjac", [q], [ys]).n_instructions() <= 6341
    assert ca.Function("fkhess", [q], [ca.jacobian(ys, q)]).n_instructions() <= 68368
