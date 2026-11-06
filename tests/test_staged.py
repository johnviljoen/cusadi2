"""cusadi2.jax.staged.StagedFunction on the GPU: a per-stage function replicated over the
horizon by cusadi2's stage mode; values, Jacobian and Lagrangian Hessian against JAX on the
stage function, and the global pattern against the whole-horizon function's autodiff."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
cj = pytest.importorskip("cusadi2.jax")
if not hasattr(cj, "jax_function"):
    pytest.skip("cusadi2.jax FFI extension not built", allow_module_level=True)
if not any(d.platform == "gpu" for d in jax.devices()):
    pytest.skip("no CUDA device", allow_module_level=True)

from cusadi2.jax.staged import StagedFunction  # noqa: E402

N, NX, NU, NP = 5, 3, 2, 2
SPECS = [dict(name="x_k", size=NX, mode="coupled", offset=0, span=1),
         dict(name="x_k1", size=NX, mode="coupled", offset=1, span=1, alias="x_k"),
         dict(name="u_k", size=NU, mode="stage"),
         dict(name="p", size=NP, mode="shared", param=True),
         dict(name="w_k", size=1, mode="stage", param=True)]


def stage(x_k, x_k1, u_k, p, w_k):
    """Integrator residual with a weight per stage: x_{k+1} - x_k - dt f(x_k, u_k)."""
    f = jnp.stack([x_k[1] * p[0], jnp.sin(x_k[2]) + u_k[0], u_k[1] * x_k[0] ** 2 * p[1]])
    return w_k[0] * (x_k1 - x_k - 0.1 * f)


def whole(X, U, p, W):
    return jnp.concatenate([stage(X[k], X[k + 1], U[k], p, W[k]) for k in range(N)])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    B = 3
    X = jnp.asarray(rng.standard_normal((B, N + 1, NX)))
    U = jnp.asarray(rng.standard_normal((B, N, NU)))
    P = jnp.asarray(rng.standard_normal((B, NP)))
    W = jnp.asarray(rng.uniform(0.5, 1.5, (B, N, 1)))
    return X, U, P, W


@pytest.fixture(scope="module")
def sf():
    return StagedFunction(stage, n_stages=N, specs=SPECS, name="stage_test")


def test_value_matches_whole_horizon(sf, data):
    X, U, P, W = data
    flat = [X.reshape(X.shape[0], -1), X.reshape(X.shape[0], -1), U.reshape(U.shape[0], -1), P, W.reshape(W.shape[0], -1)]
    y = jax.vmap(sf)(*flat)
    ref = jax.vmap(whole)(X, U, P, W)
    np.testing.assert_allclose(y, ref, rtol=1e-12, atol=1e-12)
    assert y.shape == (3, N * NX)


def test_jacobian_and_global_pattern(sf, data):
    X, U, P, W = data
    b = 0
    flat = [X[b].reshape(-1), X[b].reshape(-1), U[b].reshape(-1), P[b], W[b].reshape(-1)]
    jac = sf.jacobian()
    vals = np.asarray(jac(*flat))
    rows, cols = jac.global_pattern()
    nxg = (N + 1) * NX + N * NU
    J = np.zeros((N * NX, nxg))
    J[rows, cols] = vals
    xg = np.concatenate([np.asarray(X[b]).reshape(-1), np.asarray(U[b]).reshape(-1)])
    def whole_flat(z):
        return whole(z[:(N + 1) * NX].reshape(N + 1, NX), z[(N + 1) * NX:].reshape(N, NU), P[b], W[b])
    J_ref = np.asarray(jax.jacfwd(whole_flat)(jnp.asarray(xg)))
    np.testing.assert_allclose(J, J_ref, rtol=1e-12, atol=1e-12)
    assert len(rows) == np.count_nonzero(np.abs(J_ref) > 1e-14)        # exact global pattern


def test_lagrangian_hessian(sf, data):
    X, U, P, W = data
    b = 1
    flat = [X[b].reshape(-1), X[b].reshape(-1), U[b].reshape(-1), P[b], W[b].reshape(-1)]
    lam = jnp.asarray(np.random.default_rng(1).standard_normal(N * NX))
    whs = sf.hessian(weighted=True)
    vals = np.asarray(whs(*flat, lam=lam))
    rows, cols = whs.global_pattern()
    nxg = (N + 1) * NX + N * NU
    H = np.zeros((nxg, nxg))
    H[rows, cols] = vals
    xg = np.concatenate([np.asarray(X[b]).reshape(-1), np.asarray(U[b]).reshape(-1)])
    def lag(z):
        return jnp.dot(lam, whole(z[:(N + 1) * NX].reshape(N + 1, NX), z[(N + 1) * NX:].reshape(N, NU), P[b], W[b]))
    H_ref = np.asarray(jax.hessian(lag)(jnp.asarray(xg)))
    np.testing.assert_allclose(H, H_ref, rtol=1e-12, atol=1e-12)
