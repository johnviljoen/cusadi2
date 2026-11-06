"""cusadi2.jax.jax_function on the GPU: values under jit / vmap, sparse
Jacobians and Hessians (values and COO coordinates) against JAX autodiff, in
float64 and float32 (the dtype follows the arrays). Skipped without the FFI
extension or a CUDA device."""
import os
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
cj = pytest.importorskip("cusadi2.jax")
if not hasattr(cj, "jax_function"):
    pytest.skip("cusadi2.jax._ffi extension not built", allow_module_level=True)
if not any(d.platform == "gpu" for d in jax.devices()):
    pytest.skip("no CUDA device", allow_module_level=True)

from cusadi2.jax import jax_function  # noqa: E402

TOLS = {np.dtype("float64"): 1e-12, np.dtype("float32"): 2e-4}


def dense_from_coo(coo, values, shape):
    out = np.zeros(shape)
    out[tuple(np.asarray(coo).T)] = np.asarray(values)
    return out


def f(x, p):
    """dynamics-like coupling plus a parameter; 4 outputs."""
    return jnp.stack([x[0] * p[0] + jnp.sin(x[1]), x[1] * x[2] ** 2, jnp.exp(x[3]) * p[1], x[0] - x[3]])


@pytest.fixture(scope="module", params=["float64", "float32"], ids=["f64", "f32"])
def dtype(request):
    return np.dtype(request.param)


@pytest.fixture(scope="module")
def fn(dtype):
    return jax_function(f, param_argnums=(1,))


@pytest.fixture
def data(dtype):
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.standard_normal(4), dtype=dtype), jnp.asarray(rng.standard_normal(2), dtype=dtype)


def f64(*arrays):
    return [jnp.asarray(a, dtype=jnp.float64) for a in arrays]


def test_value_eager_jit_vmap(fn, data, dtype):
    x, p = data
    tol = TOLS[dtype]
    y = fn(x, p)
    assert y.dtype == dtype
    np.testing.assert_allclose(y, f(*f64(x, p)), rtol=tol, atol=tol)
    np.testing.assert_allclose(jax.jit(fn)(x, p), f(*f64(x, p)), rtol=tol, atol=tol)
    X = jnp.stack([x, 2 * x, -x])
    P = jnp.stack([p, p, 3 * p])
    np.testing.assert_allclose(jax.vmap(fn)(X, P), jax.vmap(f)(*f64(X, P)), rtol=tol, atol=tol)
    np.testing.assert_allclose(jax.jit(jax.vmap(fn, in_axes=(0, None)))(X, p),
                               jax.vmap(f, in_axes=(0, None))(*f64(X, p)), rtol=tol, atol=tol)
    XX = jnp.stack([X, X + 1])     # nested vmap flattens into one batch
    np.testing.assert_allclose(jax.vmap(jax.vmap(fn, in_axes=(0, None)), in_axes=(0, None))(XX, p),
                               jax.vmap(jax.vmap(f, in_axes=(0, None)), in_axes=(0, None))(*f64(XX, p)),
                               rtol=tol, atol=tol)


def test_jacobian_values_and_pattern(fn, data, dtype):
    x, p = data
    tol = TOLS[dtype]
    jac = fn.jacobian()
    coo, vals = jac.coo(x, p), jac(x, p)
    assert vals.dtype == dtype and coo.dtype == jnp.int32
    J = dense_from_coo(coo, vals, jac.shape(x, p))
    J_ref = np.asarray(jax.jacfwd(f)(*f64(x, p)))
    np.testing.assert_allclose(J, J_ref, rtol=tol, atol=tol)
    assert coo.shape[0] == int(np.count_nonzero(np.abs(J_ref) > 1e-14))     # exact pattern
    X = jnp.stack([x, -x])
    V = jax.jit(jax.vmap(jac, in_axes=(0, None)))(X, p)
    assert V.shape == (2, coo.shape[0])
    np.testing.assert_allclose(dense_from_coo(coo, V[1], J.shape), jax.jacfwd(f)(*f64(-x, p)), rtol=tol, atol=tol)


def test_true_hessian_tensor(fn, data, dtype):
    x, p = data
    tol = TOLS[dtype]
    hes = fn.hessian()
    coo, vals = hes.coo(x, p), hes(x, p)
    assert coo.shape[1] == 3
    H = dense_from_coo(coo, vals, hes.shape(x, p))
    H_ref = np.asarray(jax.hessian(f)(*f64(x, p)))           # (4, 4, 4): d2 f_k / dx_i dx_j
    np.testing.assert_allclose(H, H_ref, rtol=tol, atol=tol)
    assert coo.shape[0] == int(np.count_nonzero(np.abs(H_ref) > 1e-14))


def test_weighted_hessian(fn, data, dtype):
    x, p = data
    tol = TOLS[dtype]
    lam = jnp.asarray([0.3, -1.1, 2.0, 0.7], dtype=dtype)
    whes = fn.hessian(weighted=True)
    coo, vals = whes.coo(x, p), whes(x, p, lam=lam)
    H = dense_from_coo(coo, vals, whes.shape(x, p))
    x64, p64, lam64 = f64(x, p, lam)
    H_ref = np.asarray(jax.hessian(lambda v: jnp.dot(lam64, f(v, p64)))(x64))
    np.testing.assert_allclose(H, H_ref, rtol=tol, atol=tol)
    assert coo.shape[0] == int(np.count_nonzero(np.abs(H_ref) > 1e-14))
    X, L = jnp.stack([x, 2 * x]), jnp.stack([lam, -lam])
    V = jax.vmap(lambda xx, ll: whes(xx, p, lam=ll))(X, L)
    np.testing.assert_allclose(dense_from_coo(coo, V[1], H.shape),
                               jax.hessian(lambda v: jnp.dot(-lam64, f(v, p64)))(2 * x64), rtol=tol, atol=tol)


def test_pytrees_and_multiple_outputs():
    def g(xs, q):
        a, b = xs["a"], xs["b"]
        return jnp.sin(a) * q, {"s": jnp.sum(a * b), "v": b ** 2}
    gn = jax_function(g, param_argnums=(1,))
    xs = {"a": jnp.asarray([0.1, 0.2, 0.3]), "b": jnp.asarray([1.0, -2.0, 0.5])}
    q = jnp.asarray(2.5)
    out = jax.jit(gn)(xs, q)
    ref = g(xs, q)
    tol = TOLS[np.dtype("float64")]
    np.testing.assert_allclose(out[0], ref[0], rtol=tol, atol=tol)
    np.testing.assert_allclose(out[1]["s"], ref[1]["s"], rtol=tol, atol=tol)
    np.testing.assert_allclose(out[1]["v"], ref[1]["v"], rtol=tol, atol=tol)
    jac = gn.jacobian()
    coos, vals, shapes = jac.coo(xs, q), jac(xs, q), jac.shape(xs, q)
    assert len(coos) == 3 and shapes == ((3, 6), (1, 6), (3, 6))
    Ja, Jb = jax.jacfwd(lambda a, b: g({"a": a, "b": b}, q)[0], argnums=(0, 1))(xs["a"], xs["b"])
    np.testing.assert_allclose(dense_from_coo(coos[0], vals[0], shapes[0]), np.hstack([Ja, Jb]), rtol=tol, atol=tol)


def test_plans_are_cached_per_signature_and_dtype():
    gn = jax_function(f, param_argnums=(1,))
    x, p = jnp.ones(4), jnp.ones(2)
    gn(x, p); jax.jit(gn)(x, p); jax.vmap(gn, in_axes=(0, None))(jnp.stack([x, x]), p)
    assert len(gn._plans) == 1                      # same signature: one plan
    gn(x.astype(jnp.float32), p.astype(jnp.float32))
    assert len(gn._plans) == 2                      # float32 arrays: a second, float32 plan
    assert {pl.dtype for pl in gn._plans.values()} == {np.dtype("float64"), np.dtype("float32")}


def test_dtype_override():
    gn = jax_function(f, param_argnums=(1,), dtype=jnp.float32)
    y = gn(jnp.ones(4), jnp.ones(2))                # float64 arrays in, float32 plan requested
    assert y.dtype == jnp.float32


def test_x64_disabled_means_float32():
    """With jax_enable_x64 off (the JAX default) everything is float32 end to end."""
    code = """
import jax, jax.numpy as jnp, numpy as np
from cusadi2.jax import jax_function
f = lambda x, p: jnp.stack([x[0] * p[0] + jnp.sin(x[1]), x[1] * x[2] ** 2])
fn = jax_function(f, param_argnums=(1,))
x, p = jnp.asarray([0.3, -1.2, 0.7]), jnp.asarray([2.0, 0.5])
y = jax.jit(fn)(x, p); J = fn.jacobian()(x, p)
assert y.dtype == jnp.float32 and J.dtype == jnp.float32, (y.dtype, J.dtype)
np.testing.assert_allclose(y, f(x, p), rtol=1e-6, atol=1e-6)
print("OK")
"""
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")   # this process holds the GPU's preallocation
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout + res.stderr


def test_constant_subgraph_under_jit():
    """The interpreter's constant fallback (argsort / nonzero on constants) must evaluate eagerly
    even when the first call, and hence the trace, happens inside jax.jit."""
    keys = jnp.array([2, 0, 1])
    def g(x, p):
        order = jnp.argsort(keys)
        return x[order] * p[jnp.nonzero(keys > 0, size=1)[0][0]]
    gn = jax_function(g, param_argnums=(1,))
    x, p = jnp.asarray([0.1, 0.2, 0.3]), jnp.asarray([2.0, 3.0, 4.0])
    np.testing.assert_allclose(jax.jit(gn)(x, p), g(x, p), rtol=1e-12, atol=1e-12)
