"""Acceptance test: the quadcopter NMPC (jaxipm's JAX definition, copied to
tests/problems) through cusadi2.jax, against JAX autodiff for values and
against cuipm's hand-written CasADi SX for the nnz of all nine plans.

f(z), c(z, p), d(z) with p the initial state as a parameter:
  1. f, c, d values under jit and vmap match the JAX functions to f64 round-off
  2. jac / weighted hess values match jax.jacfwd / jax.hessian at random points
  3. nnz of jac_f, jac_c, jac_d, hess_f, hess_c, hess_d match the hand-written SX
     (cuipm/codegen/generate.py without the eps hack)
"""
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
from tests.problems.quadcopter_nav import quadcopter_nav  # noqa: E402
from tests.problems.quadcopter_nav_casadi import reference_nnz  # noqa: E402

TOL = 1e-10
N = 30


def dense_from_coo(coo, values, shape):
    out = np.zeros(shape)
    out[tuple(np.asarray(coo).T)] = np.asarray(values)
    return out


@pytest.fixture(scope="module")
def problem():
    f, c, d, z_init, p_init, dims = quadcopter_nav(N=N)
    fns = dict(f=jax_function(f, name="f"), c=jax_function(c, param_argnums=(1,), name="c"),
               d=jax_function(d, name="d"))
    rng = np.random.default_rng(1)
    z = jnp.asarray(np.asarray(z_init) + 0.1 * rng.standard_normal(z_init.shape))
    p = jnp.asarray(np.asarray(p_init) + 0.1 * rng.standard_normal(p_init.shape))
    return dict(f=f, c=c, d=d, fns=fns, z=z, p=p, dims=dims, rng=rng)


def test_values_jit_vmap(problem):
    f, c, d, fns, z, p = (problem[k] for k in ("f", "c", "d", "fns", "z", "p"))
    np.testing.assert_allclose(jax.jit(fns["f"])(z), f(z), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(jax.jit(fns["c"])(z, p), c(z, p), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(jax.jit(fns["d"])(z), d(z), rtol=TOL, atol=TOL)
    B = 8
    Z = z[None] + 0.05 * jnp.asarray(problem["rng"].standard_normal((B, z.size)))
    P = p[None] + 0.05 * jnp.asarray(problem["rng"].standard_normal((B, p.size)))
    np.testing.assert_allclose(jax.jit(jax.vmap(fns["f"]))(Z), jax.vmap(f)(Z), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(jax.jit(jax.vmap(fns["c"]))(Z, P), jax.vmap(c)(Z, P), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(jax.jit(jax.vmap(fns["d"]))(Z), jax.vmap(d)(Z), rtol=TOL, atol=TOL)


def test_derivative_values(problem):
    f, c, d, fns, z, p, dims = (problem[k] for k in ("f", "c", "d", "fns", "z", "p", "dims"))
    nx = dims["nx"]
    jf, jc, jd = fns["f"].jacobian(), fns["c"].jacobian(), fns["d"].jacobian()
    np.testing.assert_allclose(dense_from_coo(jf.coo(z), jf(z), (1, nx))[0], jax.grad(f)(z), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(dense_from_coo(jc.coo(z, p), jc(z, p), (dims["nyc"], nx)), jax.jacfwd(c)(z, p), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(dense_from_coo(jd.coo(z), jd(z), (dims["nyd"], nx)), jax.jacfwd(d)(z), rtol=TOL, atol=TOL)
    lam_c = jnp.asarray(problem["rng"].standard_normal(dims["nyc"]))
    lam_d = jnp.asarray(problem["rng"].standard_normal(dims["nyd"]))
    hf, hc, hd = fns["f"].hessian(weighted=True), fns["c"].hessian(weighted=True), fns["d"].hessian(weighted=True)
    np.testing.assert_allclose(dense_from_coo(hf.coo(z), hf(z, lam=jnp.ones(())), (nx, nx)), jax.hessian(f)(z), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(dense_from_coo(hc.coo(z, p), hc(z, p, lam=lam_c), (nx, nx)),
                               jax.hessian(lambda v: jnp.dot(lam_c, c(v, p)))(z), rtol=TOL, atol=TOL)
    np.testing.assert_allclose(dense_from_coo(hd.coo(z), hd(z, lam=lam_d), (nx, nx)),
                               jax.hessian(lambda v: jnp.dot(lam_d, d(v)))(z), rtol=TOL, atol=TOL)
    # batched weighted Hessian of c, as the IPM evaluates it
    B = 4
    Z = z[None] + 0.05 * jnp.asarray(problem["rng"].standard_normal((B, z.size)))
    L = jnp.asarray(problem["rng"].standard_normal((B, dims["nyc"])))
    V = jax.jit(jax.vmap(lambda zz, ll: hc(zz, p, lam=ll)))(Z, L)
    assert V.shape == (B, hc.coo(z, p).shape[0])
    np.testing.assert_allclose(dense_from_coo(hc.coo(z, p), V[2], (nx, nx)),
                               jax.hessian(lambda v: jnp.dot(L[2], c(v, p)))(Z[2]), rtol=TOL, atol=TOL)


def test_nnz_matches_hand_written_casadi(problem):
    fns, z, p = problem["fns"], problem["z"], problem["p"]
    ref = reference_nnz(N=N)
    got = {
        "jac_f": fns["f"].jacobian().coo(z).shape[0],
        "jac_c": fns["c"].jacobian().coo(z, p).shape[0],
        "jac_d": fns["d"].jacobian().coo(z).shape[0],
        "hess_f": fns["f"].hessian(weighted=True).coo(z).shape[0],
        "hess_c": fns["c"].hessian(weighted=True).coo(z, p).shape[0],
        "hess_d": fns["d"].hessian(weighted=True).coo(z).shape[0],
    }
    assert got == {k: ref[k] for k in got}, (got, ref)
    assert (problem["dims"]["nx"], problem["dims"]["nyc"], problem["dims"]["nyd"]) == (ref["nx"], ref["nyc"], ref["nyd"])
