"""cusadi2.jax.interpret: jaxpr -> SX with exact sparsity.

Each case traces vectorized JAX code into SX, forms the symbolic Jacobian (and
Hessian of a random linear combination of the outputs), and checks
  * values against jax.jacfwd / jax.hessian at a random point, and
  * structural nnz == numeric nnz of the JAX derivative at that point
    (structural can never be smaller; equality means the pattern is perfect).
CPU only; no CUDA extension needed.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
import casadi as ca  # noqa: E402

from cusadi2.jax import Unsupported, trace_to_sx  # noqa: E402

jax.config.update("jax_enable_x64", True)
TOL = 1e-12


def _sx_function_derivatives(f, shape):
    (x,), (y,), (out_shape,), _ = trace_to_sx(f, [shape])
    J = ca.sparsify(ca.jacobian(y, x))          # drop explicit constant zeros (e.g. d sign / d x)
    lam = ca.SX.sym("lam", y.numel())
    H = ca.sparsify(ca.hessian(ca.dot(lam, y), x)[0])
    return x, y, J, lam, H, out_shape


def check(f, shape, seed=0, exact=True):
    rng = np.random.default_rng(seed)
    x, y, J, lam, H, out_shape = _sx_function_derivatives(f, shape)
    n = int(np.prod(shape))
    xr = rng.standard_normal(n)
    lr = rng.standard_normal(y.numel())
    f_flat = lambda v: f(v.reshape(shape)).reshape(-1)

    # value
    y_num = np.asarray(ca.Function("y", [x], [y])(xr)).reshape(-1)
    y_jax = np.asarray(f_flat(jnp.asarray(xr)))
    assert y_num.shape == y_jax.shape and tuple(out_shape) == f(jnp.asarray(xr).reshape(shape)).shape
    np.testing.assert_allclose(y_num, y_jax, rtol=TOL, atol=TOL)

    # jacobian: values and perfect pattern
    J_num = np.asarray(ca.Function("J", [x], [ca.densify(J)])(xr))
    J_jax = np.asarray(jax.jacfwd(f_flat)(jnp.asarray(xr))).reshape(J_num.shape)
    np.testing.assert_allclose(J_num, J_jax, rtol=TOL, atol=TOL)
    _check_pattern(J, J_num, lambda v: np.asarray(jax.jacfwd(f_flat)(jnp.asarray(v))).reshape(J_num.shape), n, rng, exact)

    # hessian of lam.y: values and perfect pattern
    H_num = np.asarray(ca.Function("H", [x, lam], [ca.densify(H)])(xr, lr))
    H_jax = np.asarray(jax.hessian(lambda v: jnp.dot(lr, f_flat(v)))(jnp.asarray(xr))).reshape(H_num.shape)
    np.testing.assert_allclose(H_num, H_jax, rtol=TOL, atol=TOL)
    _check_pattern(H, H_num, lambda v: np.asarray(jax.hessian(lambda w: jnp.dot(lr, f_flat(w)))(jnp.asarray(v))).reshape(H_num.shape), n, rng, exact)
    return J, H


def _check_pattern(S, S_num, numeric_at, n, rng, exact):
    """Structural pattern of S must equal the numerically generic pattern. For smooth
    functions one random point is generic (exact=True). For piecewise ops (max, abs,
    sign, clip, if_else) the structural pattern is the union over branches, so compare
    against the union of numeric patterns over several points and require superset."""
    pat = np.abs(S_num) > 1e-14
    if exact:
        assert S.nnz() == int(pat.sum()), f"pattern not perfect: structural {S.nnz()} vs numeric {int(pat.sum())}"
        return
    for _ in range(24):
        pat |= np.abs(numeric_at(rng.standard_normal(n))) > 1e-14
    structural = np.zeros(S_num.shape, bool)
    structural[S.sparsity().get_triplet()[0], S.sparsity().get_triplet()[1]] = True
    assert np.all(structural[pat]), "structural pattern misses a numerically nonzero entry"
    assert S.nnz() == int(pat.sum()), f"structural {S.nnz()} vs union-of-points {int(pat.sum())}"


# ----------------------------------------------------------------- the cases
def test_zero_weights_fold():
    Q = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
    J, H = check(lambda x: jnp.sum(Q * x ** 2)[None], (5,))
    assert J.nnz() == 3 and H.nnz() == 3


def test_dense_matmul_with_sparse_constant():
    A = np.zeros((6, 6)); A[np.arange(6), np.arange(6)] = 1.0; A[np.arange(5), np.arange(1, 6)] = 2.0
    J, _ = check(lambda x: A @ x, (6,))
    assert J.nnz() == 11


def test_where_constant_mask():
    mask = jnp.array([True, False, True, False])
    J, _ = check(lambda x: jnp.where(mask, jnp.sin(x), x[::-1] ** 2), (4,))
    assert J.nnz() == 4


def step(xk, uk):
    return jnp.stack([xk[0] + 0.1 * xk[1], xk[1] + 0.1 * (uk[0] - jnp.sin(xk[0]))])


def residual_vmap(z):
    X = z[:12].reshape(6, 2); U = z[12:].reshape(5, 1)
    return (X[1:] - jax.vmap(step)(X[:-1], U)).ravel()


def residual_scan(z):
    X = z[:12].reshape(6, 2); U = z[12:].reshape(5, 1)
    def body(c, inp):
        xk, uk, xn = inp
        return c, xn - step(xk, uk)
    _, r = lax.scan(body, 0.0, (X[:-1], U, X[1:]))
    return r.ravel()


def residual_fori(z):
    X = z[:12].reshape(6, 2); U = z[12:].reshape(5, 1)
    def body(k, r):
        return r.at[k].set(X[k + 1] - step(X[k], U[k]))
    return lax.fori_loop(0, 5, body, jnp.zeros((5, 2))).ravel()


@pytest.mark.parametrize("residual", [residual_vmap, residual_scan, residual_fori, jax.jit(residual_vmap)])
def test_mpc_residual_formulations_agree(residual):
    J, H = check(residual, (17,))
    assert (J.nnz(), H.nnz()) == (35, 5)


def test_pad_and_mask():
    def masked(x):
        xp = jnp.pad(x, (0, 3))
        m = jnp.array([1.0, 0, 1, 0, 1, 0, 0, 0])
        return (m * jnp.cos(xp))[:5]
    J, _ = check(masked, (5,))
    assert J.nnz() == 3


def test_static_gather_and_scatter():
    idx = jnp.array([3, 0, 3])
    def f(x):
        g = x[idx] * x[1]                     # gather with constant indices
        return jnp.zeros(4).at[jnp.array([0, 2])].add(g[:2]) + x.at[1].set(0.0)[::-1]
    check(f, (4,))


def test_dot_general_batch_dims_and_reductions():
    def f(x):
        M = x.reshape(2, 3, 2)
        prods = jnp.einsum("bij,bjk->bik", M, jnp.swapaxes(M, 1, 2))   # batched matmul
        return jnp.concatenate([prods.max(axis=(1, 2)), jnp.cumsum(x)[:3], jnp.prod(M, axis=1).ravel()])
    check(f, (12,), exact=False)   # max is piecewise: union pattern


def test_cond_constant_and_symbolic_predicate():
    def f(x):
        a = lax.cond(True, lambda v: v ** 2, lambda v: jnp.sin(v), x)        # constant: one branch
        b = lax.cond(x[0] > 0, lambda v: v[1] ** 2, lambda v: jnp.cos(v[2]), x)  # symbolic: if_else
        return jnp.concatenate([a, b[None]])
    (x,), (y,), _, _ = trace_to_sx(f, [(3,)])
    J = ca.sparsify(ca.jacobian(y, x))
    assert J.nnz() == 3 + 2            # diag of a, union pattern of b's two branches
    xr = np.array([0.3, -1.2, 0.7])
    np.testing.assert_allclose(np.asarray(ca.Function("y", [x], [y])(xr)).ravel(), np.asarray(f(jnp.asarray(xr))), atol=TOL)


def test_elementwise_coverage():
    def f(x):
        a = jnp.abs(x) + 1.5
        return jnp.stack([jnp.exp(x[0]), jnp.log(a[1]), jnp.tanh(x[2]), jnp.sqrt(a[3]), jnp.arctan2(x[0], a[1]),
                          jnp.maximum(x[2], x[3]), jnp.sign(x[0]) * x[1], jnp.clip(x[2], -0.5, 0.5) * x[3],
                          jax.nn.sigmoid(x[1]), jax.nn.softplus(x[2]), x[0] ** 3, jnp.power(a[1], x[2]),
                          jnp.log1p(a[3]), jnp.expm1(x[0]), jnp.cbrt(a[1]), jnp.arcsinh(x[2]),
                          jnp.mean(x), jnp.min(x), jnp.square(x[3]) / a[0]])
    check(f, (4,), seed=3, exact=False)


def test_jnp_linalg_style_ops():
    def f(x):
        M = x.reshape(3, 3)
        return (M @ M.T + jnp.eye(3) * x[0]).ravel()[:5] + jnp.trace(M)
    check(f, (9,))


def test_while_loop_is_rejected():
    def f(x):
        return lax.while_loop(lambda s: s[0] < 1.0, lambda s: s * 2.0, x)
    with pytest.raises(Unsupported):
        trace_to_sx(f, [(2,)])


def test_symbolic_index_is_rejected():
    def f(x):
        return x[jnp.int32(x[0] > 0)]
    with pytest.raises(Unsupported):
        trace_to_sx(f, [(2,)])


def test_multiple_inputs_and_outputs():
    def f(x, p):
        return jnp.sin(x) * p[0], jnp.sum(x * p) + p[1] ** 2
    (x, p), (y0, y1), shapes, _ = trace_to_sx(f, [(3,), (3,)])
    assert shapes == [(3,), ()]
    assert ca.jacobian(y0, x).nnz() == 3 and ca.jacobian(y0, p).nnz() == 3
    assert ca.jacobian(y1, x).nnz() == 3 and ca.jacobian(y1, p).nnz() == 3


# ----------------------------------------------------------------- rules added for MuJoCo MJX
def test_vmapped_gather_and_scatter_batching_dims():
    """jax.vmap over indexed reads / writes lowers to gather / scatter with batching dims."""
    rows = jnp.array([3, 0, 5, 1])
    def f(x):
        a = x.reshape(4, 6)
        g = jax.vmap(lambda r, i: r[i] * r[0])(a, rows)                # gather, batching dims
        s = jax.vmap(lambda r, i: r.at[i].set(r[2] ** 2))(a, rows)      # scatter
        t = jax.vmap(lambda r, i: r.at[i].add(r[1]))(a, rows)           # scatter-add
        return jnp.concatenate([g, s[:, :3].reshape(-1), t[:, 3:].reshape(-1)])
    check(f, (24,))


def test_constant_subgraphs_evaluate_through_jax():
    """Index bookkeeping on constants (sort / argsort, nonzero, tile, all, empty) is evaluated by
    JAX itself; only the symbolic path needs SX rules."""
    mask = jnp.array([True, False, True, True, False])
    keys = jnp.array([4, 1, 3, 0, 2])
    def f(x):
        order = jnp.argsort(keys)                                       # sort on constants
        nz = jnp.nonzero(mask, size=3)[0]                               # hi primitive on constants
        picks = jnp.tile(mask[None, :], (2, 1))                         # tile
        ok = jnp.all(picks, axis=1)                                     # reduce_and
        base = jnp.concatenate([jnp.empty((0,)), x[order] * x[nz][0]])  # empty + static gathers
        return jnp.where(ok[0], base, -base) + jnp.where(jnp.isfinite(x), x ** 2, 0.0)  # is_finite: assumed True
    check(f, (5,))


def test_symbolic_sort_is_rejected():
    with pytest.raises(Unsupported):
        trace_to_sx(lambda x: jnp.sort(x), [(4,)])


def test_dot_general_with_empty_contraction():
    def f(x):
        a = x.reshape(3, 2)[:, :0]                                      # (3, 0) @ (0, 2) -> zeros (3, 2)
        return (a @ jnp.zeros((0, 2))).reshape(-1) + x * x[0]
    check(f, (6,))
