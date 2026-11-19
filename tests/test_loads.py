"""Traced lookups (cusadi2.loads, OP_LOAD): the merged program on the host and the GPU, and the
JAX front end (gather / dynamic_slice at traced positions under to_mx)."""
import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cusadi2.dag import extract_level_plan, extract_serial_program, verify_plan, verify_serial_program
from cusadi2.jax import Unsupported, to_mx
from cusadi2.loads import LoadedFunction
from cusadi2.ops import load_arg, load_index, load_operand

jax.config.update("jax_enable_x64", True)


def interp_function(n=20):
    """1-D linear interpolation of a table in p at x[0], plus a clamped and a fixed read."""
    x, p, g = ca.SX.sym("x", 2), ca.SX.sym("p", n + 1), ca.SX.sym("g", 4)
    u = x[0] * (n - 1)
    i0 = ca.fmin(ca.fmax(ca.floor(u), 0), n - 2)
    t = u - i0
    I = ca.Function("I", [x, p], [ca.vertcat(i0, i0 + 1, n - 1 - i0, 1e9)])
    y = (1 - t) * g[0] + t * g[1] + x[1] * g[2] + g[3] * x[0] ** 2
    F = ca.Function("lookup", [x, p, g], [ca.vertcat(y, ca.gradient(y, x))])
    return LoadedFunction(F, I, table_arg=1), n


def test_index_and_operand():
    assert [load_index(v, 5) for v in (-1.0, 0.0, 2.9, 4.0, 7.0, float("nan"))] == [0, 0, 2, 4, 4, 0]
    assert load_arg(load_operand(3, 1000)) == (3, 1000)
    with pytest.raises(ValueError):
        load_operand(64, 10)


def test_loaded_function_host_programs():
    L, n = interp_function()
    rng = np.random.default_rng(0)
    P = rng.normal(size=n + 1)
    X = np.array([0.37, 0.5])
    u = X[0] * (n - 1); i = int(np.floor(u)); t = u - i
    ref = (1 - t) * P[i] + t * P[i + 1] + X[1] * P[n - 1 - i] + P[n] * X[0] ** 2
    assert float(np.asarray(ca.DM(L(X, P)).nonzeros())[0]) == pytest.approx(ref, abs=1e-14)
    assert L.n_in() == 2 and L.nnz_in(1) == n + 1
    assert verify_plan(L, extract_level_plan(L)) == 0.0
    assert verify_serial_program(L, extract_serial_program(L)) == 0.0
    assert verify_plan(L, extract_level_plan(L, window=4)) == 0.0


def test_loaded_function_gpu():
    pytest.importorskip("cusadi2.jax._ffi_f64")
    from cusadi2.jax.function import GpuFunction
    L, n = interp_function()
    rng = np.random.default_rng(1)
    P = rng.normal(size=n + 1)
    for serial in (False, True):
        G = GpuFunction("lookup", L, serial=serial)
        for _ in range(50):
            X = rng.uniform(-0.2, 1.2, 2)
            np.testing.assert_array_equal(np.asarray(G(X, P)), np.asarray(ca.DM(L(X, P)).nonzeros()))
        G.free()


def _grid_fn(x, p):
    T = p[0]                                     # (4, 5) table
    i = jnp.floor(x[0] * 3).astype(jnp.int32)
    a = jnp.take(T.reshape(-1), i * 5 + 2, mode="clip")
    b = T[i, i + 1]                              # dynamic_slice at traced starts
    return (a + b * x[1] + jnp.sum(T[0]),)


def test_to_mx_records_lookups():
    T = np.arange(20.0).reshape(4, 5)
    g = to_mx(_grid_fn, jnp.zeros(2), (jnp.asarray(T),))
    assert g.g is not None and g.idx is not None and g.g.numel() == 2
    F = ca.Function("F", g.x + g.p + [g.g], g.outs + [g.idx])
    for xv in ([0.1, 2.0], [0.5, -1.0], [0.9, 0.3], [1.7, 1.0]):
        xv = np.array(xv)
        # positions without the values (the values only enter f), then the values read there
        pos = np.asarray(ca.DM(F(xv, T.T, np.zeros(2))[1]).nonzeros())
        vals = T.reshape(-1)[[load_index(q, 20) for q in pos]]
        got = float(F(xv, T.T, vals)[0])
        assert got == pytest.approx(float(_grid_fn(jnp.asarray(xv), (jnp.asarray(T),))[0]), abs=1e-12)


def test_lookup_outside_p_is_unsupported():
    def fn(x, p):
        T = jnp.arange(10.0) * x[1]              # a table built from x, not a slice of p
        return (jnp.take(T, jnp.floor(x[0]).astype(jnp.int32), mode="clip"),)
    with pytest.raises(Unsupported):
        to_mx(fn, jnp.zeros(2), (jnp.zeros(1),))


def test_fill_mode_is_unsupported():
    def fn(x, p):
        return (jnp.take(p[0], jnp.floor(x[0]).astype(jnp.int32), mode="fill"),)
    with pytest.raises(Unsupported):
        to_mx(fn, jnp.zeros(1), (jnp.zeros(5),))
