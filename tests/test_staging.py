"""cusadi2.staging (stage structure from a CasADi MX graph) and cusadi2.jax.to_mx (JAX -> MX)."""
import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cusadi2.staging import NotStaged, StagingError, derive

jax.config.update("jax_enable_x64", True)
N = 5


def toy(loop: bool):
    X, U, p = ca.MX.sym("x", 2, N + 1), ca.MX.sym("u", 1, N), ca.MX.sym("p", 2)
    x0, x1, u = ca.SX.sym("x0", 2), ca.SX.sym("x1", 2), ca.SX.sym("u")
    F = ca.Function("dyn", [x0, x1, u], [x1 - x0 - 0.1 * ca.vertcat(x0[1], ca.sin(u))])
    L = ca.Function("cost", [x1, u], [ca.sumsqr(x1) + u ** 2])
    if loop:
        dyn = ca.horzcat(*[F(X[:, k], X[:, k + 1], U[:, k]) for k in range(N)])
        cost = sum(L(X[:, k + 1], U[:, k]) for k in range(N))
    else:
        dyn, cost = F.map(N)(X[:, :-1], X[:, 1:], U), ca.sum2(L.map(N)(X[:, 1:], U))
    f = cost + ca.sumsqr(X[:, N])
    return [X, U], [p], f, ca.vertcat(X[:, 0] - p, ca.vec(dyn)), ca.vec(U), (X, U, p, F, L)


def staged(loop):
    x, p, f, c, d, _ = toy(loop)
    return derive(x, p, {"f": f}, {"c": c, "d": d})


def evaluate(st, x, p):
    """Whole-horizon f, c, d from the staging (stage functions on the specs' gathers)."""
    X, U = x
    f, c, d = 0.0, {}, {}
    args_of = []
    for s in range(st.N):
        args = []
        for sp in st.specs:
            if sp["name"] == "p":
                args.append(p)
            elif sp["name"] == "consts":
                args.append(st.consts[s])
            else:
                leaf = {"x": X, "u": U}[sp.get("alias", sp["name"])]
                args.append(leaf[:, s + sp["offset"]])
        args_of.append(args)
        f += float(st.functions["f"](*args))
    for name, out in (("c", c), ("d", d)):
        for u, (s, r) in enumerate(st.user_rows[name]):
            out[u] = float(np.asarray(st.functions[name](*args_of[s])).reshape(-1)[r])
    return f, np.array([c[i] for i in range(len(c))]), np.array([d[i] for i in range(len(d))])


@pytest.mark.parametrize("loop", [False, True])
def test_derive_reproduces_the_graph(loop):
    x, p, f, c, d, _ = toy(loop)
    st = derive(x, p, {"f": f}, {"c": c, "d": d})
    assert [sp["name"] for sp in st.specs] == ["x", "x+1", "u", "p", "consts"] and st.n_maps == 2
    G = ca.Function("G", x + p, [f, c, d])
    rng = np.random.default_rng(0)
    xv = [rng.standard_normal(v.shape) for v in x]
    pv = rng.standard_normal(2)
    want = [np.asarray(o).reshape(-1) for o in G(*xv, pv)]
    got = evaluate(st, xv, pv)
    np.testing.assert_allclose(got[0], want[0][0], rtol=1e-12)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(got[2], want[2], rtol=1e-12, atol=1e-14)


def test_loop_and_map_give_the_same_staging():
    a, b = staged(False), staged(True)
    assert a.user_rows == b.user_rows and all(np.array_equal(a.active[k], b.active[k]) for k in "cd")
    rng = np.random.default_rng(1)
    for k in "fcd":
        ins = [rng.standard_normal(a.functions[k].size_in(i)) for i in range(a.functions[k].n_in())]
        np.testing.assert_array_equal(np.asarray(a.functions[k](*ins)), np.asarray(b.functions[k](*ins)))


def test_rejections():
    x, p, _, _, _, (X, U, P, F, L) = toy(False)
    zero = 0 * X[0, 0]
    with pytest.raises(StagingError, match="another map's output"):
        a = ca.SX.sym("a", 2)
        G2 = ca.Function("g2", [a], [2 * a])
        derive(x, p, {"f": zero}, {"c": ca.vec(G2.map(N)(F.map(N)(X[:, :-1], X[:, 1:], U))), "d": ca.MX(0, 1)})
    with pytest.raises(StagingError, match="same x slice in every iteration"):
        derive(x, p, {"f": zero}, {"c": ca.vec(F.map(N)(X[:, :-1], X[:, 1:], U[:, 0])), "d": ca.MX(0, 1)})
    with pytest.raises(NotStaged):
        derive(x, p, {"f": ca.sumsqr(X)}, {"c": X[:, 0] - P, "d": ca.MX(0, 1)})


@pytest.mark.parametrize("mapping", ["vmap", "lax_map"])
def test_to_mx_from_jax(mapping):
    from cusadi2.jax import to_mx
    vm = jax.vmap if mapping == "vmap" else (lambda f: lambda *a: jax.lax.map(lambda t: f(*t), a))

    def nlp(x, p):
        q, u = x["x"], x["u"]
        dyn = vm(lambda a, b, w: b - a - 0.1 * jnp.stack([a[1], jnp.sin(w[0])]))(q[:-1], q[1:], u)
        cost = vm(lambda b, w: jnp.sum(b ** 2) + w[0] ** 2)(q[1:], u)
        return jnp.sum(cost) + jnp.sum(q[N] ** 2), jnp.concatenate([q[0] - p, dyn.ravel()]), u.ravel()
    xe, pe = {"u": jnp.zeros((N, 1)), "x": jnp.zeros((N + 1, 2))}, jnp.zeros(2)
    g = to_mx(nlp, xe, pe)
    assert g.n_maps == 2 and g.x_names == ["u", "x"]
    st = derive(g.x, g.p, {"f": g.outs[0]}, {"c": g.outs[1], "d": g.outs[2]}, x_names=g.x_names)
    ref = staged(False)
    assert sorted(sp["name"] for sp in st.specs) == sorted(sp["name"] for sp in ref.specs)
    assert st.user_rows == ref.user_rows
    # the MX graph computes what JAX computes
    G = ca.Function("G", g.x + g.p, g.outs)
    rng = np.random.default_rng(2)
    xv = {"u": rng.standard_normal((N, 1)), "x": rng.standard_normal((N + 1, 2))}
    pv = rng.standard_normal(2)
    want = nlp(xv, pv)
    got = G(xv["u"].T, xv["x"].T, pv)
    for a, b in zip(got, want):
        np.testing.assert_allclose(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1), rtol=1e-12, atol=1e-14)
