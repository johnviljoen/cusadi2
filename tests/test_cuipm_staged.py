"""cusadi2.jax.ipm.StagedNLP / export_cuipm_staged: the quadcopter NMPC given stage-wise,
composed into cuipm's nine whole-horizon plans. Checks: the composed functions (host) and
the composed plans (GPU) against CasADi's own whole-horizon derivatives of the same NLP at
random points; the plan files (version 4, merged global patterns, scratch for the reduced
Hessians); and that IPOPT finds the same optimum as on the whole-horizon `quadcopter_nav`."""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import casadi as ca  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cusadi2 import read_plan  # noqa: E402
from cusadi2.jax.ipm import NAMES, StagedNLP, build_functions, export_cuipm_staged, solve_ipopt  # noqa: E402
from tests.problems.quadcopter_nav import quadcopter_nav  # noqa: E402
from tests.problems.quadcopter_nav_staged import quadcopter_nav_staged  # noqa: E402

jax.config.update("jax_enable_x64", True)
N = 30
GPU = any(d.platform == "gpu" for d in jax.devices()) and hasattr(__import__("cusadi2.jax").jax, "jax_function")


@pytest.fixture(scope="module")
def staged():
    stage_f, stage_c, stage_d, n_stages, specs, c_active, d_active, x0, p_init, s = quadcopter_nav_staged(N=N)
    nlp = StagedNLP(stage_f, stage_c, stage_d, n_stages, specs, c_active=c_active, d_active=d_active)
    p0 = nlp.layout.pack("p", {"p": p_init, "s_k": s})
    return nlp, x0, p0


@pytest.fixture(scope="module")
def reference(staged):
    """CasADi's whole-horizon derivatives of the composed NLP (symbolic stage calls)."""
    nlp, _, _ = staged
    return build_functions(None, None, None, nlp.nx, nlp.np, nlp=nlp.whole_horizon())


def _args(name, x, p, lc, ld):
    if name in ("f", "jac_f", "hess_f"):
        return (x, p, np.zeros(1))
    if name == "hess_c":
        return (x, p, lc)
    if name == "hess_d":
        return (x, p, ld)
    return (x, p)


def _dense(fn, *args):
    return np.asarray(ca.densify(fn(*args)))


def test_layout_and_dims(staged):
    nlp, x0, p0 = staged
    assert (nlp.nx, nlp.np, nlp.nyc, nlp.nyd) == (N * 13 + (N - 1) * 4, 13 + 2 * (N - 1), N * 13, 4 * (N - 1))
    assert [b[0] for b in nlp.layout.blocks("x")] == ["x_k", "u_k"] and [b[0] for b in nlp.layout.blocks("p")] == ["p", "s_k"]
    assert nlp.c_rows[0, :13].tolist() == list(range(13)) and nlp.c_rows[1, 13:].max() == -1   # init rows at stage 0 only


def test_composed_functions_match_whole_horizon(staged, reference):
    nlp, x0, p0 = staged
    fns = nlp.functions()
    rng = np.random.default_rng(0)
    for _ in range(2):
        x, p = rng.standard_normal(nlp.nx), rng.standard_normal(nlp.np)
        lc, ld = rng.standard_normal(nlp.nyc), rng.standard_normal(nlp.nyd)
        for name in NAMES:
            got = fns[name].dense(*_args(name, x, p, lc, ld))
            want = _dense(reference[name], *_args(name, x, p, lc, ld))
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, err_msg=name)
    # the global patterns: Jacobians exact; Hessians contain CasADi's (union over the stages)
    for name in ("jac_c", "jac_d", "jac_f"):
        assert fns[name].nnz_out == reference[name].nnz_out(0), name
    for name in ("hess_f", "hess_c", "hess_d"):
        assert fns[name].nnz_out >= reference[name].nnz_out(0), name
    # f (summed over stages) and hess_f (the knot-0 and knot-1 blocks of consecutive stages meet)
    # go through the scratch stream and a reduce; the Jacobians are written directly; hess_c has
    # no overlap here (x_{k+1} enters the Euler constraint linearly)
    assert fns["f"].composition.scratch_per_instance == N - 1 and fns["hess_f"].composition.reduces
    assert fns["jac_c"].composition.scratch_per_instance == 0 and not fns["hess_c"].composition.reduces


def test_plan_files(staged, tmp_path):
    nlp, x0, p0 = staged
    info = export_cuipm_staged(nlp, x0, p0, str(tmp_path), d_L=np.zeros(nlp.nyd), d_U=np.full(nlp.nyd, np.inf))
    for name in NAMES:
        pl = read_plan(info["plans"][name]["path"])
        assert pl["version"] == 4 and pl["n_stages"] == N - 1
        assert pl["n_in"] == info["plans"][name]["n_in"] and pl["outputs"][0]["nnz"] == info["plans"][name]["nnz"]
        assert pl["inputs"][0]["nnz"] == nlp.nx and pl["inputs"][1]["nnz"] == nlp.np
    hf = read_plan(info["plans"]["hess_f"]["path"])
    assert hf["scratch_per_instance"] > 0 and len(hf["reduces"]) == 1 and hf["stage_out"][0]["arg"] == -1
    assert info["staged"]["x_layout"][1] == dict(name="u_k", base=N * 13, count=N - 1, size=4)
    assert os.path.exists(os.path.join(str(tmp_path), "problem.json"))


@pytest.mark.skipif(not GPU, reason="no CUDA device or FFI extension")
def test_gpu_plans_match_whole_horizon(staged, reference):
    from cusadi2.jax.ipm import gpu_functions
    nlp, x0, p0 = staged
    calls = gpu_functions(nlp.functions())
    rng = np.random.default_rng(1)
    B = 3
    X, P = rng.standard_normal((B, nlp.nx)), rng.standard_normal((B, nlp.np))
    Lc, Ld, E = rng.standard_normal((B, nlp.nyc)), rng.standard_normal((B, nlp.nyd)), np.zeros((B, 1))
    for name in NAMES:
        args = {"hess_c": (X, P, Lc), "hess_d": (X, P, Ld)}.get(name, (X, P, E) if name.endswith("f") else (X, P))
        got = np.asarray(jax.jit(jax.vmap(calls[name]))(*[jnp.asarray(a) for a in args]))
        assert got.shape == (B, calls[name].nnz)
        coo = calls[name].coo
        for b in range(B):
            M = np.zeros(reference[name].sparsity_out(0).shape)
            M[coo[:, 0], coo[:, 1]] = got[b]
            want = _dense(reference[name], *_args(name, X[b], P[b], Lc[b], Ld[b]))
            np.testing.assert_allclose(M, want, rtol=1e-12, atol=1e-12, err_msg=name)
    for g in calls.values():
        g.free()


def test_ipopt_same_optimum_as_whole_horizon(staged):
    nlp, x0, p0 = staged
    opts = {"max_iter": 200, "tol": 1e-8}
    res = solve_ipopt(None, None, None, x0, p0, d_L=np.zeros(nlp.nyd), d_U=np.full(nlp.nyd, np.inf), options=opts,
                      nlp=nlp.whole_horizon())
    f, c, d, z0, p_init, dims = quadcopter_nav(N=N)
    ref = solve_ipopt(lambda z, p: f(z), c, lambda z, p: d(z), z0, p_init, d_L=np.zeros(dims["nyd"]),
                      d_U=np.full(dims["nyd"], np.inf), options=opts)
    assert res["success"] and ref["success"]
    assert abs(res["f"] - ref["f"]) < 1e-6 * max(1.0, abs(ref["f"]))
    np.testing.assert_allclose(res["x"], ref["x"], rtol=1e-5, atol=1e-5)    # same global layout, same optimum
