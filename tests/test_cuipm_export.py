"""cusadi2.jax.ipm.export_cuipm: the quadcopter NMPC defined in JAX exported to cuipm's nine
plans. Checks against the hand-written CasADi NLP (values of all nine functions at random
points, exact structural nnz), the plan files' own host verifier, and, when cuipm's codegen
plans exist on this machine, the patterns of its c / d plans. CPU only."""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import casadi as ca  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cusadi2 import extract_level_plan, read_plan, verify_plan  # noqa: E402
from cusadi2.jax.ipm import NAMES, export_cuipm  # noqa: E402
from tests.problems.quadcopter_nav import quadcopter_nav  # noqa: E402
from tests.problems.quadcopter_nav_casadi import build_quadcopter_nav_sx, reference_nnz  # noqa: E402

jax.config.update("jax_enable_x64", True)
N = 30
CUIPM_PLANS = os.path.expanduser("~/code/cuipm/codegen/plans")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    f, c, d, z0, p0, dims = quadcopter_nav(N=N)
    out = tmp_path_factory.mktemp("plans")
    info = export_cuipm(lambda z, p: f(z), c, lambda z, p: d(z), z0, p0, str(out),
                        d_L=np.zeros(dims["nyd"]), d_U=np.full(dims["nyd"], np.inf))
    return info, dims


def test_dims_and_exact_nnz(exported):
    info, dims = exported
    ref = reference_nnz(N=N)
    assert (info["nx"], info["np"], info["nyc"], info["nyd"]) == (dims["nx"], dims["np"], dims["nyc"], dims["nyd"])
    for name in ("jac_f", "jac_c", "jac_d", "hess_f", "hess_c", "hess_d"):
        assert info["plans"][name]["nnz"] == ref[name], name
    assert info["plans"]["f"]["n_in"] == 3 and info["plans"]["hess_c"]["nnz_in"] == [dims["nx"], dims["np"], dims["nyc"]]
    assert os.path.exists(os.path.join(os.path.dirname(info["plans"]["f"]["path"]), "problem.json"))


def test_values_match_hand_written_casadi(exported):
    info, dims = exported
    fns = info["functions"]
    x, p, f_sx, c_sx, d_sx = build_quadcopter_nav_sx(N)
    lam_c, lam_d = ca.SX.sym("lam_c", c_sx.numel()), ca.SX.sym("lam_d", d_sx.numel())
    ref = {
        "f": ca.Function("f", [x, p], [f_sx]), "c": ca.Function("c", [x, p], [c_sx]), "d": ca.Function("d", [x, p], [d_sx]),
        "jac_f": ca.Function("jf", [x, p], [ca.jacobian(f_sx, x).T]),
        "jac_c": ca.Function("jc", [x, p], [ca.jacobian(c_sx, x)]), "jac_d": ca.Function("jd", [x, p], [ca.jacobian(d_sx, x)]),
        "hess_f": ca.Function("hf", [x, p], [ca.hessian(f_sx, x)[0]]),
        "hess_c": ca.Function("hc", [x, p, lam_c], [ca.hessian(ca.dot(lam_c, c_sx), x)[0]]),
        "hess_d": ca.Function("hd", [x, p, lam_d], [ca.hessian(ca.dot(lam_d, d_sx), x)[0]]),
    }
    rng = np.random.default_rng(0)
    for _ in range(3):
        xv, pv = rng.standard_normal(dims["nx"]), rng.standard_normal(dims["np"])
        lc, ld = rng.standard_normal(dims["nyc"]), rng.standard_normal(dims["nyd"])
        for name in NAMES:
            if name in ("f", "jac_f", "hess_f"):
                got, want = fns[name](xv, pv, 0.0), ref[name](xv, pv)
            elif name == "hess_c":
                got, want = fns[name](xv, pv, lc), ref[name](xv, pv, lc)
            elif name == "hess_d":
                got, want = fns[name](xv, pv, ld), ref[name](xv, pv, ld)
            else:
                got, want = fns[name](xv, pv), ref[name](xv, pv)
            np.testing.assert_allclose(np.asarray(ca.densify(got)), np.asarray(ca.densify(want)), rtol=1e-12, atol=1e-12,
                                       err_msg=name)


def test_plan_files_are_faithful(exported):
    info, _ = exported
    for name in ("c", "jac_c", "hess_c", "hess_f"):
        fn = info["functions"][name]
        assert verify_plan(fn, extract_level_plan(fn)) < 1e-9
        pl = read_plan(info["plans"][name]["path"])
        assert pl["n_in"] == fn.n_in() and pl["n_out"] == 1


@pytest.mark.skipif(not os.path.exists(os.path.join(CUIPM_PLANS, "hess_c.bin")), reason="cuipm codegen plans not present")
def test_patterns_match_cuipm_codegen_plans(exported):
    """cuipm's own plans (hand-written CasADi + eps hack): c / d-side patterns identical,
    f-side patterns are a superset of ours by the eps hack's diagonal entries."""
    info, _ = exported

    for name in ("c", "d", "jac_c", "jac_d", "hess_c", "hess_d"):
        theirs = read_plan(os.path.join(CUIPM_PLANS, f"{name}.bin"))
        ours = read_plan(info["plans"][name]["path"])
        assert theirs["n_in"] == ours["n_in"], name
        assert _nnz_out(theirs) == _nnz_out(ours) == info["plans"][name]["nnz"], name
    for name in ("jac_f", "hess_f"):
        theirs = read_plan(os.path.join(CUIPM_PLANS, f"{name}.bin"))
        assert _nnz_out(theirs) >= info["plans"][name]["nnz"], name


def _nnz_out(pl):
    return int(pl["outputs"][0]["nnz"])
