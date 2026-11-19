"""output_order (cusadi2.order): "row" and "col" plans compute the same outputs (host verifiers and
the GPU, plain Functions and LoadedFunction), "auto" keeps the fitting / fewer-level plan, and the
order is part of the plan cache key and recorded in the plan's metadata."""
import json

import casadi as ca
import numpy as np
import pytest

import cusadi2
from cusadi2.cache import plan_key
from cusadi2.dag import extract_level_plan, extract_serial_program, verify_plan, verify_serial_program
from cusadi2.order import OrderedFunction, plan_in_order, with_output_order
from tests.test_loads import interp_function

R, C, K = 40, 8, 12


def _chain(v, k):
    for j in range(k):
        v = ca.sin(v) * (1.0 + 0.01 * j) + v * 0.3
    return v


def per_row_and_per_col():
    """M[r, c] = h_r * (c + 2) with h_r an expensive chain of x_r (favours row order: in CCS order
    the first column computes every h_r and they all stay live), and its transpose (favours col)."""
    x = ca.SX.sym("x", R)
    h = [_chain(x[r], K) for r in range(R)]
    rowM, colM = ca.SX(R, C), ca.SX(C, R)
    for r in range(R):
        for c in range(C):
            rowM[r, c] = h[r] * (c + 2.0)
            colM[c, r] = h[r] * (c + 2.0)
    return ca.Function("per_row", [x], [rowM]), ca.Function("per_col", [x], [colM])


def jacobian_function():
    """A sparse Jacobian and a second (vector) output: two outputs to map back."""
    x = ca.SX.sym("x", 6)
    y = ca.vertcat(*[ca.sin(x[i]) * x[(i + 1) % 6] + x[i] ** 2 for i in range(6)], ca.sum1(ca.exp(0.1 * x)))
    return ca.Function("jacfun", [x], [ca.jacobian(y, x), y])


@pytest.mark.parametrize("make", [jacobian_function, lambda: per_row_and_per_col()[0], lambda: interp_function()[0]])
def test_row_and_col_same_outputs_host(make):
    fn = make()
    rng = np.random.default_rng(0)
    args = [rng.uniform(-1, 1, fn.nnz_in(a)) for a in range(fn.n_in())]
    for order in ("row", "col"):
        f = with_output_order(fn, order)
        assert verify_plan(f, extract_level_plan(f)) == 0.0
        assert verify_plan(f, extract_level_plan(f, window=64)) == 0.0
        assert verify_serial_program(f, extract_serial_program(f)) == 0.0
    # the row stream writes the matrix outputs in a different order, mapped back to the same CCS outputs
    def writes(f):
        return [tuple(f.instruction_output(k)) for k in range(f.n_instructions()) if f.instruction_id(k) == ca.OP_OUTPUT]
    f_row = with_output_order(fn, "row")
    if any(not fn.sparsity_out(k).is_column() for k in range(fn.n_out())):
        assert writes(f_row) != writes(with_output_order(fn, "col"))
    assert sorted(writes(f_row)) == sorted(writes(with_output_order(fn, "col")))
    ref = fn(*[ca.DM(a) for a in args])
    got = f_row(*[ca.DM(a) for a in args])
    for r, g in zip(ref if isinstance(ref, (list, tuple)) else [ref], got if isinstance(got, (list, tuple)) else [got]):
        np.testing.assert_array_equal(np.asarray(ca.DM(r).nonzeros()), np.asarray(ca.DM(g).nonzeros()))


def test_col_is_the_function_as_built():
    fn = jacobian_function()
    assert with_output_order(fn, "col") is fn
    assert isinstance(with_output_order(fn, "row"), OrderedFunction)
    L, _ = interp_function()
    assert with_output_order(L, "row").output_order == "row"
    assert with_output_order(L, "col").output_order == "col"


@pytest.mark.parametrize("make", [jacobian_function, lambda: interp_function()[0]])
def test_row_and_col_same_outputs_gpu(make):
    pytest.importorskip("cusadi2.jax._ffi_f64")
    from cusadi2.jax.function import GpuFunction
    fn = make()
    rng = np.random.default_rng(1)
    gs = {o: GpuFunction(f"order_{o}", fn, output_order=o) for o in ("row", "col")}
    gs_serial = {o: GpuFunction(f"order_{o}_s", fn, output_order=o, serial=True) for o in ("row", "col")}
    for _ in range(20):
        args = [rng.uniform(-0.2, 1.2, fn.nnz_in(a)) for a in range(fn.n_in())]
        ref = np.asarray(ca.DM((fn(*[ca.DM(a) for a in args]) if fn.n_out() == 1 else fn(*[ca.DM(a) for a in args])[0])).nonzeros())
        # the two orders give bit-identical GPU outputs (each kernel); against CasADi on the host
        # to rounding (the GPU may contract a multiply-add into an FMA)
        for pair in (gs, gs_serial):
            np.testing.assert_array_equal(np.asarray(pair["row"](*args)), np.asarray(pair["col"](*args)))
            np.testing.assert_allclose(np.asarray(pair["row"](*args)), ref, rtol=1e-14, atol=1e-15)
    for g in list(gs.values()) + list(gs_serial.values()):
        g.free()


@pytest.mark.parametrize("which, budget, expect, both_fit", [
    ("per_row", 120, "row", False),     # only row fits
    ("per_row", 200, "row", True),      # both fit, row has fewer levels
    ("per_col", 200, "col", True),      # both fit, col has fewer levels
    ("per_col", 120, "col", False),     # only col fits
])
def test_auto_picks_fitting_then_fewer_levels(which, budget, expect, both_fit):
    fn = dict(zip(("per_row", "per_col"), per_row_and_per_col()))[which]
    plans = {}
    for o in ("row", "col"):
        try:
            plans[o] = plan_in_order(fn, o, budget)[1]
        except ValueError:
            pass
    assert (len(plans) == 2) == both_fit and expect in plans
    if both_fit:
        other = "col" if expect == "row" else "row"
        assert plans[expect].n_levels < plans[other].n_levels
    f, P, chosen = plan_in_order(fn, "auto", budget)
    assert chosen == expect and getattr(f, "output_order", "col") == expect
    assert (P.n_levels, P.peak_live_slots) == (plans[expect].n_levels, plans[expect].peak_live_slots)


def test_auto_tie_keeps_col_and_neither_fits_raises():
    fn = per_row_and_per_col()[0]
    _, P, chosen = plan_in_order(fn, "auto", None)        # no budget: same depth and peak either way
    assert chosen == "col"
    with pytest.raises(ValueError):
        plan_in_order(fn, "auto", 10)


def test_cache_key_and_metadata(tmp_path):
    fn = per_row_and_per_col()[0]
    keys = {o: plan_key(fn, None, 120, None, True, o) for o in cusadi2.OUTPUT_ORDERS}
    assert len(set(keys.values())) == 3
    with pytest.raises(ValueError):
        plan_key(fn, None, 120, None, True, "diagonal")
    built = {o: cusadi2.compile(fn, cache_dir=str(tmp_path), max_live=120, block_size=None, output_order=o)
             for o in ("auto", "row")}
    assert built["auto"].key == keys["auto"] and built["row"].key == keys["row"]
    assert built["auto"].path != built["row"].path
    for o, plan in built.items():
        meta = json.load(open(plan.path[:-len(".bin")] + ".json"))
        assert meta["output_order"] == plan.output_order == "row"      # auto chose row (col does not fit)
    again = cusadi2.compile(fn, cache_dir=str(tmp_path), max_live=120, block_size=None, output_order="auto")
    assert not again.built and again.output_order == "row"
    with pytest.raises(ValueError):
        cusadi2.compile(fn, cache_dir=str(tmp_path), max_live=120, block_size=None, output_order="col")
