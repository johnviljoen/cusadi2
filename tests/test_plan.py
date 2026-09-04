"""Host-side tests: plan extraction is correct under simulated parallel
execution, and the plan file round-trips."""
import casadi as ca
import pytest

from cusadi2 import extract_level_plan, read_plan, verify_plan, write_plan
from cusadi2.dag import _demo_function, pick_block_size, validate_specs


def dyn_function():
    u = ca.SX.sym("u", 3)
    v = ca.SX.sym("v", 2)
    return ca.Function("dyn", [u, v], [ca.tanh(u) + ca.vertcat(v, v[0]), ca.sum1(u)])


def wide_function():
    """Many independent chains, so levels are wide and slots get recycled."""
    x = ca.SX.sym("x", 64)
    y = ca.sin(x) * ca.cos(x[::-1]) + ca.exp(-x * x)
    for _ in range(5):
        y = y * ca.sin(y) + x
    return ca.Function("wide", [x], [y, ca.sum1(y)])


@pytest.mark.parametrize("make", [_demo_function, dyn_function, wide_function])
def test_plan_matches_casadi_under_parallel_execution(make):
    f = make()
    plan = extract_level_plan(f)
    assert verify_plan(f, plan) < 1e-12


def test_slot_recycling_bounds_shared_memory():
    plan = extract_level_plan(wide_function())
    assert plan.peak_live_slots < plan.n_nodes
    assert pick_block_size(plan) in (32, 64, 128, 256)


def test_plan_file_roundtrip(tmp_path):
    f = dyn_function()
    plan = extract_level_plan(f)
    path = write_plan(f, str(tmp_path / "dyn.plan"), n_stages=4,
                      in_mode=["coupled", "shared"], in_offset=[0, 0], in_span=[1, 0])
    d = read_plan(path)
    assert d["name"] == "dyn" and d["n_stages"] == 4
    assert d["block_size"] == pick_block_size(plan)
    assert d["peak_live_slots"] == plan.peak_live_slots
    assert [a["count"] for a in d["inputs"]] == [5, 1]
    assert [a["mode"] for a in d["inputs"]] == ["coupled", "shared"]
    assert [a["count"] for a in d["outputs"]] == [4, 4]
    assert d["inputs"][0]["rowidx"] == [0, 1, 2] and d["inputs"][0]["colptr"] == [0, 3]
    for k in ("const_slots", "const_vals", "input_slots", "input_args", "input_elems",
              "level_offsets", "level_widths", "node_ops", "node_outs", "node_in_a",
              "node_in_b", "out_slots", "out_args", "out_elems"):
        assert d[k] == getattr(plan, k), k


def test_validate_specs_rejects_bad_coupling():
    f = dyn_function()
    with pytest.raises(ValueError):
        validate_specs(f, 4, ["coupled", "stage"], [1, 0], [0, 0], ["stage", "stage"])
    with pytest.raises(ValueError):
        validate_specs(f, 4, ["stage"], [0], [0], ["stage", "stage"])
    with pytest.raises(ValueError):
        validate_specs(f, 4, ["stage", "stage"], [0, 0], [0, 0], ["shared", "stage"])
