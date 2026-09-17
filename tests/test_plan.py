"""Host-side tests: the level plan is correct under simulated parallel
execution, the serial program is bit-exact sequentially, and the plan file
round-trips."""
import casadi as ca
import pytest

from cusadi2 import (extract_level_plan, extract_serial_program, read_plan, verify_plan,
                     verify_serial_program, write_plan, write_unrolled_source)
from cusadi2.dag import IO_OPS, _demo_function, validate_specs
from cusadi2.ops import SUPPORTED_OPS


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


def late_ops_function():
    """The scalar ops CasADi added after OP_OUTPUT (ids above 46) that libm
    evaluates exactly as CasADi does: log1p, expm1, hypot, remainder. The kernel
    must dispatch these as compute, not IO."""
    x = ca.SX.sym("x", 4)
    y = ca.SX.sym("y", 4)
    z = ca.log1p(x * x) + ca.expm1(y) + ca.hypot(x, y) + ca.remainder(x, 0.7)
    return ca.Function("late_ops", [x, y], [z, ca.sum1(z)])


def erfinv_function():
    """OP_ERFINV (id 86). CasADi evaluates it with a rational approximation good
    to about 1e-11; the CUDA library and the host verifier are exact, so this
    one is checked to a tolerance rather than bit-exact."""
    x = ca.SX.sym("x", 4)
    return ca.Function("erfinv", [x], [ca.erfinv(0.5 * x)])


ALL_FUNCTIONS = [_demo_function, dyn_function, wide_function, late_ops_function]


@pytest.mark.parametrize("make", ALL_FUNCTIONS)
def test_plan_matches_casadi_under_parallel_execution(make):
    f = make()
    plan = extract_level_plan(f)
    assert verify_plan(f, plan) < 1e-12


@pytest.mark.parametrize("make", ALL_FUNCTIONS)
def test_serial_program_is_casadi_stream_and_bit_exact(make):
    f = make()
    prog = extract_serial_program(f)
    assert prog.n_instr == f.n_instructions() and prog.sz_w == f.sz_w()
    assert all(op in IO_OPS or op in SUPPORTED_OPS for op in prog.ops)
    assert verify_serial_program(f, prog) == 0.0


def test_erfinv_matches_casadi_to_its_approximation():
    f = erfinv_function()
    assert verify_plan(f, extract_level_plan(f)) < 1e-9
    assert verify_serial_program(f, extract_serial_program(f)) < 1e-9


def test_unsupported_op_is_rejected_at_extraction():
    x, y = ca.SX.sym("x"), ca.SX.sym("y")
    f = ca.Function("pm", [x, y], [x.printme(y)])          # OP_PRINTME reaches an SX stream
    with pytest.raises(ValueError, match="OP_PRINTME"):
        extract_level_plan(f)
    with pytest.raises(ValueError, match="OP_PRINTME"):
        extract_serial_program(f)


def test_level_plan_renames_io_but_serial_program_cannot():
    # CasADi reuses IO slots, so the stream keeps its IO moves inline; the level
    # plan preloads constants and schedules INPUT / OUTPUT as nodes on fresh
    # slots. Both must be full-coverage descriptions of `f`.
    f = wide_function()
    plan, prog = extract_level_plan(f), extract_serial_program(f)
    assert ca.OP_CONST not in plan.node_ops
    assert sum(op in IO_OPS for op in prog.ops) == len(plan.const_slots) + sum(op in IO_OPS for op in plan.node_ops)
    assert prog.n_instr - len(plan.const_slots) == plan.n_nodes
    assert prog.sz_w < plan.peak_live_slots


@pytest.mark.parametrize("chunk", [200000, 5])
def test_unrolled_source_has_one_statement_per_instruction(tmp_path, chunk):
    f = wide_function()
    path = write_unrolled_source(f, str(tmp_path / "wide.cu"), n_stages=3, chunk=chunk,
                                 split_above=chunk)
    src = open(path).read()
    body = [ln.strip() for ln in src.splitlines() if ln.startswith("    w[") or ln.startswith("    outputs[")]
    assert len(body) == f.n_instructions()
    assert src.count("_p0") >= 1 and (chunk == 5) == ("__noinline__" in src)
    assert f"SZ_W = {f.sz_w()}" in src and "N_STAGES = 3" in src


def test_slot_recycling_bounds_shared_memory():
    plan = extract_level_plan(wide_function())
    assert plan.peak_live_slots < plan.n_nodes


def test_max_live_windows_levels_to_fit():
    f = wide_function()
    asap = extract_level_plan(f)
    plan = extract_level_plan(f, max_live=asap.peak_live_slots // 2)
    assert plan.peak_live_slots <= asap.peak_live_slots // 2
    assert plan.n_levels > asap.n_levels and plan.n_nodes == asap.n_nodes
    assert verify_plan(f, plan) < 1e-12
    with pytest.raises(ValueError):
        extract_level_plan(f, max_live=1)


def test_plan_file_roundtrip(tmp_path):
    f = dyn_function()
    plan, prog = extract_level_plan(f), extract_serial_program(f)
    path = write_plan(f, str(tmp_path / "dyn.plan"), n_stages=4,
                      in_mode=["coupled", "shared"], in_offset=[0, 0], in_span=[1, 0])
    d = read_plan(path)
    assert d["name"] == "dyn" and d["n_stages"] == 4
    assert d["block_size"] == 0                  # chosen by the runtime at upload
    assert d["peak_live_slots"] == plan.peak_live_slots
    assert [a["count"] for a in d["inputs"]] == [5, 1]
    assert [a["mode"] for a in d["inputs"]] == ["coupled", "shared"]
    assert [a["count"] for a in d["outputs"]] == [4, 4]
    assert d["inputs"][0]["rowidx"] == [0, 1, 2] and d["inputs"][0]["colptr"] == [0, 3]
    for k in ("const_slots", "const_vals", "level_offsets", "level_widths", "node_ops",
              "node_outs", "node_in_a", "node_in_b"):
        assert d[k] == getattr(plan, k), k
    assert d["sz_w"] == prog.sz_w
    for k in ("ops", "outs", "in_a", "in_b", "const_vals"):
        assert d["serial_" + k] == getattr(prog, k), k


def test_validate_specs_rejects_bad_coupling():
    f = dyn_function()
    with pytest.raises(ValueError):
        validate_specs(f, 4, ["coupled", "stage"], [1, 0], [0, 0], ["stage", "stage"])
    with pytest.raises(ValueError):
        validate_specs(f, 4, ["stage"], [0], [0], ["stage", "stage"])
    with pytest.raises(ValueError):
        validate_specs(f, 4, ["stage", "stage"], [0, 0], [0, 0], ["shared", "stage"])
