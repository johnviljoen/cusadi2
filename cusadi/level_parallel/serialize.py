"""Backend-agnostic flattening of a LevelPlan into plain Python arrays.

This is the single source of truth for turning a `LevelPlan` (and optional
stage-axis specs) into the flat int/double arrays that every backend uploads to
the device. The torch (`pytorch/lp_function.py`) and JAX (`jax_ffi/function.py`)
backends build GPU buffers from these arrays; the native backend
(`cusadi/native/`) emits them as a C header. Keeping the flattening here means
the three backends cannot drift in slot/offset conventions.

Carries no torch / jax / casadi-evaluation dependency.
"""
from dataclasses import dataclass, field
from typing import List, Sequence

from cusadi.level_parallel.codegen import pick_block_size
from cusadi.level_parallel.dag import LevelPlan
from cusadi.level_parallel.stage import StageMode, StageSpec


@dataclass
class FlatPlan:
    """Per-stage (or whole-function) level-parallel plan as flat arrays.

    Counts the kernels need are `len(...)` of the corresponding array; the
    convenience properties expose them with the kernel's parameter names.
    """
    nnz_in: List[int]
    nnz_out: List[int]
    block_size: int
    peak_live_slots: int
    const_slots: List[int]
    const_vals: List[float]
    input_slots: List[int]
    input_args: List[int]
    input_elems: List[int]
    level_offsets: List[int]
    level_widths: List[int]
    node_ops: List[int]
    node_outs: List[int]
    node_in_a: List[int]
    node_in_b: List[int]
    out_slots: List[int]
    out_args: List[int]
    out_elems: List[int]

    @property
    def n_args_in(self) -> int: return len(self.nnz_in)
    @property
    def n_args_out(self) -> int: return len(self.nnz_out)
    @property
    def n_consts(self) -> int: return len(self.const_slots)
    @property
    def n_input_maps(self) -> int: return len(self.input_slots)
    @property
    def n_levels(self) -> int: return len(self.level_offsets)
    @property
    def n_nodes(self) -> int: return len(self.node_ops)
    @property
    def n_output_maps(self) -> int: return len(self.out_slots)
    @property
    def shared_bytes(self) -> int: return self.peak_live_slots * 8


def flatten_plan(plan: LevelPlan, block_size: int = None) -> FlatPlan:
    """Flatten a LevelPlan into FlatPlan. block_size defaults to pick_block_size."""
    ops, outs, in_a, in_b, loff, lwid = [], [], [], [], [], []
    off = 0
    for level in plan.compute_levels:
        loff.append(off)
        lwid.append(level.width)
        for nd in level.nodes:
            ops.append(nd.op_id)
            outs.append(nd.ssa_out)
            in_a.append(nd.ssa_inputs[0] if len(nd.ssa_inputs) >= 1 else 0)
            in_b.append(nd.ssa_inputs[1] if len(nd.ssa_inputs) >= 2 else 0)
        off += level.width
    return FlatPlan(
        nnz_in=list(plan.nnz_in),
        nnz_out=list(plan.nnz_out),
        block_size=block_size if block_size is not None else pick_block_size(plan),
        peak_live_slots=plan.peak_live_slots,
        const_slots=[c.ssa_slot for c in plan.constants],
        const_vals=[float(c.value) for c in plan.constants],
        input_slots=[m.ssa_slot for m in plan.input_mappings],
        input_args=[m.input_arg for m in plan.input_mappings],
        input_elems=[m.element_idx for m in plan.input_mappings],
        level_offsets=loff,
        level_widths=lwid,
        node_ops=ops,
        node_outs=outs,
        node_in_a=in_a,
        node_in_b=in_b,
        out_slots=[m.ssa_slot for m in plan.output_mappings],
        out_args=[m.output_arg for m in plan.output_mappings],
        out_elems=[m.element_idx for m in plan.output_mappings],
    )


@dataclass
class FlatStageSpecs:
    """Per-arg stage-axis metadata (mode/offset/span); counts derive at runtime
    from n_stages. mode: 0=STAGE, 1=SHARED, 2=COUPLED."""
    in_mode: List[int]
    in_offset: List[int]
    in_span: List[int]
    out_mode: List[int]

    def in_stage_count(self, arg: int, n_stages: int) -> int:
        m = self.in_mode[arg]
        if m == int(StageMode.SHARED):
            return 1
        if m == int(StageMode.COUPLED):
            return n_stages + self.in_span[arg]
        return n_stages

    def out_stage_count(self, arg: int, n_stages: int) -> int:
        return 1 if self.out_mode[arg] == int(StageMode.SHARED) else n_stages


def flatten_stage_specs(input_specs: Sequence[StageSpec],
                        output_specs: Sequence[StageSpec]) -> FlatStageSpecs:
    return FlatStageSpecs(
        in_mode=[int(s.mode) for s in input_specs],
        in_offset=[int(s.offset) for s in input_specs],
        in_span=[int(s.span) for s in input_specs],
        out_mode=[int(s.mode) for s in output_specs],
    )
