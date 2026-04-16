"""
DAG extraction with level-aware SSA slot recycling (Option C).

Produces a LevelPlan where every compute node's output writes to a unique
SSA-renamed slot. Slots are recycled at level boundaries once their last
reader completes. This eliminates both write-write and read-write conflicts
within a level, enabling safe parallel execution with __syncthreads() only
between levels.

The naive topological level assignment (longest-path from inputs) is used —
it preserves the maximum available parallelism. Slot conflicts from CasADi's
work-vector reuse are resolved entirely through SSA renaming, not by
inflating the critical path.
"""

import dataclasses
import math
from typing import Dict, List, Tuple

import casadi
import numpy as np
from casadi import (OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ, OP_ASSIGN,
                    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_NEG,
                    OP_EXP, OP_LOG, OP_POW, OP_CONSTPOW, OP_SQRT, OP_TWICE,
                    OP_SIN, OP_COS, OP_TAN, OP_ASIN, OP_ACOS, OP_ATAN, OP_ATAN2,
                    OP_SINH, OP_COSH, OP_TANH, OP_ASINH, OP_ACOSH, OP_ATANH,
                    OP_LT, OP_LE, OP_EQ, OP_NE, OP_NOT, OP_AND, OP_OR,
                    OP_FLOOR, OP_CEIL, OP_FMOD, OP_FABS, OP_SIGN, OP_COPYSIGN,
                    OP_IF_ELSE_ZERO, OP_ERF, OP_FMIN, OP_FMAX, OP_INV)


@dataclasses.dataclass
class SSANode:
    """One compute operation in the SSA-renamed DAG."""
    instr_idx: int
    op_id: int
    ssa_out: int
    ssa_inputs: Tuple[int, ...]
    level: int
    const_value: float = 0.0


@dataclasses.dataclass
class SSAConst:
    """A constant loaded into an SSA slot at level 0."""
    ssa_slot: int
    value: float


@dataclasses.dataclass
class SSAInput:
    """An external input mapped to an SSA slot at level 0."""
    ssa_slot: int
    input_arg: int
    element_idx: int


@dataclasses.dataclass
class SSAOutput:
    """An SSA slot mapped to an external output."""
    ssa_slot: int
    output_arg: int
    element_idx: int


@dataclasses.dataclass
class Level:
    """One topological level of compute nodes."""
    depth: int
    nodes: List[SSANode]

    @property
    def width(self) -> int:
        return len(self.nodes)


@dataclasses.dataclass
class LevelPlan:
    """Complete level-decomposed, SSA-renamed execution plan."""
    function_name: str
    n_ssa_slots: int
    peak_live_slots: int
    n_inputs: int
    n_outputs: int
    nnz_in: List[int]
    nnz_out: List[int]

    constants: List[SSAConst]
    input_mappings: List[SSAInput]
    output_mappings: List[SSAOutput]
    compute_levels: List[Level]

    @property
    def n_compute_levels(self) -> int:
        return len(self.compute_levels)

    @property
    def total_compute_ops(self) -> int:
        return sum(lev.width for lev in self.compute_levels)

    @property
    def critical_path(self) -> int:
        return self.n_compute_levels

    @property
    def max_width(self) -> int:
        return max((lev.width for lev in self.compute_levels), default=0)

    @property
    def median_width(self) -> int:
        if not self.compute_levels:
            return 0
        widths = sorted(lev.width for lev in self.compute_levels)
        return widths[len(widths) // 2]

    @property
    def shared_mem_bytes(self) -> int:
        return self.peak_live_slots * 8


def extract_level_plan(f: casadi.Function) -> LevelPlan:
    """Extract an SSA-renamed, level-decomposed plan from a CasADi Function.

    Two passes:
      1. Naive level assignment using CasADi's original slots (for topology).
      2. SSA renaming with slot recycling (for conflict-free parallel execution).
    """
    n_instr = f.n_instructions()
    n_in = f.n_in()
    n_out = f.n_out()
    nnz_in = [f.nnz_in(i) for i in range(n_in)]
    nnz_out = [f.nnz_out(i) for i in range(n_out)]

    input_idx = []
    input_idx_lengths = [0]
    output_idx = []
    output_idx_lengths = [0]
    for i in range(n_instr):
        input_idx.extend(f.instruction_input(i))
        input_idx_lengths.append(len(f.instruction_input(i)))
        output_idx.extend(f.instruction_output(i))
        output_idx_lengths.append(len(f.instruction_output(i)))
    operations = [f.instruction_id(i) for i in range(n_instr)]
    const_values = [f.instruction_constant(i) for i in range(n_instr)]

    # ---- Pass 1: Naive level assignment on CasADi's original slots ----
    # Also build SSA node list referencing original slot indices.
    slot_depth: Dict[int, int] = {}
    slot_to_ssa: Dict[int, int] = {}
    ssa_counter = 0

    raw_constants: List[Tuple[int, int, float]] = []
    raw_inputs: List[Tuple[int, int, int, int]] = []
    raw_outputs: List[Tuple[int, int, int]] = []
    raw_compute: List[Tuple[int, int, int, Tuple[int, ...], int, float]] = []

    i_ptr = 0
    o_ptr = 0
    for k in range(n_instr):
        op = operations[k]
        n_inp = input_idx_lengths[k + 1]
        n_outp = output_idx_lengths[k + 1]
        inp = tuple(input_idx[i_ptr:i_ptr + n_inp])
        out = tuple(output_idx[o_ptr:o_ptr + n_outp])

        if op == OP_CONST:
            work_slot = out[0]
            sid = ssa_counter; ssa_counter += 1
            slot_to_ssa[work_slot] = sid
            slot_depth[work_slot] = 0
            raw_constants.append((sid, work_slot, const_values[k]))

        elif op == OP_INPUT:
            work_slot = out[0]
            sid = ssa_counter; ssa_counter += 1
            slot_to_ssa[work_slot] = sid
            slot_depth[work_slot] = 0
            raw_inputs.append((sid, inp[0], inp[1], work_slot))

        elif op == OP_OUTPUT:
            src_slot = inp[0]
            src_ssa = slot_to_ssa.get(src_slot, -1)
            raw_outputs.append((src_ssa, out[0], out[1]))

        else:
            if op == OP_SQ:
                dep_slots = (inp[0],)
            elif n_inp >= 2:
                dep_slots = (inp[0], inp[1])
            elif n_inp >= 1:
                dep_slots = (inp[0],)
            else:
                dep_slots = ()

            dep_ssas = tuple(slot_to_ssa.get(s, -1) for s in dep_slots)
            max_d = max((slot_depth.get(s, 0) for s in dep_slots), default=0)
            d = max_d + 1

            work_slot = out[0] if out else -1
            sid = ssa_counter; ssa_counter += 1
            if work_slot >= 0:
                slot_to_ssa[work_slot] = sid
                slot_depth[work_slot] = d

            raw_compute.append((k, op, sid, dep_ssas, d, const_values[k]))

        i_ptr += n_inp
        o_ptr += n_outp

    total_ssa = ssa_counter

    # ---- Pass 2: SSA slot recycling ----
    # Compute last-read level for each SSA value.
    ssa_produced_level = [0] * total_ssa
    ssa_last_read_level = [0] * total_ssa

    for sid, _, _ in raw_constants:
        ssa_produced_level[sid] = 0
    for sid, _, _, _ in raw_inputs:
        ssa_produced_level[sid] = 0
    for k, op, sid, dep_ssas, d, cv in raw_compute:
        ssa_produced_level[sid] = d
        for ds in dep_ssas:
            if ds >= 0 and d > ssa_last_read_level[ds]:
                ssa_last_read_level[ds] = d
    for src_ssa, _, _ in raw_outputs:
        if src_ssa >= 0:
            max_compute_level = max((d for _, _, _, _, d, _ in raw_compute), default=0)
            out_level = max_compute_level + 1
            if out_level > ssa_last_read_level[src_ssa]:
                ssa_last_read_level[src_ssa] = out_level

    # Ensure produced values that are never read still get freed
    for s in range(total_ssa):
        if ssa_last_read_level[s] < ssa_produced_level[s]:
            ssa_last_read_level[s] = ssa_produced_level[s]

    # Group compute ops by level
    levels_dict: Dict[int, List[Tuple[int, int, int, Tuple[int, ...], float]]] = {}
    for k, op, sid, dep_ssas, d, cv in raw_compute:
        levels_dict.setdefault(d, []).append((k, op, sid, dep_ssas, cv))

    max_level = max(levels_dict.keys()) if levels_dict else 0

    # Allocate physical slots using a free-list, level by level.
    # A slot becomes free at the START of the level after its last reader.
    ssa_to_phys = [0] * total_ssa
    free_slots: List[int] = []
    next_phys = [0]

    def alloc_slot() -> int:
        if free_slots:
            return free_slots.pop()
        s = next_phys[0]
        next_phys[0] += 1
        return s

    # Track which SSA values are freed at the start of each level.
    free_at_level: Dict[int, List[int]] = {}
    for s in range(total_ssa):
        free_level = ssa_last_read_level[s] + 1
        free_at_level.setdefault(free_level, []).append(s)

    # Level 0: allocate physical slots for constants and inputs.
    # Free any SSA values whose last reader is at level 0 (read before level 1).
    for sid, _, _ in raw_constants:
        p = alloc_slot()
        ssa_to_phys[sid] = p
    for sid, _, _, _ in raw_inputs:
        p = alloc_slot()
        ssa_to_phys[sid] = p

    peak_live = next_phys[0] - len(free_slots)

    # Process compute levels.
    for d in range(1, max_level + 1):
        # Free slots from SSA values whose last reader completed at level d-1.
        for s in free_at_level.get(d, []):
            free_slots.append(ssa_to_phys[s])

        # Allocate slots for this level's writers.
        for k, op, sid, dep_ssas, cv in levels_dict.get(d, []):
            p = alloc_slot()
            ssa_to_phys[sid] = p

        live = next_phys[0] - len(free_slots)
        if live > peak_live:
            peak_live = live

    # Free remaining after last compute level (for output reads).
    for s in free_at_level.get(max_level + 1, []):
        free_slots.append(ssa_to_phys[s])

    n_phys_slots = next_phys[0]

    # ---- Build the final LevelPlan with physical slot references ----
    constants = [
        SSAConst(ssa_slot=ssa_to_phys[sid], value=val)
        for sid, _, val in raw_constants
    ]
    input_mappings = [
        SSAInput(ssa_slot=ssa_to_phys[sid], input_arg=arg, element_idx=elem)
        for sid, arg, elem, _ in raw_inputs
    ]
    output_mappings = [
        SSAOutput(ssa_slot=ssa_to_phys[src_ssa], output_arg=oarg, element_idx=oelem)
        for src_ssa, oarg, oelem in raw_outputs
        if src_ssa >= 0
    ]

    compute_levels = []
    for d in sorted(levels_dict.keys()):
        nodes = []
        for k, op, sid, dep_ssas, cv in levels_dict[d]:
            phys_inputs = tuple(ssa_to_phys[ds] for ds in dep_ssas if ds >= 0)
            nodes.append(SSANode(
                instr_idx=k,
                op_id=op,
                ssa_out=ssa_to_phys[sid],
                ssa_inputs=phys_inputs,
                level=d,
                const_value=cv,
            ))
        compute_levels.append(Level(depth=d, nodes=nodes))

    return LevelPlan(
        function_name=f.name(),
        n_ssa_slots=n_phys_slots,
        peak_live_slots=peak_live,
        n_inputs=n_in,
        n_outputs=n_out,
        nnz_in=nnz_in,
        nnz_out=nnz_out,
        constants=constants,
        input_mappings=input_mappings,
        output_mappings=output_mappings,
        compute_levels=compute_levels,
    )


# ---------------------------------------------------------------------------
# Op interpreter for verification
# ---------------------------------------------------------------------------

def _exec_op(op_id: int, a: float, b: float) -> float:
    if op_id == OP_ASSIGN: return a
    if op_id == OP_ADD: return a + b
    if op_id == OP_SUB: return a - b
    if op_id == OP_MUL: return a * b
    if op_id == OP_DIV: return a / b
    if op_id == OP_NEG: return -a
    if op_id == OP_EXP: return math.exp(a)
    if op_id == OP_LOG: return math.log(a) if a > 0 else float('-inf')
    if op_id == OP_POW: return a ** b
    if op_id == OP_CONSTPOW: return a ** b
    if op_id == OP_SQRT: return math.sqrt(a) if a >= 0 else float('nan')
    if op_id == OP_SQ: return a * a
    if op_id == OP_TWICE: return 2.0 * a
    if op_id == OP_SIN: return math.sin(a)
    if op_id == OP_COS: return math.cos(a)
    if op_id == OP_TAN: return math.tan(a)
    if op_id == OP_ASIN: return math.asin(a) if -1 <= a <= 1 else float('nan')
    if op_id == OP_ACOS: return math.acos(a) if -1 <= a <= 1 else float('nan')
    if op_id == OP_ATAN: return math.atan(a)
    if op_id == OP_ATAN2: return math.atan2(a, b)
    if op_id == OP_SINH: return math.sinh(a)
    if op_id == OP_COSH: return math.cosh(a)
    if op_id == OP_TANH: return math.tanh(a)
    if op_id == OP_ASINH: return math.asinh(a)
    if op_id == OP_ACOSH: return math.acosh(a) if a >= 1 else float('nan')
    if op_id == OP_ATANH: return math.atanh(a) if -1 < a < 1 else float('nan')
    if op_id == OP_LT: return float(a < b)
    if op_id == OP_LE: return float(a <= b)
    if op_id == OP_EQ: return float(a == b)
    if op_id == OP_NE: return float(a != b)
    if op_id == OP_NOT: return float(not a)
    if op_id == OP_AND: return float(bool(a) and bool(b))
    if op_id == OP_OR: return float(bool(a) or bool(b))
    if op_id == OP_FLOOR: return math.floor(a)
    if op_id == OP_CEIL: return math.ceil(a)
    if op_id == OP_FMOD: return math.fmod(a, b)
    if op_id == OP_FABS: return abs(a)
    if op_id == OP_SIGN: return (1.0 if a > 0 else (-1.0 if a < 0 else 0.0))
    if op_id == OP_COPYSIGN: return math.copysign(a, b)
    if op_id == OP_IF_ELSE_ZERO: return 0.0 if a == 0.0 else b
    if op_id == OP_ERF: return math.erf(a)
    if op_id == OP_FMIN: return min(a, b)
    if op_id == OP_FMAX: return max(a, b)
    if op_id == OP_INV: return 1.0 / a
    raise ValueError(f"Unknown op_id {op_id}")


def verify_level_plan_parallel(
    f: casadi.Function,
    plan: LevelPlan,
    n_tests: int = 5,
    atol: float = 1e-10,
) -> bool:
    """Verify a LevelPlan by simulating parallel execution on renamed SSA slots.

    For each level, ALL nodes execute simultaneously: they read from a snapshot
    of the work vector taken BEFORE any writes at this level, then all writes
    are applied. This catches intra-level read-write and write-write conflicts
    that sequential verification would miss.
    """
    np.random.seed(42)
    all_pass = True

    for test in range(n_tests):
        inputs_np = [np.random.randn(f.nnz_in(i)) for i in range(f.n_in())]

        work = np.full(plan.n_ssa_slots, np.nan, dtype=np.float64)

        for c in plan.constants:
            work[c.ssa_slot] = c.value

        for im in plan.input_mappings:
            work[im.ssa_slot] = inputs_np[im.input_arg][im.element_idx]

        for level in plan.compute_levels:
            # Snapshot: all reads at this level see the pre-write state.
            snapshot = work.copy()

            # Detect write-write conflicts (should not exist after SSA rename).
            write_slots = [node.ssa_out for node in level.nodes]
            if len(write_slots) != len(set(write_slots)):
                from collections import Counter
                dupes = {s: c for s, c in Counter(write_slots).items() if c > 1}
                print(f"  WRITE-WRITE CONFLICT at level {level.depth}: {dupes}")
                all_pass = False

            # Execute all nodes "simultaneously" using snapshot for reads.
            results = []
            for node in level.nodes:
                a = snapshot[node.ssa_inputs[0]] if len(node.ssa_inputs) >= 1 else 0.0
                b = snapshot[node.ssa_inputs[1]] if len(node.ssa_inputs) >= 2 else 0.0
                val = _exec_op(node.op_id, a, b)
                results.append((node.ssa_out, val))

            # Apply all writes.
            for slot, val in results:
                work[slot] = val

        # Gather outputs.
        plan_outputs = []
        for i in range(plan.n_outputs):
            out_arr = np.zeros(plan.nnz_out[i], dtype=np.float64)
            for om in plan.output_mappings:
                if om.output_arg == i:
                    out_arr[om.element_idx] = work[om.ssa_slot]
            plan_outputs.append(out_arr)

        # Compare against CasADi oracle.
        casadi_outputs = f.call(inputs_np)
        for i in range(f.n_out()):
            ref = np.asarray(casadi_outputs[i].nonzeros(), dtype=np.float64)
            err = np.max(np.abs(plan_outputs[i] - ref))
            if err > atol:
                print(f"  MISMATCH test={test} out[{i}]: max|err|={err:.2e}")
                all_pass = False

    return all_pass
