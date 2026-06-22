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


@dataclasses.dataclass
class ChunkBoundary:
    """Spill/reload data at a boundary between two consecutive chunks."""
    spill_src_slots: List[int]
    spill_buffer_pos: List[int]
    reload_buffer_pos: List[int]
    reload_dst_slots: List[int]


@dataclasses.dataclass
class ChunkInfo:
    """Metadata for one chunk in a tiled execution plan."""
    level_start: int
    level_end: int
    peak_live_slots: int


@dataclasses.dataclass
class TiledLevelPlan:
    """Level plan with shared-memory tiling for large functions.

    When n_chunks == 1, this is semantically identical to a LevelPlan:
    zero spills, zero reloads, identical performance.
    """
    function_name: str
    n_ssa_slots: int
    max_smem_slots: int
    n_inputs: int
    n_outputs: int
    nnz_in: List[int]
    nnz_out: List[int]

    constants: List[SSAConst]
    input_mappings: List[SSAInput]
    output_mappings: List[SSAOutput]
    compute_levels: List[Level]

    chunks: List[ChunkInfo]
    boundaries: List[ChunkBoundary]
    spill_buffer_stride: int
    sub_batch_spills: Dict[int, Tuple[List[int], List[int]]]

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

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
    def peak_live_slots(self) -> int:
        return max((c.peak_live_slots for c in self.chunks), default=0)

    @property
    def shared_mem_bytes(self) -> int:
        return self.peak_live_slots * 8

    @property
    def total_spills(self) -> int:
        return sum(len(b.spill_src_slots) for b in self.boundaries)

    @property
    def total_reloads(self) -> int:
        return sum(len(b.reload_dst_slots) for b in self.boundaries)


def _extract_pass1(f: casadi.Function):
    """Pass 1: parse CasADi instructions, assign topological levels, build SSA ID map.

    Returns (raw_constants, raw_inputs, raw_outputs, raw_compute, total_ssa,
             n_in, n_out, nnz_in, nnz_out).
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

    return (raw_constants, raw_inputs, raw_outputs, raw_compute,
            ssa_counter, n_in, n_out, nnz_in, nnz_out)


def _compute_liveness(raw_constants, raw_inputs, raw_compute, raw_outputs, total_ssa):
    """Compute per-SSA-value produced/last-read levels and free schedule.

    Returns (ssa_produced_level, ssa_last_read_level, free_at_level, levels_dict, max_level).
    """
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
    # Hoisted: max compute level is invariant across the output-mapping loop.
    # Recomputing it inside the loop turns a linear pass into N×M (was 142M
    # genexpr calls / 22 s for H1's 1.8k outputs × 78k instructions).
    if raw_compute:
        max_compute_level = max(d for _, _, _, _, d, _ in raw_compute)
        out_level = max_compute_level + 1
        for src_ssa, _, _ in raw_outputs:
            if src_ssa >= 0:
                if out_level > ssa_last_read_level[src_ssa]:
                    ssa_last_read_level[src_ssa] = out_level

    for s in range(total_ssa):
        if ssa_last_read_level[s] < ssa_produced_level[s]:
            ssa_last_read_level[s] = ssa_produced_level[s]

    levels_dict: Dict[int, List[Tuple[int, int, int, Tuple[int, ...], float]]] = {}
    for k, op, sid, dep_ssas, d, cv in raw_compute:
        levels_dict.setdefault(d, []).append((k, op, sid, dep_ssas, cv))

    max_level = max(levels_dict.keys()) if levels_dict else 0

    free_at_level: Dict[int, List[int]] = {}
    for s in range(total_ssa):
        free_level = ssa_last_read_level[s] + 1
        free_at_level.setdefault(free_level, []).append(s)

    return ssa_produced_level, ssa_last_read_level, free_at_level, levels_dict, max_level


def extract_level_plan(f: casadi.Function) -> LevelPlan:
    """Extract an SSA-renamed, level-decomposed plan from a CasADi Function.

    Two passes:
      1. Naive level assignment using CasADi's original slots (for topology).
      2. SSA renaming with slot recycling (for conflict-free parallel execution).
    """
    (raw_constants, raw_inputs, raw_outputs, raw_compute,
     total_ssa, n_in, n_out, nnz_in, nnz_out) = _extract_pass1(f)

    (ssa_produced_level, ssa_last_read_level, free_at_level,
     levels_dict, max_level) = _compute_liveness(
        raw_constants, raw_inputs, raw_compute, raw_outputs, total_ssa)

    # Allocate physical slots using a free-list, level by level.
    ssa_to_phys = [0] * total_ssa
    free_slots: List[int] = []
    next_phys = [0]

    def alloc_slot() -> int:
        if free_slots:
            return free_slots.pop()
        s = next_phys[0]
        next_phys[0] += 1
        return s

    # Level 0: allocate physical slots for constants and inputs.
    for sid, _, _ in raw_constants:
        ssa_to_phys[sid] = alloc_slot()
    for sid, _, _, _ in raw_inputs:
        ssa_to_phys[sid] = alloc_slot()

    peak_live = next_phys[0] - len(free_slots)

    for d in range(1, max_level + 1):
        for s in free_at_level.get(d, []):
            free_slots.append(ssa_to_phys[s])
        for k, op, sid, dep_ssas, cv in levels_dict.get(d, []):
            ssa_to_phys[sid] = alloc_slot()
        live = next_phys[0] - len(free_slots)
        if live > peak_live:
            peak_live = live

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


def extract_tiled_level_plan(
    f: casadi.Function,
    max_smem_slots: int,
) -> TiledLevelPlan:
    """Extract a tiled, SSA-renamed, level-decomposed plan.

    When peak liveness fits within max_smem_slots, produces a single-chunk plan
    identical to extract_level_plan(). When it doesn't, inserts chunk boundaries
    with spill-to-global / reload-from-global at each boundary.

    Two levels of tiling:
    - Inter-level: chunk boundaries between levels (spill all of smem, selective
      reload for the next level's operands).
    - Intra-level: wide levels whose width exceeds remaining smem after reload
      are split into sub-batches. Each sub-batch's outputs are spilled before
      the next sub-batch begins. Operands stay pinned throughout.
    """
    (raw_constants, raw_inputs, raw_outputs, raw_compute,
     total_ssa, n_in, n_out, nnz_in, nnz_out) = _extract_pass1(f)

    (ssa_produced_level, ssa_last_read_level, free_at_level,
     levels_dict, max_level) = _compute_liveness(
        raw_constants, raw_inputs, raw_compute, raw_outputs, total_ssa)

    # Precompute read sets: for each level, which SSA values do its nodes read?
    level_read_sets: Dict[int, set] = {}
    for d, writers in levels_dict.items():
        rs: set = set()
        for k, op, sid, dep_ssas, cv in writers:
            for ds in dep_ssas:
                if ds >= 0:
                    rs.add(ds)
        level_read_sets[d] = rs

    # -- Tiled slot allocation --
    ssa_to_phys = [0] * total_ssa
    ssa_to_spill: Dict[int, int] = {}   # SSA id → spill buffer position
    in_smem: set = set()                # SSA values with valid smem slots
    alive: set = set()                  # SSA values produced and not yet dead
    free_slots: List[int] = []
    next_phys = [0]

    def alloc_slot() -> int:
        if free_slots:
            return free_slots.pop()
        s = next_phys[0]
        next_phys[0] += 1
        return s

    def current_live() -> int:
        return next_phys[0] - len(free_slots)

    def reset_pool():
        free_slots.clear()
        next_phys[0] = 0

    # Level 0: constants + inputs (always in chunk 0)
    for sid, _, _ in raw_constants:
        ssa_to_phys[sid] = alloc_slot()
        in_smem.add(sid)
        alive.add(sid)
    for sid, _, _, _ in raw_inputs:
        ssa_to_phys[sid] = alloc_slot()
        in_smem.add(sid)
        alive.add(sid)

    const_phys = [ssa_to_phys[sid] for sid, _, _ in raw_constants]
    input_phys = [ssa_to_phys[sid] for sid, _, _, _ in raw_inputs]

    n_init = current_live()
    if n_init > max_smem_slots:
        raise ValueError(
            f"Constants + inputs alone require {n_init} slots, "
            f"exceeding budget of {max_smem_slots}")

    chunk_start_level = 1
    chunk_peak = n_init
    chunks: List[ChunkInfo] = []
    boundaries: List[ChunkBoundary] = []
    spill_pos_counter = 0
    compute_levels: List[Level] = []
    # Maps compute_levels index → (smem_slots, spill_positions) for
    # sub-batch spills that occur after processing that Level.
    sub_batch_spills: Dict[int, Tuple[List[int], List[int]]] = {}

    def _build_level(d: int, writers):
        """Build Level object from writers using current ssa_to_phys."""
        nodes = []
        for k, op, sid, dep_ssas, cv in writers:
            phys_inputs = tuple(ssa_to_phys[ds] for ds in dep_ssas if ds >= 0)
            nodes.append(SSANode(
                instr_idx=k, op_id=op, ssa_out=ssa_to_phys[sid],
                ssa_inputs=phys_inputs, level=d, const_value=cv,
            ))
        compute_levels.append(Level(depth=d, nodes=nodes))

    def _free_expired(d: int):
        """Free slots for SSA values whose last reader completed before level d."""
        for s in free_at_level.get(d, []):
            if s in alive:
                if s in in_smem:
                    free_slots.append(ssa_to_phys[s])
                    in_smem.discard(s)
                alive.discard(s)

    def _spill_all_smem():
        """Spill everything currently in shared memory to the spill buffer."""
        nonlocal spill_pos_counter
        smem_sids = sorted(in_smem)
        src_slots = [ssa_to_phys[s] for s in smem_sids]
        dst_positions = list(range(spill_pos_counter,
                                   spill_pos_counter + len(smem_sids)))
        for s, pos in zip(smem_sids, dst_positions):
            ssa_to_spill[s] = pos
        spill_pos_counter += len(smem_sids)
        in_smem.clear()
        return src_slots, dst_positions

    def _reload_selective(sids_to_reload):
        """Reload specific SSA values from spill buffer into shared memory."""
        reload_src = []
        reload_dst = []
        for s in sids_to_reload:
            p = alloc_slot()
            ssa_to_phys[s] = p
            in_smem.add(s)
            reload_src.append(ssa_to_spill[s])
            reload_dst.append(p)
        return reload_src, reload_dst

    def _insert_boundary(d: int):
        """Insert a chunk boundary: end current chunk, spill smem, reset pool."""
        nonlocal chunk_start_level, chunk_peak

        chunks.append(ChunkInfo(
            level_start=chunk_start_level,
            level_end=d,
            peak_live_slots=chunk_peak,
        ))

        spill_src, spill_dst = _spill_all_smem()
        reset_pool()

        # Selective reload: only what level d reads from the spill buffer.
        read_set = level_read_sets.get(d, set())
        to_reload = sorted(s for s in read_set if s in alive)

        reload_src, reload_dst = _reload_selective(to_reload)

        boundaries.append(ChunkBoundary(
            spill_src_slots=spill_src,
            spill_buffer_pos=spill_dst,
            reload_buffer_pos=reload_src,
            reload_dst_slots=reload_dst,
        ))

        chunk_start_level = d
        chunk_peak = current_live()

    def _process_level_normal(d: int):
        """Process a level that fits entirely in shared memory."""
        nonlocal chunk_peak
        _free_expired(d)
        writers = levels_dict.get(d, [])
        for k, op, sid, dep_ssas, cv in writers:
            ssa_to_phys[sid] = alloc_slot()
            in_smem.add(sid)
            alive.add(sid)
        live = current_live()
        if live > chunk_peak:
            chunk_peak = live
        _build_level(d, writers)

    def _process_wide_level(d: int, avail: int):
        """Process a level via sub-batching. avail = slots available for outputs."""
        nonlocal chunk_peak, spill_pos_counter
        writers = levels_dict.get(d, [])
        n_total = len(writers)

        for batch_start in range(0, n_total, avail):
            batch_writers = writers[batch_start:batch_start + avail]
            batch_sids = []

            # Allocate output slots for this sub-batch.
            for k, op, sid, dep_ssas, cv in batch_writers:
                ssa_to_phys[sid] = alloc_slot()
                in_smem.add(sid)
                alive.add(sid)
                batch_sids.append(sid)

            live = current_live()
            if live > chunk_peak:
                chunk_peak = live

            # Build Level object for this sub-batch.
            _build_level(d, batch_writers)

            # Spill this sub-batch's outputs, free their smem slots.
            sb_src = [ssa_to_phys[s] for s in batch_sids]
            sb_dst = list(range(spill_pos_counter,
                                spill_pos_counter + len(batch_sids)))
            for s, pos in zip(batch_sids, sb_dst):
                ssa_to_spill[s] = pos
                free_slots.append(ssa_to_phys[s])
                in_smem.discard(s)
            spill_pos_counter += len(batch_sids)

            sub_batch_spills[len(compute_levels) - 1] = (sb_src, sb_dst)

        # Free operands consumed at this level (last_read == d).
        _free_expired(d)

    # -- Main level-processing loop --
    for d in range(1, max_level + 1):
        freed_sids = free_at_level.get(d, [])
        new_writers = levels_dict.get(d, [])
        n_freed = sum(1 for s in freed_sids if s in alive and s in in_smem)
        n_alloc = len(new_writers)
        projected = current_live() - n_freed + n_alloc

        # Check if this level needs any operand that's in the spill buffer
        # (alive but not in shared memory). If so, we must insert a boundary
        # to reload it, even if the projected count would fit.
        read_set_d = level_read_sets.get(d, set())
        needs_spill_values = any(s in alive and s not in in_smem for s in read_set_d)

        if projected <= max_smem_slots and not needs_spill_values:
            _process_level_normal(d)
        else:
            # Insert chunk boundary with selective reload for level d.
            _insert_boundary(d)

            n_reloaded = current_live()
            avail = max_smem_slots - n_reloaded

            # Free values consumed at level d (that were reloaded).
            _free_expired(d)
            avail_after_free = max_smem_slots - current_live()

            if n_alloc <= avail_after_free:
                # Level fits after selective reload + freeing consumed operands.
                for k, op, sid, dep_ssas, cv in new_writers:
                    ssa_to_phys[sid] = alloc_slot()
                    in_smem.add(sid)
                    alive.add(sid)
                live = current_live()
                if live > chunk_peak:
                    chunk_peak = live
                _build_level(d, new_writers)
            else:
                # Wide level: sub-batch.
                _process_wide_level(d, avail_after_free)

    # Record the final compute chunk.
    chunks.append(ChunkInfo(
        level_start=chunk_start_level,
        level_end=max_level + 1,
        peak_live_slots=chunk_peak,
    ))

    # Ensure output values are in shared memory. If any are in the spill
    # buffer (produced in earlier chunks, not reloaded for the last chunk),
    # insert a final boundary to reload them.
    output_sids = set()
    for src_ssa, _, _ in raw_outputs:
        if src_ssa >= 0:
            output_sids.add(src_ssa)
    output_needs_reload = sorted(s for s in output_sids if s in alive and s not in in_smem)

    if output_needs_reload:
        # Spill current smem, reset, reload output values.
        spill_src, spill_dst = _spill_all_smem()
        reset_pool()
        reload_src, reload_dst = _reload_selective(output_needs_reload)
        # Also reload any output values that were already in smem (now spilled).
        remaining_outputs = sorted(s for s in output_sids if s in alive and s not in in_smem)
        if remaining_outputs:
            rs2, rd2 = _reload_selective(remaining_outputs)
            reload_src.extend(rs2)
            reload_dst.extend(rd2)

        boundaries.append(ChunkBoundary(
            spill_src_slots=spill_src,
            spill_buffer_pos=spill_dst,
            reload_buffer_pos=reload_src,
            reload_dst_slots=reload_dst,
        ))
        chunks.append(ChunkInfo(
            level_start=max_level + 1,
            level_end=max_level + 1,
            peak_live_slots=current_live(),
        ))

    for s in free_at_level.get(max_level + 1, []):
        if s in alive:
            if s in in_smem:
                free_slots.append(ssa_to_phys[s])
                in_smem.discard(s)
            alive.discard(s)

    n_phys_slots = max(
        (c.peak_live_slots for c in chunks), default=0)

    constants = [
        SSAConst(ssa_slot=const_phys[i], value=val)
        for i, (sid, _, val) in enumerate(raw_constants)
    ]
    input_mappings = [
        SSAInput(ssa_slot=input_phys[i], input_arg=arg, element_idx=elem)
        for i, (sid, arg, elem, _) in enumerate(raw_inputs)
    ]
    output_mappings = [
        SSAOutput(ssa_slot=ssa_to_phys[src_ssa], output_arg=oarg, element_idx=oelem)
        for src_ssa, oarg, oelem in raw_outputs
        if src_ssa >= 0
    ]

    return TiledLevelPlan(
        function_name=f.name(),
        n_ssa_slots=n_phys_slots,
        max_smem_slots=max_smem_slots,
        n_inputs=n_in,
        n_outputs=n_out,
        nnz_in=nnz_in,
        nnz_out=nnz_out,
        constants=constants,
        input_mappings=input_mappings,
        output_mappings=output_mappings,
        compute_levels=compute_levels,
        chunks=chunks,
        boundaries=boundaries,
        spill_buffer_stride=spill_pos_counter,
        sub_batch_spills=sub_batch_spills,
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


def verify_tiled_level_plan(
    f: casadi.Function,
    plan: TiledLevelPlan,
    n_tests: int = 5,
    atol: float = 1e-10,
) -> bool:
    """Verify a TiledLevelPlan by simulating chunked execution with spill/reload.

    Simulates:
    - Chunk boundaries: spill smem → reset → selective reload
    - Normal levels: snapshot-read-write within smem
    - Sub-batched levels: process sub-batch → spill outputs → free output slots
    """
    np.random.seed(42)
    all_pass = True

    # Build ordered list of (compute_levels_index, depth) for levels in each chunk.
    # A sub-batched level produces multiple consecutive Level objects with the
    # same depth.
    level_indices_for_chunk: List[List[int]] = []
    cl_idx = 0
    for chunk in plan.chunks:
        indices = []
        while cl_idx < len(plan.compute_levels):
            lev = plan.compute_levels[cl_idx]
            if lev.depth >= chunk.level_end:
                break
            if lev.depth >= chunk.level_start:
                indices.append(cl_idx)
            cl_idx += 1
        level_indices_for_chunk.append(indices)

    for test in range(n_tests):
        inputs_np = [np.random.randn(f.nnz_in(i)) for i in range(f.n_in())]

        smem_size = plan.n_ssa_slots
        work = np.full(smem_size, np.nan, dtype=np.float64)
        spill_buf = (np.full(plan.spill_buffer_stride, np.nan, dtype=np.float64)
                     if plan.spill_buffer_stride > 0
                     else np.empty(0, dtype=np.float64))

        for c in plan.constants:
            work[c.ssa_slot] = c.value
        for im in plan.input_mappings:
            work[im.ssa_slot] = inputs_np[im.input_arg][im.element_idx]

        for chunk_idx, chunk in enumerate(plan.chunks):
            if chunk_idx > 0:
                boundary = plan.boundaries[chunk_idx - 1]
                work = np.full(smem_size, np.nan, dtype=np.float64)
                for src_pos, dst_slot in zip(boundary.reload_buffer_pos,
                                             boundary.reload_dst_slots):
                    work[dst_slot] = spill_buf[src_pos]

            for cl_idx in level_indices_for_chunk[chunk_idx]:
                level = plan.compute_levels[cl_idx]

                snapshot = work.copy()

                write_slots = [node.ssa_out for node in level.nodes]
                if len(write_slots) != len(set(write_slots)):
                    from collections import Counter
                    dupes = {s: cnt for s, cnt in Counter(write_slots).items() if cnt > 1}
                    print(f"  WRITE-WRITE CONFLICT chunk={chunk_idx} "
                          f"cl_idx={cl_idx}: {dupes}")
                    all_pass = False

                results = []
                for node in level.nodes:
                    a = snapshot[node.ssa_inputs[0]] if len(node.ssa_inputs) >= 1 else 0.0
                    b = snapshot[node.ssa_inputs[1]] if len(node.ssa_inputs) >= 2 else 0.0
                    val = _exec_op(node.op_id, a, b)
                    results.append((node.ssa_out, val))

                for slot, val in results:
                    work[slot] = val

                # Sub-batch spill: move outputs to spill buffer, clear smem slots.
                if cl_idx in plan.sub_batch_spills:
                    sb_slots, sb_positions = plan.sub_batch_spills[cl_idx]
                    for slot, pos in zip(sb_slots, sb_positions):
                        spill_buf[pos] = work[slot]
                        work[slot] = np.nan

            # Inter-chunk spill (skip for last chunk).
            if chunk_idx < len(plan.chunks) - 1:
                boundary = plan.boundaries[chunk_idx]
                for src_slot, dst_pos in zip(boundary.spill_src_slots,
                                             boundary.spill_buffer_pos):
                    spill_buf[dst_pos] = work[src_slot]

        plan_outputs = []
        for i in range(plan.n_outputs):
            out_arr = np.zeros(plan.nnz_out[i], dtype=np.float64)
            for om in plan.output_mappings:
                if om.output_arg == i:
                    out_arr[om.element_idx] = work[om.ssa_slot]
            plan_outputs.append(out_arr)

        casadi_outputs = f.call(inputs_np)
        for i in range(f.n_out()):
            ref = np.asarray(casadi_outputs[i].nonzeros(), dtype=np.float64)
            err = np.max(np.abs(plan_outputs[i] - ref))
            if err > atol:
                print(f"  MISMATCH test={test} out[{i}]: max|err|={err:.2e}")
                all_pass = False

    return all_pass
