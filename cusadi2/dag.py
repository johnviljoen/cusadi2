"""Level-parallel DAG scheduler.

Turns one CasADi function into the level-decomposed, SSA-renamed plan that the
function-independent kernel (``kernel_template.py``) interprets on device.

What ``LevelPlan`` holds, and how the kernel consumes it:

  * **node arrays** (``node_ops`` / ``node_outs`` / ``node_in_a`` / ``node_in_b``)
    plus the ``level_offsets`` / ``level_widths`` CSR index: every compute op
    flattened into parallel arrays, grouped into topological levels. The kernel
    runs one level at a time with one ``__syncthreads()`` between levels; within
    a level all nodes are independent. Node *i* is ``node_ops[i]`` writing slot
    ``node_outs[i]`` from slots ``node_in_a[i]`` / ``node_in_b[i]`` (unary ops
    ignore ``in_b``). Level *d* spans nodes
    ``[level_offsets[d], level_offsets[d] + level_widths[d])``.
  * **constants / input maps / output maps**: what to preload into the
    shared-memory work array before compute, and what to write back after.
  * **peak_live_slots**: the recycled-slot peak; the kernel's shared-memory
    budget is ``peak_live_slots * sizeof(scalar)``.

Two passes (``extract_level_plan``):
  1. Longest-path level assignment on CasADi's own work slots, which preserves
     the maximum available parallelism.
  2. SSA renaming with slot recycling: every node writes a fresh slot; freed
     slots are reused at level boundaries. This removes all write-write and
     read-write conflicts *within* a level, so the only barrier needed is the
     one between levels.

``verify_plan`` executes a plan on the host the way the kernel does (all reads
of a level before all writes) and compares against CasADi. It is the guard for
the conflict class of bug that a sequential interpreter would hide.
"""
import dataclasses
from typing import Dict, List, Tuple

import casadi
import numpy as np
from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ


@dataclasses.dataclass
class LevelPlan:
    """Level-decomposed, SSA-renamed execution plan for one function, as plain
    parallel lists (this is the wire format; ``plan.py`` writes it verbatim)."""
    function_name: str
    peak_live_slots: int
    n_inputs: int
    n_outputs: int
    nnz_in: List[int]
    nnz_out: List[int]

    const_slots: List[int]      # constants loaded into slots before compute
    const_vals: List[float]
    input_slots: List[int]      # external input element -> slot, before compute
    input_args: List[int]
    input_elems: List[int]
    out_slots: List[int]        # slot -> external output element, after compute
    out_args: List[int]
    out_elems: List[int]

    level_offsets: List[int]    # level d spans nodes [offsets[d], offsets[d] + widths[d])
    level_widths: List[int]
    node_ops: List[int]         # CasADi op id per node
    node_outs: List[int]        # slot each node writes
    node_in_a: List[int]        # first input slot (0 if none)
    node_in_b: List[int]        # second input slot (0 if unary)

    @property
    def n_levels(self) -> int:
        return len(self.level_offsets)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ops)


def _extract_pass1(f: casadi.Function):
    """Parse CasADi instructions, assign topological levels, map work slots to
    SSA ids. Returns (raw_constants, raw_inputs, raw_outputs, raw_compute, total_ssa)."""
    n_instr = f.n_instructions()
    slot_depth: Dict[int, int] = {}
    slot_to_ssa: Dict[int, int] = {}
    ssa = 0

    raw_constants: List[Tuple[int, float]] = []              # (sid, value)
    raw_inputs: List[Tuple[int, int, int]] = []              # (sid, arg, elem)
    raw_outputs: List[Tuple[int, int, int]] = []             # (src_sid, arg, elem)
    raw_compute: List[Tuple[int, int, Tuple[int, ...], int]] = []  # (op, sid, dep_sids, level)

    for k in range(n_instr):
        op = f.instruction_id(k)
        inp = tuple(f.instruction_input(k))
        out = tuple(f.instruction_output(k))

        if op == OP_CONST:
            slot_to_ssa[out[0]] = ssa
            slot_depth[out[0]] = 0
            raw_constants.append((ssa, f.instruction_constant(k)))
            ssa += 1
        elif op == OP_INPUT:
            slot_to_ssa[out[0]] = ssa
            slot_depth[out[0]] = 0
            raw_inputs.append((ssa, inp[0], inp[1]))
            ssa += 1
        elif op == OP_OUTPUT:
            raw_outputs.append((slot_to_ssa.get(inp[0], -1), out[0], out[1]))
        else:
            dep_slots = inp[:1] if op == OP_SQ else inp[:2]
            dep_ssas = tuple(slot_to_ssa.get(s, -1) for s in dep_slots)
            d = max((slot_depth.get(s, 0) for s in dep_slots), default=0) + 1
            if out:
                slot_to_ssa[out[0]] = ssa
                slot_depth[out[0]] = d
            raw_compute.append((op, ssa, dep_ssas, d))
            ssa += 1

    return raw_constants, raw_inputs, raw_outputs, raw_compute, ssa


def _compute_liveness(raw_constants, raw_inputs, raw_compute, raw_outputs, total_ssa):
    """Group compute nodes by level and compute the slot free schedule.

    Returns (free_at_level, levels, max_level): ``free_at_level[d]`` lists the
    SSA values whose slots become reusable at level d (last read at d-1);
    ``levels[d]`` holds the compute nodes at level d."""
    produced = [0] * total_ssa
    last_read = [0] * total_ssa
    for op, sid, dep_ssas, d in raw_compute:
        produced[sid] = d
        for ds in dep_ssas:
            if ds >= 0 and d > last_read[ds]:
                last_read[ds] = d
    if raw_compute:
        out_level = max(d for _, _, _, d in raw_compute) + 1
        for src, _, _ in raw_outputs:
            if src >= 0 and out_level > last_read[src]:
                last_read[src] = out_level
    for s in range(total_ssa):
        if last_read[s] < produced[s]:
            last_read[s] = produced[s]

    levels: Dict[int, List[Tuple[int, int, Tuple[int, ...]]]] = {}
    for op, sid, dep_ssas, d in raw_compute:
        levels.setdefault(d, []).append((op, sid, dep_ssas))
    max_level = max(levels) if levels else 0

    free_at_level: Dict[int, List[int]] = {}
    for s in range(total_ssa):
        free_at_level.setdefault(last_read[s] + 1, []).append(s)
    return free_at_level, levels, max_level


def extract_level_plan(f: casadi.Function) -> LevelPlan:
    """Extract an SSA-renamed, level-decomposed plan from a CasADi Function."""
    raw_constants, raw_inputs, raw_outputs, raw_compute, total_ssa = _extract_pass1(f)
    free_at_level, levels, max_level = _compute_liveness(
        raw_constants, raw_inputs, raw_compute, raw_outputs, total_ssa)

    # Allocate physical slots from a free-list, level by level. A slot is freed
    # one level after its last read, so the high-water mark of allocated slots
    # is the peak number simultaneously live.
    ssa_to_phys = [0] * total_ssa
    free_slots: List[int] = []
    next_phys = 0

    def alloc() -> int:
        nonlocal next_phys
        if free_slots:
            return free_slots.pop()
        next_phys += 1
        return next_phys - 1

    for sid, _ in raw_constants:
        ssa_to_phys[sid] = alloc()
    for sid, _, _ in raw_inputs:
        ssa_to_phys[sid] = alloc()
    peak_live = next_phys - len(free_slots)

    for d in range(1, max_level + 1):
        for s in free_at_level.get(d, []):
            free_slots.append(ssa_to_phys[s])
        for _, sid, _ in levels.get(d, []):
            ssa_to_phys[sid] = alloc()
        peak_live = max(peak_live, next_phys - len(free_slots))

    level_offsets, level_widths = [], []
    node_ops, node_outs, node_in_a, node_in_b = [], [], [], []
    for d in sorted(levels):
        level_offsets.append(len(node_ops))
        for op, sid, dep_ssas in levels[d]:
            phys = [ssa_to_phys[ds] for ds in dep_ssas if ds >= 0]
            node_ops.append(op)
            node_outs.append(ssa_to_phys[sid])
            node_in_a.append(phys[0] if phys else 0)
            node_in_b.append(phys[1] if len(phys) > 1 else 0)
        level_widths.append(len(node_ops) - level_offsets[-1])

    outs = [(ssa_to_phys[src], a, e) for src, a, e in raw_outputs if src >= 0]
    return LevelPlan(
        function_name=f.name(),
        peak_live_slots=peak_live,
        n_inputs=f.n_in(),
        n_outputs=f.n_out(),
        nnz_in=[f.nnz_in(i) for i in range(f.n_in())],
        nnz_out=[f.nnz_out(i) for i in range(f.n_out())],
        const_slots=[ssa_to_phys[sid] for sid, _ in raw_constants],
        const_vals=[float(v) for _, v in raw_constants],
        input_slots=[ssa_to_phys[sid] for sid, _, _ in raw_inputs],
        input_args=[a for _, a, _ in raw_inputs],
        input_elems=[e for _, _, e in raw_inputs],
        out_slots=[s for s, _, _ in outs],
        out_args=[a for _, a, _ in outs],
        out_elems=[e for _, _, e in outs],
        level_offsets=level_offsets,
        level_widths=level_widths,
        node_ops=node_ops,
        node_outs=node_outs,
        node_in_a=node_in_a,
        node_in_b=node_in_b,
    )


def pick_block_size(plan: LevelPlan) -> int:
    """Smallest power of two >= the plan's median level width, capped at 256.
    One block per (instance, stage); this covers the typical level in one pass."""
    widths = sorted(plan.level_widths)
    med = widths[len(widths) // 2] if widths else 0
    bs = 32
    while bs < med and bs < 256:
        bs *= 2
    return bs


def verify_plan(f: casadi.Function, plan: LevelPlan, n_samples: int = 3,
                lo: float = -1.0, up: float = 1.0, seed: int = 0) -> float:
    """Execute `plan` on the host as the kernel does (every read in a level before
    every write) at random inputs and return the max abs error against `f`.

    A sequential interpreter would hide intra-level slot conflicts; this one
    exposes them, which is the bug class SSA renaming exists to prevent."""
    from cusadi2.ops import OP_PY
    rng = np.random.default_rng(seed)
    w = [0.0] * plan.peak_live_slots
    err = 0.0
    for _ in range(n_samples):
        args = [rng.uniform(lo, up, n) for n in plan.nnz_in]
        for s, v in zip(plan.const_slots, plan.const_vals):
            w[s] = v
        for s, a, e in zip(plan.input_slots, plan.input_args, plan.input_elems):
            w[s] = float(args[a][e])
        for off, wid in zip(plan.level_offsets, plan.level_widths):
            rng_ = range(off, off + wid)
            reads = [(w[plan.node_in_a[i]], w[plan.node_in_b[i]]) for i in rng_]
            for i, (ra, rb) in zip(rng_, reads):
                w[plan.node_outs[i]] = OP_PY[plan.node_ops[i]](ra, rb)
        got = [np.zeros(n) for n in plan.nnz_out]
        for s, a, e in zip(plan.out_slots, plan.out_args, plan.out_elems):
            got[a][e] = w[s]
        ref = f(*[casadi.DM(x) for x in args])
        ref = ref if isinstance(ref, (list, tuple)) else [ref]
        for g, r in zip(got, ref):
            err = max(err, float(np.max(np.abs(g - np.array(casadi.DM(r).nonzeros())), initial=0.0)))
    return err


# ---------------------------------------------------------------------------
# Stage axis (horizon). The kernel runs one block per (instance, stage) over a
# (batch x n_stages) grid; each arg is indexed along the stage axis by a per-arg
# mode. Plain batch-only evaluation is every arg "stage" with n_stages == 1.
#   "stage"   - independent value per stage    (leading-axis length n_stages)
#   "shared"  - one value broadcast to stages  (leading-axis length 1)
#   "coupled" - stage k reads element k+offset (leading-axis length n_stages+span)
# ---------------------------------------------------------------------------
STAGE_MODES = ("stage", "shared", "coupled")   # index == kernel wire value


def stage_count(mode: str, span: int, n_stages: int) -> int:
    """Leading-axis length an arg's buffer needs for `n_stages`."""
    if mode == "shared":
        return 1
    if mode == "coupled":
        return n_stages + span
    return n_stages


def validate_specs(fn, n_stages, in_mode, in_offset, in_span, out_mode) -> None:
    """Raise if the stage specs (parallel lists) are inconsistent with `fn`, the
    horizon, or each other. Inputs may be any mode; outputs "stage" only."""
    if len(in_mode) != fn.n_in():
        raise ValueError(f"in_mode has {len(in_mode)} entries; fn has {fn.n_in()} inputs.")
    if len(out_mode) != fn.n_out():
        raise ValueError(f"out_mode has {len(out_mode)} entries; fn has {fn.n_out()} outputs.")
    if n_stages <= 0:
        raise ValueError(f"n_stages must be positive, got {n_stages}")
    for i, (m, off, span) in enumerate(zip(in_mode, in_offset, in_span)):
        if m not in STAGE_MODES:
            raise ValueError(f"input {i}: unknown stage mode {m!r}; expected one of {STAGE_MODES}")
        if m == "coupled":
            if off < 0 or (n_stages - 1) + off >= stage_count(m, span, n_stages):
                raise ValueError(
                    f"input {i}: coupled offset={off} span={span} reads outside "
                    f"[0, {stage_count(m, span, n_stages)}) for n_stages={n_stages}")
        elif off != 0 or span != 0:
            raise ValueError(f"input {i}: offset/span are only valid for coupled mode")
    for i, m in enumerate(out_mode):
        if m != "stage":
            raise ValueError(f'output {i}: only "stage" mode is supported on outputs.')


def _demo_function() -> casadi.Function:
    """Tiny function exercising every instruction category. `a` is read twice,
    so its slot stays live across a level; 2.0 becomes an OP_CONST."""
    x = casadi.SX.sym("x", 2)
    p = casadi.SX.sym("p")
    a = casadi.sin(x[0]) + p
    return casadi.Function("demo", [x, p], [casadi.vertcat(a * x[1], a + 2.0)])


def _print_plan(plan: LevelPlan) -> None:
    print(f"function: {plan.function_name}")
    print(f"  inputs={plan.n_inputs} nnz_in={plan.nnz_in}  outputs={plan.n_outputs} nnz_out={plan.nnz_out}")
    print(f"  levels={plan.n_levels} ops={plan.n_nodes} peak_live={plan.peak_live_slots} "
          f"block_size={pick_block_size(plan)}")
    print("  constants:", list(zip(plan.const_slots, plan.const_vals)))
    print("  inputs:   ", list(zip(plan.input_slots, plan.input_args, plan.input_elems)))
    print("  outputs:  ", list(zip(plan.out_slots, plan.out_args, plan.out_elems)))
    for d, (off, wid) in enumerate(zip(plan.level_offsets, plan.level_widths), start=1):
        print(f"  level {d} (width {wid}):")
        for i in range(off, off + wid):
            print(f"    op={plan.node_ops[i]:<3d} w[{plan.node_outs[i]}] "
                  f"<- [{plan.node_in_a[i]}, {plan.node_in_b[i]}]")


if __name__ == "__main__":
    f = _demo_function()
    plan = extract_level_plan(f)
    _print_plan(plan)
    print("max abs err vs casadi:", verify_plan(f, plan))
