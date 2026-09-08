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
    INPUT and OUTPUT are nodes too, encoded as in the serial program (slot,
    argument, element): inputs load at the level under their window's floor,
    outputs store at the level after their window's deepest.
  * **constants**: preloaded into the shared-memory work array (a preamble).
  * **peak_live_slots**: the recycled-slot peak; the kernel's shared-memory
    budget is ``peak_live_slots * sizeof(scalar)``.

Two passes (``extract_level_plan``):
  1. Longest-path level assignment on CasADi's own work slots, which preserves
     the maximum available parallelism; optionally windowed (`window` /
     `max_live`) to trade levels for a smaller live set.
  2. SSA renaming with slot recycling: every node writes a fresh slot; freed
     slots are reused at level boundaries. This removes all write-write and
     read-write conflicts *within* a level, so the only barrier needed is the
     one between levels.

``verify_plan`` executes a plan on the host the way the kernel does (all reads
of a level before all writes) and compares against CasADi. It is the guard for
the conflict class of bug that a sequential interpreter would hide.

``SerialProgram`` (``extract_serial_program``) is the other program in a plan
file: CasADi's instruction stream verbatim, in instruction order on CasADi's
own work slots (``sz_w`` of them), with the CONST / INPUT / OUTPUT moves inline
where CasADi placed them. The serial kernel walks it one thread per item. The
IO moves cannot be renamed away in that stream: CasADi reuses a constant's or
input's slot after its last read and an output's source slot after the OUTPUT
instruction. ``verify_serial_program`` runs it sequentially on the host; it is
bit-exact against CasADi.
"""
import dataclasses
from typing import Dict, List, Tuple

import casadi
import numpy as np
from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ

IO_OPS = (OP_CONST, OP_INPUT, OP_OUTPUT)   # inline in SerialProgram; LevelPlan: CONST preamble, INPUT/OUTPUT nodes


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

    level_offsets: List[int]    # level d spans nodes [offsets[d], offsets[d] + widths[d])
    level_widths: List[int]
    node_ops: List[int]         # CasADi op id per node
    node_outs: List[int]        # slot each node writes (OP_OUTPUT: the slot it stores)
    node_in_a: List[int]        # first input slot (0 if none); OP_INPUT/OP_OUTPUT: argument
    node_in_b: List[int]        # second input slot (0 if unary); OP_INPUT/OP_OUTPUT: element

    @property
    def n_levels(self) -> int:
        return len(self.level_offsets)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ops)


@dataclasses.dataclass
class SerialProgram:
    """CasADi's instruction stream verbatim, for the serial kernel. Instruction
    i is ``ops[i]`` on CasADi's work slots. Compute ops write ``outs[i]`` from
    ``in_a[i]`` / ``in_b[i]``; OP_CONST writes ``outs[i]`` from
    ``const_vals[in_a[i]]``; OP_INPUT writes ``outs[i]`` from input arg
    ``in_a[i]`` element ``in_b[i]``; OP_OUTPUT reads ``outs[i]`` into output
    arg ``in_a[i]`` element ``in_b[i]``. The work vector has ``sz_w`` slots."""
    sz_w: int
    ops: List[int]
    outs: List[int]
    in_a: List[int]
    in_b: List[int]
    const_vals: List[float]

    @property
    def n_instr(self) -> int:
        return len(self.ops)


def extract_serial_program(f: casadi.Function) -> SerialProgram:
    """Copy `f`'s instruction stream, order and slots untouched."""
    ops, outs, in_a, in_b, consts = [], [], [], [], []
    for k in range(f.n_instructions()):
        op = f.instruction_id(k)
        inp = tuple(f.instruction_input(k))
        out = tuple(f.instruction_output(k))
        if op == OP_CONST:
            o, a, b = out[0], len(consts), 0
            consts.append(float(f.instruction_constant(k)))
        elif op == OP_INPUT:
            o, a, b = out[0], inp[0], inp[1]
        elif op == OP_OUTPUT:
            o, a, b = inp[0], out[0], out[1]
        else:
            o = out[0] if out else 0
            a = inp[0] if inp else 0
            b = inp[1] if (len(inp) > 1 and op != OP_SQ) else 0
        ops.append(op); outs.append(o); in_a.append(a); in_b.append(b)
    return SerialProgram(sz_w=f.sz_w(), ops=ops, outs=outs, in_a=in_a, in_b=in_b,
                         const_vals=consts)


def verify_serial_program(f: casadi.Function, prog: SerialProgram, n_samples: int = 3,
                          lo: float = -1.0, up: float = 1.0, seed: int = 0) -> float:
    """Run `prog` sequentially on the host at random inputs and return the max
    abs error against `f`. Same ops, same order, same slots: expect exactly 0."""
    from cusadi2.ops import OP_PY
    rng = np.random.default_rng(seed)
    w = [0.0] * prog.sz_w
    err = 0.0
    for _ in range(n_samples):
        args = [rng.uniform(lo, up, f.nnz_in(a)) for a in range(f.n_in())]
        got = [np.zeros(f.nnz_out(a)) for a in range(f.n_out())]
        for op, o, a, b in zip(prog.ops, prog.outs, prog.in_a, prog.in_b):
            if op == OP_CONST:
                w[o] = prog.const_vals[a]
            elif op == OP_INPUT:
                w[o] = float(args[a][b])
            elif op == OP_OUTPUT:
                got[a][b] = w[o]
            else:
                w[o] = OP_PY[op](w[a], w[b])
        ref = f(*[casadi.DM(x) for x in args])
        ref = ref if isinstance(ref, (list, tuple)) else [ref]
        for g, r in zip(got, ref):
            err = max(err, float(np.max(np.abs(g - np.array(casadi.DM(r).nonzeros())), initial=0.0)))
    return err


def _instructions(f: casadi.Function) -> List[Tuple[int, Tuple[int, ...], Tuple[int, ...], float]]:
    """CasADi's instruction stream as (op, inputs, outputs, constant) tuples."""
    return [(f.instruction_id(k), tuple(f.instruction_input(k)), tuple(f.instruction_output(k)),
             f.instruction_constant(k)) for k in range(f.n_instructions())]


def _extract_pass1(instrs, window: int = None):
    """Assign topological levels and map work slots to SSA ids. Returns
    (constants [(ssa, value)], nodes [(op, ssa, dep ssas, level, arg, elem)], n_ssa).
    Constants sit at depth 0 (a preamble); an INPUT loads at the level under the
    current floor and an OUTPUT stores at the level after its window's deepest,
    so without windows inputs form level 0 and outputs the last level (coalesced
    runs). With `window`, every that many compute instructions the pending
    outputs are stored and the floor moves below the deepest level so far, so
    consumers stay close to their producers (CasADi's order) and the live set
    stays near CasADi's `sz_w`, at the cost of levels."""
    slot_depth: Dict[int, int] = {}
    slot_to_ssa: Dict[int, int] = {}
    consts, nodes, pending = [], [], []
    ssa, floor, deepest, n_win = 0, 1, 0, 0

    def close_window():
        nonlocal floor, n_win
        nodes.extend((op, -1, deps, deepest + 1, a, b) for op, deps, a, b in pending)
        pending.clear()
        floor, n_win = deepest + 1, 0

    for op, inp, out, const in instrs:
        if op == OP_CONST:
            slot_to_ssa[out[0]], slot_depth[out[0]] = ssa, 0
            consts.append((ssa, const))
            ssa += 1
        elif op == OP_INPUT:
            slot_to_ssa[out[0]], slot_depth[out[0]] = ssa, floor - 1
            nodes.append((op, ssa, (), floor - 1, inp[0], inp[1]))
            ssa += 1
        elif op == OP_OUTPUT:
            if inp[0] in slot_to_ssa:
                pending.append((op, (slot_to_ssa[inp[0]],), out[0], out[1]))
        else:
            dep_slots = inp[:1] if op == OP_SQ else inp[:2]
            d = max(floor, max((slot_depth.get(s, 0) for s in dep_slots), default=0) + 1)
            nodes.append((op, ssa, tuple(slot_to_ssa.get(s, -1) for s in dep_slots), d, 0, 0))
            if out:
                slot_to_ssa[out[0]], slot_depth[out[0]] = ssa, d
            ssa += 1
            deepest, n_win = max(deepest, d), n_win + 1
            if n_win == window:
                close_window()
    close_window()
    return consts, nodes, ssa


def _compute_liveness(nodes, total_ssa):
    """Group nodes by level and compute the slot free schedule: ``free_at_level[d]``
    lists the SSA values whose slots become reusable at level d (last read at d-1)."""
    produced = [0] * total_ssa
    last_read = [0] * total_ssa
    for _, sid, dep_ssas, d, _, _ in nodes:
        if sid >= 0:
            produced[sid] = d
        for ds in dep_ssas:
            if ds >= 0 and d > last_read[ds]:
                last_read[ds] = d
    levels: Dict[int, list] = {}
    for n in nodes:
        levels.setdefault(n[3], []).append(n)
    for ns in levels.values():   # compute first, then IO by (arg, element): coalesced, no divergence
        ns.sort(key=lambda n: (n[0] in IO_OPS, n[4], n[5]) if n[0] in IO_OPS else (False, 0, 0))
    free_at_level: Dict[int, List[int]] = {}
    for s in range(total_ssa):
        free_at_level.setdefault(max(last_read[s], produced[s]) + 1, []).append(s)
    return free_at_level, levels


def extract_level_plan(f: casadi.Function, window: int = None, max_live: int = None) -> LevelPlan:
    """Extract an SSA-renamed, level-decomposed plan from a CasADi Function.

    `window` forms levels within windows of that many compute instructions (see
    `_extract_pass1`). `max_live` instead searches: the plain longest-path plan
    first, then halving power-of-two windows down to 64, returning the first
    whose `peak_live_slots` fits, i.e. the fewest extra levels for that budget
    (typically the device's per-block shared memory, `plan.device_max_live`)."""
    instrs = _instructions(f)
    if max_live is None:
        return _level_plan(f, instrs, window)
    n = sum(op not in IO_OPS for op, _, _, _ in instrs)
    for w in [None] + [1 << k for k in range(n.bit_length() - 1, 5, -1)]:
        plan = _level_plan(f, instrs, w)
        if plan.peak_live_slots <= max_live:
            return plan
    raise ValueError(f"{f.name()}: no window brings peak_live_slots ({plan.peak_live_slots}) under {max_live}")


def _level_plan(f: casadi.Function, instrs, window: int) -> LevelPlan:
    consts, nodes, total_ssa = _extract_pass1(instrs, window)
    free_at_level, levels = _compute_liveness(nodes, total_ssa)

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

    for sid, _ in consts:
        ssa_to_phys[sid] = alloc()
    peak_live = next_phys
    for d in range(max(levels, default=-1) + 1):
        for s in free_at_level.get(d, []):
            free_slots.append(ssa_to_phys[s])
        for _, sid, _, _, _, _ in levels.get(d, []):
            if sid >= 0:
                ssa_to_phys[sid] = alloc()
        peak_live = max(peak_live, next_phys - len(free_slots))

    level_offsets, level_widths = [], []
    node_ops, node_outs, node_in_a, node_in_b = [], [], [], []
    for d in sorted(levels):
        level_offsets.append(len(node_ops))
        for op, sid, dep_ssas, _, arg, elem in levels[d]:
            phys = [ssa_to_phys[ds] for ds in dep_ssas if ds >= 0]
            node_ops.append(op)
            if op in IO_OPS:                       # INPUT: slot <- arg[elem]; OUTPUT: arg[elem] <- slot
                node_outs.append(ssa_to_phys[sid] if sid >= 0 else phys[0])
                node_in_a.append(arg)
                node_in_b.append(elem)
            else:
                node_outs.append(ssa_to_phys[sid])
                node_in_a.append(phys[0] if phys else 0)
                node_in_b.append(phys[1] if len(phys) > 1 else 0)
        level_widths.append(len(node_ops) - level_offsets[-1])

    return LevelPlan(
        function_name=f.name(),
        peak_live_slots=peak_live,
        n_inputs=f.n_in(),
        n_outputs=f.n_out(),
        nnz_in=[f.nnz_in(i) for i in range(f.n_in())],
        nnz_out=[f.nnz_out(i) for i in range(f.n_out())],
        const_slots=[ssa_to_phys[sid] for sid, _ in consts],
        const_vals=[float(v) for _, v in consts],
        level_offsets=level_offsets,
        level_widths=level_widths,
        node_ops=node_ops,
        node_outs=node_outs,
        node_in_a=node_in_a,
        node_in_b=node_in_b,
    )


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
        got = [np.zeros(n) for n in plan.nnz_out]
        for s, v in zip(plan.const_slots, plan.const_vals):
            w[s] = v
        for off, wid in zip(plan.level_offsets, plan.level_widths):
            nodes = [(plan.node_ops[i], plan.node_outs[i], plan.node_in_a[i], plan.node_in_b[i])
                     for i in range(off, off + wid)]
            reads = [w[o] if op == OP_OUTPUT else None if op == OP_INPUT else (w[a], w[b])
                     for op, o, a, b in nodes]
            for (op, o, a, b), r in zip(nodes, reads):
                if op == OP_INPUT:
                    w[o] = float(args[a][b])
                elif op == OP_OUTPUT:
                    got[a][b] = r
                else:
                    w[o] = OP_PY[op](r[0], r[1])
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
    print(f"  levels={plan.n_levels} ops={plan.n_nodes} peak_live={plan.peak_live_slots}")
    print("  constants:", list(zip(plan.const_slots, plan.const_vals)))
    for d, (off, wid) in enumerate(zip(plan.level_offsets, plan.level_widths)):
        print(f"  level {d} (width {wid}):")
        for i in range(off, off + wid):
            op, o, a, b = plan.node_ops[i], plan.node_outs[i], plan.node_in_a[i], plan.node_in_b[i]
            print(f"    op={op:<3d} " + (f"w[{o}] <- in[{a}][{b}]" if op == OP_INPUT else
                                         f"out[{a}][{b}] <- w[{o}]" if op == OP_OUTPUT else
                                         f"w[{o}] <- [{a}, {b}]"))


if __name__ == "__main__":
    f = _demo_function()
    plan = extract_level_plan(f)
    _print_plan(plan)
    print("max abs err vs casadi:", verify_plan(f, plan))
    prog = extract_serial_program(f)
    print(f"serial program: {prog.n_instr} instructions, sz_w={prog.sz_w}, "
          f"max abs err vs casadi: {verify_serial_program(f, prog)}")
