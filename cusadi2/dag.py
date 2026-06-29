"""Level-parallel DAG scheduler.

This is the *compiler front-end* of the pipeline: it turns one CasADi function
into the level-decomposed, SSA-renamed plan that the data-driven kernel (see
``kernel_template.py``) interprets on device. The kernel is function-agnostic;
``LevelPlan`` is the per-function "program" it executes.

What it produces, and how the kernel consumes it:

  * **node arrays** (``node_ops`` / ``node_outs`` / ``node_in_a`` / ``node_in_b``)
    plus the ``level_offsets`` / ``level_widths`` CSR index — every compute op
    flattened into parallel arrays, grouped into topological levels. The kernel
    runs one level at a time with a single ``__syncthreads()`` between levels;
    within a level all nodes are independent and run in parallel. Node *i* is
    ``node_ops[i]`` writing slot ``node_outs[i]`` from slots ``node_in_a[i]`` /
    ``node_in_b[i]`` (binary max — unary ops ignore ``in_b``). Level *d* (0-based)
    spans nodes ``[level_offsets[d], level_offsets[d] + level_widths[d])``.
  * **constants / input maps / output maps** (parallel lists) — what to preload
    into the shared-memory work array before compute, and what to write back after.
  * **peak_live_slots** — the level-parallel recycled peak; the kernel's
    shared-memory budget is ``peak_live_slots * 8`` (fp64). ``n_work_seq`` is the
    CasADi sequential work-vector size (``f.sz_w()``), kept so the unified plan can
    also model the old sequential kernel's per-instance footprint for comparison.

Two passes (see ``extract_level_plan``):
  1. Naive longest-path level assignment on CasADi's own work slots — preserves
     maximum available parallelism.
  2. SSA renaming with slot recycling — every node writes a fresh slot; freed
     slots are reused at level boundaries. This removes all write-write and
     read-write conflicts *within* a level, so the only barrier needed is the
     one between levels.

Tiling note: the original code carried two near-identical plan types,
``LevelPlan`` and ``TiledLevelPlan`` (whose own docstring admitted "when
n_chunks == 1, this is semantically identical to a LevelPlan"). Here there is a
single ``LevelPlan`` with a ``.tiling`` pytree (see ``_empty_tiling``): all-empty
lists are the untiled fast path (working set fits in shared memory), populated
lists are the overflow path. The tiled scheduler that fills it is the next piece
to port; ``extract_level_plan`` below leaves it empty.
"""

import dataclasses
from typing import Dict, List, Tuple

import casadi
from casadi import OP_CONST, OP_INPUT, OP_OUTPUT, OP_SQ


# ---------------------------------------------------------------------------
# Tiling schedule — a pure-python pytree mirroring the original Chunk /
# ChunkBoundary / spill dataclasses as plain dicts of parallel lists. Empty
# leaves are the untiled fast path; the forthcoming tiled scheduler fills them
# (and the kernel reads them) when a function's working set overflows smem.
# ---------------------------------------------------------------------------
def _empty_tiling() -> dict:
    """Empty tiling schedule (untiled fast path).

      chunks     — contiguous spans of levels run before a shared-memory reset,
                   as parallel lists: level_start, level_end (exclusive),
                   peak_live (per-chunk smem high-water). Plan smem =
                   max(peak_live); a single chunk is exactly the untiled case.
      boundaries — between two consecutive chunks: spill (smem -> global) then
                   reload (global -> smem). One entry per boundary; the inner
                   lists are variable length (slots spilled/reloaded that time).
      spill_buffer_stride — per-instance global spill-buffer width; > 0 iff
                   tiling is active.
      sub_batch  — levels too wide to hold their outputs in smem at once, keyed
                   by compute-level index -> (smem slots, spill-buffer positions).
    """
    return {
        "chunks": {"level_start": [], "level_end": [], "peak_live": []},
        "boundaries": {"spill_src_slots": [], "spill_buffer_pos": [],
                       "reload_buffer_pos": [], "reload_dst_slots": []},
        "spill_buffer_stride": 0,
        "sub_batch": {"level": [], "slots": [], "pos": []},
    }


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class LevelPlan:
    """Complete level-decomposed, SSA-renamed execution plan for one function.

    A pytree of plain parallel lists — this *is* the wire format the native
    backend emits; there is no separate serializer. The I/O maps bind slots to
    constants / external elements; the node arrays are every compute op flattened
    in level order, indexed by the ``level_offsets`` / ``level_widths`` CSR.
    """
    function_name: str
    n_work_seq: int          # CasADi sequential work-vector size (f.sz_w()) —
                             # the old sequential model's per-instance footprint
    peak_live_slots: int     # level-parallel recycled peak = this kernel's smem
    n_inputs: int
    n_outputs: int
    nnz_in: List[int]
    nnz_out: List[int]

    # constants loaded into slots before compute
    const_slots: List[int]
    const_vals: List[float]
    # external input element -> slot, before compute
    input_slots: List[int]
    input_args: List[int]
    input_elems: List[int]
    # slot -> external output element, after compute
    out_slots: List[int]
    out_args: List[int]
    out_elems: List[int]

    # compute nodes flattened in level order; level d spans nodes
    # [level_offsets[d], level_offsets[d] + level_widths[d])
    level_offsets: List[int]
    level_widths: List[int]
    node_ops: List[int]      # CasADi op id per node
    node_outs: List[int]     # slot each node writes
    node_in_a: List[int]     # first input slot (0 if none)
    node_in_b: List[int]     # second input slot (0 if unary)

    # Tiling schedule (see _empty_tiling): a pure-python pytree, all-empty on the
    # untiled fast path (working set fits in shared memory). The forthcoming
    # tiled scheduler populates its chunks / boundaries / spill lists.
    tiling: dict = dataclasses.field(default_factory=_empty_tiling)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _extract_pass1(f: casadi.Function, debug: bool = False):
    """Pass 1: parse CasADi instructions, assign topological levels, build the
    work-slot -> SSA-id map.

    Returns (raw_constants, raw_inputs, raw_outputs, raw_compute, total_ssa,
             n_in, n_out, nnz_in, nnz_out). With ``debug=True``, also pops a
    matplotlib plot of the level-layered dependency DAG.
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

    if debug:
        import matplotlib.pyplot as plt
        from collections import defaultdict
        op_name = {getattr(casadi, n): n for n in dir(casadi) if n.startswith("OP_")}
        # ssa_id -> (level, op_id, label); plus dependency edges
        nodes = {sid: (0, OP_CONST, f"CONST\n{v:g}") for sid, _, v in raw_constants}
        nodes.update({sid: (0, OP_INPUT, f"IN[{a},{e}]")
                      for sid, a, e, _ in raw_inputs})
        edges = []
        for _, op, sid, deps, d, _ in raw_compute:
            nodes[sid] = (d, op, op_name[op][3:])           # strip "OP_"
            edges += [(ds, sid) for ds in deps if ds >= 0]
        by_level = defaultdict(list)
        for sid, (lvl, _, _) in nodes.items():
            by_level[lvl].append(sid)
        pos = {sid: (i - (len(s) - 1) / 2, -lvl)            # x = position in level, y = -depth (top-down)
               for lvl, s in by_level.items() for i, sid in enumerate(sorted(s))}
        ops = sorted({o for _, o, _ in nodes.values()})     # one colour per op type
        colour = {o: plt.cm.tab10(i % 10) for i, o in enumerate(ops)}
        nlev = max(by_level)
        _, ax = plt.subplots(figsize=(2.5 + 1.4 * max(map(len, by_level.values())),
                                      1.5 + 1.1 * nlev))
        for lvl in by_level:                                # per-level background band
            ax.axhspan(-lvl - .45, -lvl + .45, color=plt.cm.Pastel1(lvl % 9), alpha=.4)
        for a, b in edges:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.3))
        for sid, (lvl, op, lab) in nodes.items():           # circles coloured by op
            x, y = pos[sid]
            ax.scatter(x, y, s=1300, color=colour[op], edgecolors="k", zorder=3)
            ax.text(x, y, f"{lab}\n#{sid}", ha="center", va="center", fontsize=7, zorder=4)
        ax.set_yticks([-l for l in range(nlev + 1)], [f"L{l}" for l in range(nlev + 1)])
        ax.set_xticks([])
        ax.set_ylabel("level (depth = serial cost)   |   row width = parallelism")
        ax.set_title(f"{f.name()}: SSA dependency DAG by level")
        plt.tight_layout()
        plt.savefig("lp_dag.png", dpi=130, bbox_inches="tight")
        plt.close()

    return (raw_constants, raw_inputs, raw_outputs, raw_compute,
            ssa_counter, n_in, n_out, nnz_in, nnz_out)


def _compute_liveness(raw_constants, raw_inputs, raw_compute, raw_outputs, total_ssa):
    """Group compute nodes by level and compute the slot free schedule.

    Returns (free_at_level, levels_dict, max_level). ``free_at_level[d]`` lists
    the SSA values whose slots become reusable at level d (last read at d-1);
    ``levels_dict[d]`` holds the compute nodes at level d. The intermediate
    produced/last-read arrays stay local — nothing downstream consumes them.
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
    # Hoisted: the max compute level is invariant across the output loop.
    # Recomputing it inside turns a linear pass into N*M (142M genexpr calls /
    # 22 s for H1's 1.8k outputs x 78k instructions).
    if raw_compute:
        max_compute_level = max(d for _, _, _, _, d, _ in raw_compute)
        out_level = max_compute_level + 1
        for src_ssa, _, _ in raw_outputs:
            if src_ssa >= 0 and out_level > ssa_last_read_level[src_ssa]:
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

    return free_at_level, levels_dict, max_level


def extract_level_plan(f: casadi.Function) -> LevelPlan:
    """Extract an SSA-renamed, level-decomposed plan from a CasADi Function.

    Returns an untiled plan. If its shared-memory footprint
    (``peak_live_slots * 8`` bytes) exceeds the device budget, the caller routes
    to the (forthcoming) tiled scheduler instead.
    """
    (raw_constants, raw_inputs, raw_outputs, raw_compute,
     total_ssa, n_in, n_out, nnz_in, nnz_out) = _extract_pass1(f)

    free_at_level, levels_dict, max_level = _compute_liveness(
        raw_constants, raw_inputs, raw_compute, raw_outputs, total_ssa)

    # Allocate physical slots with a free-list, level by level. A slot is freed
    # one level after its last read and immediately reusable, so the high-water
    # mark of allocated slots equals the peak number simultaneously live.
    ssa_to_phys = [0] * total_ssa
    free_slots: List[int] = []
    next_phys = [0]

    def alloc_slot() -> int:
        if free_slots:
            return free_slots.pop()
        s = next_phys[0]
        next_phys[0] += 1
        return s

    # Level 0: constants and inputs.
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

    # ---- Build the final plan with physical slot references ----
    const_slots = [ssa_to_phys[sid] for sid, _, _ in raw_constants]
    const_vals = [val for _, _, val in raw_constants]

    input_slots, input_args, input_elems = [], [], []
    for sid, arg, elem, _ in raw_inputs:
        input_slots.append(ssa_to_phys[sid])
        input_args.append(arg)
        input_elems.append(elem)

    out_slots, out_args, out_elems = [], [], []
    for src_ssa, oarg, oelem in raw_outputs:
        if src_ssa >= 0:
            out_slots.append(ssa_to_phys[src_ssa])
            out_args.append(oarg)
            out_elems.append(oelem)

    # Flatten compute nodes in level order into CSR-indexed parallel arrays.
    level_offsets, level_widths = [], []
    node_ops, node_outs, node_in_a, node_in_b = [], [], [], []
    for d in sorted(levels_dict.keys()):
        level_offsets.append(len(node_ops))
        for _k, op, sid, dep_ssas, _cv in levels_dict[d]:
            phys = [ssa_to_phys[ds] for ds in dep_ssas if ds >= 0]
            node_ops.append(op)
            node_outs.append(ssa_to_phys[sid])
            node_in_a.append(phys[0] if phys else 0)
            node_in_b.append(phys[1] if len(phys) > 1 else 0)
        level_widths.append(len(node_ops) - level_offsets[-1])

    return LevelPlan(
        function_name=f.name(),
        n_work_seq=f.sz_w(),
        peak_live_slots=peak_live,
        n_inputs=n_in,
        n_outputs=n_out,
        nnz_in=nnz_in,
        nnz_out=nnz_out,
        const_slots=const_slots,
        const_vals=const_vals,
        input_slots=input_slots,
        input_args=input_args,
        input_elems=input_elems,
        out_slots=out_slots,
        out_args=out_args,
        out_elems=out_elems,
        level_offsets=level_offsets,
        level_widths=level_widths,
        node_ops=node_ops,
        node_outs=node_outs,
        node_in_a=node_in_a,
        node_in_b=node_in_b,
    )


def pick_block_size(plan: LevelPlan) -> int:
    """Smallest power of two >= the plan's median level width, capped at 256.

    The kernel uses one block per (instance, stage); this sizes the block to the
    typical level so most levels are covered in a single thread pass.
    """
    widths = sorted(plan.level_widths)
    med = widths[len(widths) // 2] if widths else 0
    if med <= 0:
        return 32
    bs = 32
    while bs < med and bs < 256:
        bs *= 2
    return bs


# ---------------------------------------------------------------------------
# Stage axis (horizon). The kernel runs one block per (instance, stage) over a
# (batch x n_stages) grid; each arg is indexed along the stage axis by a per-arg
# mode. Plain batch-only LP is every arg "stage" with n_stages == 1. Specs are
# plain parallel lists (SoA, like LevelPlan) — no per-arg record type: modes are
# readable strings, mapped to their kernel wire value (position in STAGE_MODES)
# only at emit. in_mode / in_offset / in_span for inputs, out_mode for outputs.
#   "stage"   — independent value per stage    (leading-axis length n_stages)
#   "shared"  — one value broadcast to stages  (leading-axis length 1)
#   "coupled" — stage k reads element k+offset (leading-axis length n_stages+span)
# ---------------------------------------------------------------------------
STAGE_MODES = ("stage", "shared", "coupled")   # index == kernel wire value (0/1/2)


def stage_count(mode: str, span: int, n_stages: int) -> int:
    """Leading-axis length an arg's buffer needs for `n_stages`."""
    if mode == "shared":
        return 1
    if mode == "coupled":
        return n_stages + span
    return n_stages


def validate_specs(fn, n_stages, in_mode, in_offset, in_span, out_mode) -> None:
    """Raise if the stage specs (parallel lists) are inconsistent with `fn`, the
    horizon, or each other. Inputs may be "stage"/"shared"/"coupled"; outputs
    "stage" only."""
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
        elif m == "stage" and off != 0:
            raise ValueError(f"input {i}: offset is only valid for coupled mode")
    for i, m in enumerate(out_mode):
        if m != "stage":
            raise ValueError(f'output {i}: only "stage" mode is supported on outputs.')


# ---------------------------------------------------------------------------
# Debug entry point — run this file directly (or with the VSCode debugger) to
# step through extraction on a tiny function. Good breakpoints to set:
#   - inside _extract_pass1: watch `op`, `inp`, `out`, `slot_to_ssa`, `d`
#   - inside extract_level_plan's level loop: watch `free_slots`, `next_phys`,
#     `peak_live` to see slot recycling in action.
# ---------------------------------------------------------------------------
def _demo_function() -> casadi.Function:
    """Tiny function that exercises every instruction category.

    `a = sin(x0) + p` is reused by both outputs, so its slot stays live across a
    level (visible in liveness) and the constant 2.0 becomes an OP_CONST.
        inputs:  x (2 elements), p (1 scalar)
        outputs: [a * x1, a + 2.0]
    """
    import casadi as ca
    x = ca.SX.sym("x", 2)
    p = ca.SX.sym("p")
    a = ca.sin(x[0]) + p          # OP_SIN, OP_ADD; read twice below
    out = ca.vertcat(a * x[1],    # OP_MUL
                     a + 2.0)     # OP_CONST + OP_ADD
    return ca.Function("demo", [x, p], [out])


def _print_plan(plan: LevelPlan) -> None:
    n_levels = len(plan.level_offsets)
    n_ops = len(plan.node_ops)
    shared_bytes = plan.peak_live_slots * 8
    print(f"function: {plan.function_name}")
    print(f"  inputs={plan.n_inputs} nnz_in={plan.nnz_in}  "
          f"outputs={plan.n_outputs} nnz_out={plan.nnz_out}")
    print(f"  levels={n_levels} ops={n_ops} "
          f"peak_live={plan.peak_live_slots} n_work_seq={plan.n_work_seq} "
          f"shared_bytes={shared_bytes} block_size={pick_block_size(plan)}")
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
    plan = extract_level_plan(f)   # <-- set a breakpoint here and step in
    _print_plan(plan)
    _extract_pass1(f, debug=True)  # writes lp_dag.png (level-layered DAG, top-down)
