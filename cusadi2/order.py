"""The order in which a plan computes each output's nonzeros (`output_order`).

The plan file always stores outputs in CasADi's CCS (column-major) nonzero order; this only
decides the order the planner sees them in, and with it CasADi's depth-first instruction order
and so the live set:

    "col"   CCS order, the function as built (cusadi2's original behaviour).
    "row"   each output's nonzeros row by row (by row, then column): the function is rebuilt
            with every output as a column of its permuted nonzeros, and every OP_OUTPUT of that
            stream is mapped back to its CCS position. For a Jacobian this computes one
            constraint's value and derivatives together; in CCS order the first column already
            touches every row, so all their intermediates stay live until the last column.
    "auto"  both level plans are extracted and the better kept: one that fits the `max_live`
            budget, then fewer levels, then fewer peak live slots (ties: "col").

Both the level plan and the serial program follow the chosen order (the serial program is
extracted from the same reordered stream). Plain casadi Functions and `cusadi2.loads.
LoadedFunction` share this implementation (`with_output_order`).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import casadi as ca
import numpy as np
from casadi import OP_OUTPUT

OUTPUT_ORDERS = ("auto", "row", "col")


def check_output_order(order: str) -> str:
    if order not in OUTPUT_ORDERS:
        raise ValueError(f"cusadi2: output_order must be one of {OUTPUT_ORDERS}, got {order!r}")
    return order


def nonzero_order(sp: ca.Sparsity, order: str) -> np.ndarray:
    """Positions (in CCS nonzero order) of `sp`'s nonzeros listed in `order` ("row" / "col")."""
    if order == "col" or sp.nnz() == 0:
        return np.arange(sp.nnz())
    r, c = sp.get_triplet()
    return np.lexsort((np.asarray(c), np.asarray(r)))


def ordered_outputs(outs: Sequence[ca.SX], order: str) -> Tuple[List[ca.SX], Optional[List[np.ndarray]]]:
    """(outputs to build the planned Function from, perms): "col" keeps the outputs as they are
    (perms None); "row" gives each as a column of its nonzeros in row-major order, perms[k][j]
    being the CCS position of entry j of output k."""
    if order == "col":
        return list(outs), None
    if order != "row":
        raise ValueError(f"cusadi2: ordered_outputs takes 'row' or 'col', got {order!r}")
    flat, perms = [], []
    for o in outs:
        perm = nonzero_order(o.sparsity(), "row")
        perms.append(perm)
        flat.append(ca.vertcat(*[o.nz[int(i)] for i in perm]) if o.nnz() else ca.SX(0, 1))
    return flat, perms


def remap_output(instr: tuple, perms: Optional[List[np.ndarray]]) -> tuple:
    """An (op, inputs, outputs, constant) instruction with an OP_OUTPUT of the reordered Function
    pointed back at its output's CCS position (other instructions unchanged)."""
    op, inp, o, c = instr
    if op != OP_OUTPUT or perms is None:
        return instr
    return (op, inp, (o[0], int(perms[o[0]][o[1]])), c)


class OrderedFunction:
    """A casadi.Function planned with its outputs in `order` ("row"): the planner's view of `fn`
    (every casadi.Function attribute is fn's: interface, serialize, calls) whose instruction
    stream is that of fn rebuilt with reordered outputs, OP_OUTPUTs mapped back to CCS."""

    def __init__(self, fn: ca.Function, order: str):
        if order not in ("row", "col"):
            raise ValueError(f"OrderedFunction: order 'row' or 'col', got {order!r}")
        if not fn.is_a("SXFunction"):
            fn = fn.expand()
        self.fn, self.output_order = fn, order
        self._stream = None

    def __getattr__(self, name):                 # interface, serialize, sparsity ...: fn's
        return getattr(self.fn, name)

    def __call__(self, *args, **kw):
        return self.fn(*args, **kw)

    def _build(self):
        if self._stream is None:
            ins = self.fn.sx_in()
            flat, perms = ordered_outputs(self.fn.call(ins), self.output_order)
            F = ca.Function(self.fn.name(), ins, flat)
            instrs = [remap_output((F.instruction_id(k), tuple(F.instruction_input(k)), tuple(F.instruction_output(k)),
                                    F.instruction_constant(k)), perms) for k in range(F.n_instructions())]
            self._stream = (instrs, F.sz_w())
        return self._stream

    def n_instructions(self) -> int:
        return len(self._build()[0])

    def instruction_id(self, k: int) -> int:
        return self._build()[0][k][0]

    def instruction_input(self, k: int) -> tuple:
        return self._build()[0][k][1]

    def instruction_output(self, k: int) -> tuple:
        return self._build()[0][k][2]

    def instruction_constant(self, k: int) -> float:
        return self._build()[0][k][3]

    def sz_w(self) -> int:
        return self._build()[1]


def with_output_order(fn, order: str):
    """`fn` as the planner should see it with its outputs in `order` ("row" / "col"): a
    LoadedFunction carries the order itself; a plain Function is returned unchanged for "col"
    (today's stream, byte for byte) and wrapped in an OrderedFunction for "row"."""
    from cusadi2.loads import LoadedFunction
    if order not in ("row", "col"):
        raise ValueError(f"cusadi2: with_output_order takes 'row' or 'col', got {order!r}")
    if isinstance(fn, OrderedFunction):
        fn = fn.fn
    if isinstance(fn, LoadedFunction):
        return fn.with_output_order(order)
    return fn if order == "col" else OrderedFunction(fn, order)


def output_order_of(fn) -> str:
    return getattr(fn, "output_order", "col")


def plan_in_order(fn, order: str, max_live: Optional[int], window: Optional[int] = None):
    """(the planner's fn, its level plan, the order used) for `output_order` `order`; "auto"
    extracts both and keeps the one that fits `max_live`, then has fewer levels, then fewer peak
    live slots (ties: "col"). Raises like `extract_level_plan` when no candidate fits."""
    from cusadi2.dag import extract_level_plan
    check_output_order(order)

    def extract(f):
        return extract_level_plan(f, window=window) if max_live is None else extract_level_plan(f, max_live=max_live)
    if order != "auto":
        f = with_output_order(fn, order)
        return f, extract(f), order
    found, errors = [], []
    for o in ("col", "row"):
        f = with_output_order(fn, o)
        try:
            found.append((f, extract(f), o))
        except ValueError as e:              # this order does not fit the budget
            errors.append(e)
    if not found:
        raise errors[0]
    return min(found, key=lambda t: (t[1].n_levels, t[1].peak_live_slots, t[2] != "col"))
