"""Traced lookups: a function of values read from its inputs at computed positions.

SX cannot index with a symbolic position, so a lookup is split at build time into two plain CasADi
Functions on the same leading inputs:

    index(a0, ..., ak)       -> idx (n_g)          the element positions, as SX of the inputs
    fn(a0, ..., an, g)       -> outputs            g (n_g): the looked-up values, as an input

with g_j = arg[table_arg][idx_j]. CasADi sees g as an ordinary input, so derivatives of fn taken
with respect to a0.. treat the looked-up values as constants: exactly the derivative of a table
lookup at an integer position (floor has derivative 0), the case of interpolation on a grid,
where every derivative comes from the interpolation weights. Both Functions are serialisable,
so problems that carry them can be cached and reloaded like any other.

`LoadedFunction(fn, index, table_arg)` is what `cusadi2.compile` / `write_plan` plan: it stands in
for a casadi.Function with fn's declared inputs except g. Its program is one CasADi graph,
fn with g_j replaced by idx_j * M (index's positions times a private marker symbol M), so that
CasADi orders it: each position is computed and each value read next to where it is used, and
common subexpressions (e.g. forward kinematics feeding both the positions and the function) are
shared. In that graph's instruction stream every product with M becomes a LOAD
(`cusadi2.ops.OP_LOAD`: slot <- arg[table_arg][(int) idx_j], truncated and clamped to the
argument) and M itself is never read, so the live set stays that of the function: reading all
values up front would keep every one of them live at once. The runtime interface of the plan is
fn's without g. Index values are truncated toward zero and clamped to the argument (NaN reads
element 0), so a plan never reads out of bounds; front ends compute the exact positions they
want, clamping included (`cusadi2.jax.interpret`'s gather).
"""
from __future__ import annotations

from typing import List, Tuple

import casadi as ca
import numpy as np
from casadi import OP_CONST, OP_INPUT, OP_MUL, OP_OUTPUT, OP_SQ

from cusadi2.ops import OP_LOAD, load_index, load_operand
from cusadi2.order import ordered_outputs, remap_output


class LoadedFunction:
    """fn (last input: the looked-up values g) behind index (fn's first inputs -> positions of g
    in input `table_arg`); see the module docstring. Duck-types the parts of casadi.Function the
    planner uses: declared inputs are fn's without g."""

    def __init__(self, fn: ca.Function, index: ca.Function, table_arg: int, output_order: str = "col",
                 cse: bool = False):
        """output_order: "row" / "col", the order of the planned graph's outputs (cusadi2.order;
        the planner's `output_order` option sets it). cse: eliminate common subexpressions in the
        merged graph (what `cse()` gives)."""
        if fn.n_in() < 2:
            raise ValueError("LoadedFunction: fn needs its declared inputs and the looked-up values g (last)")
        n_decl = fn.n_in() - 1
        n_g = fn.nnz_in(n_decl)
        if not fn.sparsity_in(n_decl).is_dense() or not fn.sparsity_in(n_decl).is_column():
            raise ValueError("LoadedFunction: g (fn's last input) must be a dense column")
        if index.n_out() != 1 or index.nnz_out(0) != n_g or not index.sparsity_out(0).is_dense():
            raise ValueError(f"LoadedFunction: index must have one dense output of {n_g} positions")
        if index.n_in() > n_decl:
            raise ValueError("LoadedFunction: index takes more inputs than fn declares")
        for a in range(index.n_in()):
            if index.sparsity_in(a) != fn.sparsity_in(a):
                raise ValueError(f"LoadedFunction: input {a} of index and fn differ in sparsity")
        if not 0 <= table_arg < n_decl:
            raise ValueError(f"LoadedFunction: table_arg {table_arg} is not a declared input of fn")
        if output_order not in ("row", "col"):
            raise ValueError(f"LoadedFunction: output_order 'row' or 'col', got {output_order!r}")
        self.fn, self.index, self.table_arg = fn, index, int(table_arg)
        self.output_order, self._cse = output_order, bool(cse)
        self.n_g = n_g
        self._stream = None

    # ---------------------------------------------------------------- casadi.Function surface
    def name(self) -> str:
        return self.fn.name()

    def n_in(self) -> int:
        return self.fn.n_in() - 1

    def n_out(self) -> int:
        return self.fn.n_out()

    def nnz_in(self, a: int) -> int:
        return self.fn.nnz_in(self._decl(a))

    def nnz_out(self, k: int) -> int:
        return self.fn.nnz_out(k)

    def sparsity_in(self, a: int) -> ca.Sparsity:
        return self.fn.sparsity_in(self._decl(a))

    def sparsity_out(self, k: int) -> ca.Sparsity:
        return self.fn.sparsity_out(k)

    def name_in(self, a: int = None):
        return self.fn.name_in()[:-1] if a is None else self.fn.name_in(self._decl(a))

    def name_out(self, k: int = None):
        return self.fn.name_out() if k is None else self.fn.name_out(k)

    def is_a(self, kind: str, recursive: bool = True) -> bool:
        return False

    def serialize(self) -> str:
        """The lookup's definition (not the planning order or CSE, which the plan key records)."""
        return "cusadi2.LoadedFunction\0%d\0%s\0%s" % (self.table_arg, self.fn.serialize(), self.index.serialize())

    def _decl(self, a: int) -> int:
        if not 0 <= a < self.n_in():
            raise IndexError(f"{self.name()}: no declared input {a}")
        return a

    # ---------------------------------------------------------------- evaluation (host reference)
    def gather(self, args) -> np.ndarray:
        """g at the declared inputs `args` (numeric), as the kernel computes it."""
        idx = np.asarray(ca.DM(self.index(*[ca.DM(a) for a in args[:self.index.n_in()]])).nonzeros())
        table = np.asarray(ca.DM(args[self.table_arg]).nonzeros())
        return np.array([table[load_index(v, len(table))] for v in idx], dtype=float)

    def __call__(self, *args):
        g = self.gather(args)
        return self.fn(*[ca.DM(a) for a in args], ca.DM(g))

    def call(self, args):
        if any(isinstance(a, (ca.SX, ca.MX)) for a in args):
            raise TypeError("LoadedFunction: symbolic calls are not defined (a lookup has no SX form)")
        out = self(*args)
        return list(out) if isinstance(out, (list, tuple)) else [out]

    def cse(self) -> "LoadedFunction":
        """The same lookup with common subexpressions eliminated in the planned graph (the
        planner's `cse_function`; applied to the merged graph, so it spans index and fn)."""
        return LoadedFunction(self.fn, self.index, self.table_arg, self.output_order, cse=True)

    def with_output_order(self, order: str) -> "LoadedFunction":
        """The same lookup planned with its outputs in `order` ("row" / "col"; cusadi2.order)."""
        return LoadedFunction(self.fn, self.index, self.table_arg, order, cse=self._cse)

    # ---------------------------------------------------------------- merged instruction stream
    def merged_function(self) -> ca.Function:
        """fn(a0.., idx * M) as one SX Function of the declared inputs and the marker M (last), its
        outputs in `output_order` (cusadi2.order.ordered_outputs; `_perm` maps them back)."""
        F, I = self.fn, self.index
        if not F.is_a("SXFunction"):
            F = F.expand()
        if not I.is_a("SXFunction"):
            I = I.expand()
        decl = [ca.SX.sym(F.name_in(a), F.sparsity_in(a)) for a in range(self.n_in())]
        M = ca.SX.sym("cusadi2_load_mark")
        idx = ca.densify(I.call(decl[:I.n_in()])[0])
        table, n = decl[self.table_arg], self.fn.nnz_in(self.table_arg)
        # a position CasADi folded to a constant reads that table element directly (c * M would fold
        # to 0 or M, losing the read)
        g = ca.vertcat(*[table.nz[load_index(float(idx[j]), n)] if idx[j].is_constant() else idx[j] * M
                         for j in range(idx.numel())])
        flat, self._perm = ordered_outputs(F.call(decl + [g]), self.output_order)
        if self._cse:
            flat = [ca.cse(o) if o.nnz() else o for o in flat]
        return ca.Function(F.name(), decl + [M], flat)

    def _merged(self) -> Tuple[List[tuple], int]:
        if self._stream is not None:
            return self._stream
        C = self.merged_function()
        mark_arg = C.n_in() - 1
        operand = load_operand(self.table_arg, self.fn.nnz_in(self.table_arg))
        mark_slots, out, n_loads = set(), [], 0
        for k in range(C.n_instructions()):
            op, inp, o, c = C.instruction_id(k), tuple(C.instruction_input(k)), tuple(C.instruction_output(k)), \
                C.instruction_constant(k)
            if op == OP_INPUT and inp[0] == mark_arg:
                mark_slots.add(o[0])               # M lives here; never read as a value
                continue
            if op == OP_OUTPUT:                    # back to the output's CCS position
                out.append(remap_output((op, inp, o, c), self._perm))
                continue
            if op == OP_MUL and len(inp) == 2 and (inp[0] in mark_slots) != (inp[1] in mark_slots):
                src = inp[1] if inp[0] in mark_slots else inp[0]
                out.append((OP_LOAD, (src, operand), o, 0.0))
                n_loads += 1
            else:
                if op not in (OP_CONST, OP_INPUT) and any(s in mark_slots for s in (inp[:1] if op == OP_SQ else inp[:2])):
                    raise RuntimeError(f"{self.name()}: the load marker reached a {op} instruction; "
                                       "the lookup could not be planned")
                out.append((op, inp, o, c))
            if op != OP_OUTPUT and o:              # the slot is overwritten: it no longer holds M
                mark_slots.discard(o[0])
        self.n_loads = n_loads
        self._stream = (out, C.sz_w())
        return self._stream

    def n_instructions(self) -> int:
        return len(self._merged()[0])

    def instruction_id(self, k: int) -> int:
        return self._merged()[0][k][0]

    def instruction_input(self, k: int) -> tuple:
        return self._merged()[0][k][1]

    def instruction_output(self, k: int) -> tuple:
        return self._merged()[0][k][2]

    def instruction_constant(self, k: int) -> float:
        return self._merged()[0][k][3]

    def sz_w(self) -> int:
        return self._merged()[1]
