"""Composed plans: a per-stage function behind a whole-horizon interface.

A plan's program is one function's graph, run once per (instance, stage) by the
kernel. What the kernel needs to know besides the program is where each stage's
inputs come from and where its outputs go. A `Composition` says exactly that,
per element, with tables:

  * declared arguments: the buffers the caller passes (`inputs`) and receives
    (`outputs`), each with its own sparsity and per-instance length;
  * `stage_in[a]` for the function's input `a`: which declared input it reads
    (`arg`) and, per (stage, element), the index inside that buffer's instance
    block, or -1 to read a zero;
  * `stage_out[k]` for the function's output `k`: which declared output it
    writes (`arg`, or -1 for the plan's scratch stream) and, per (stage,
    element), the index inside the destination's instance block, or -1 to
    discard the value;
  * `reduces`: declared outputs whose entries are sums of several stage values
    (a Lagrangian Hessian where one stage's x_{k+1} block meets the next
    stage's x_k block; a cost summed over stages). The kernel writes those
    stage values stage-major into a scratch stream of `scratch_per_instance`
    scalars per instance and a second launch sums each entry's segment in a
    fixed order, so the result is deterministic.

`legacy_composition` expresses the plain stage specs (`write_plan`'s
`n_stages`, `in_mode`, `in_offset`, `in_span`): every function argument is its
own declared buffer and the tables are the stride arithmetic. `compose` builds
a composition from gather tables and global (row, col) tables, merging the
global sparsity into CasADi's column-major nonzero order so that a consumer
reading the declared output's CCS (cuipm) sees the whole-horizon pattern.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Tuple

import numpy as np

from cusadi2.dag import STAGE_MODES, stage_count, validate_specs


@dataclasses.dataclass
class DeclaredArg:
    """One buffer of the plan's interface: CasADi CCS sparsity of one block and the
    stage-axis layout the header records (mode, offset, span, count); the buffer
    holds `count * nnz` scalars per instance."""
    name: str
    rows: int
    cols: int
    colptr: np.ndarray
    rowidx: np.ndarray
    mode: str = "stage"
    offset: int = 0
    span: int = 0
    count: int = 1

    def __post_init__(self):
        self.colptr = np.ascontiguousarray(self.colptr, dtype=np.int32)
        self.rowidx = np.ascontiguousarray(self.rowidx, dtype=np.int32)
        if len(self.colptr) != self.cols + 1 or int(self.colptr[-1]) != len(self.rowidx):
            raise ValueError(f"{self.name}: inconsistent CCS")
        if self.mode not in STAGE_MODES:
            raise ValueError(f"{self.name}: unknown mode {self.mode!r}")

    @property
    def nnz(self) -> int:
        return int(len(self.rowidx))

    @property
    def length(self) -> int:
        """Scalars per instance."""
        return self.count * self.nnz

    def coo(self) -> Tuple[np.ndarray, np.ndarray]:
        rows = self.rowidx
        cols = np.repeat(np.arange(self.cols, dtype=np.int32), np.diff(self.colptr))
        return rows, cols

    @staticmethod
    def dense(name: str, n: int, **kw) -> "DeclaredArg":
        """A dense column of n scalars."""
        return DeclaredArg(name, n, 1, np.array([0, n]), np.arange(n), **kw)

    @staticmethod
    def from_sparsity(name: str, sp, **kw) -> "DeclaredArg":
        """From a casadi.Sparsity."""
        return DeclaredArg(name, sp.size1(), sp.size2(), np.asarray(sp.colind()), np.asarray(sp.row()), **kw)

    @staticmethod
    def from_coo(name: str, rows: int, cols: int, r, c, **kw) -> "DeclaredArg":
        """From (row, col) pairs, unique, in any order; stored column-major sorted."""
        r, c = np.asarray(r, np.int64), np.asarray(c, np.int64)
        order = np.lexsort((r, c))
        r, c = r[order], c[order]
        colptr = np.zeros(cols + 1, np.int64)
        np.add.at(colptr, c + 1, 1)
        return DeclaredArg(name, rows, cols, np.cumsum(colptr), r, **kw)


@dataclasses.dataclass
class StageMap:
    """Where function argument reads from / writes to: the declared argument `arg`
    (-1 on outputs: the scratch stream) and the (n_stages, nnz) table of indices
    inside that argument's instance block (-1: zero on inputs, discard on outputs)."""
    arg: int
    table: np.ndarray

    def __post_init__(self):
        self.table = np.ascontiguousarray(self.table, dtype=np.int32)
        if self.table.ndim != 2:
            raise ValueError("StageMap table must be (n_stages, nnz)")

    @property
    def nnz(self) -> int:
        return int(self.table.shape[1])


@dataclasses.dataclass
class Reduce:
    """Declared output `dst` (G entries): entry g is the sum of the scratch-stream
    scalars segidx[segptr[g]:segptr[g+1]] (indices inside the instance's scratch block)."""
    dst: int
    segptr: np.ndarray
    segidx: np.ndarray

    def __post_init__(self):
        self.segptr = np.ascontiguousarray(self.segptr, dtype=np.int32)
        self.segidx = np.ascontiguousarray(self.segidx, dtype=np.int32)


@dataclasses.dataclass
class Composition:
    n_stages: int
    inputs: List[DeclaredArg]
    outputs: List[DeclaredArg]
    stage_in: List[StageMap]
    stage_out: List[StageMap]
    reduces: List[Reduce] = dataclasses.field(default_factory=list)
    scratch_per_instance: int = 0

    def validate(self, fn=None) -> None:
        """Raise if the tables do not fit the declared arguments (and `fn` if given)."""
        if fn is not None:
            if fn.n_in() != len(self.stage_in) or fn.n_out() != len(self.stage_out):
                raise ValueError("composition: argument count differs from the function's")
            for a, m in enumerate(self.stage_in):
                if m.nnz != fn.nnz_in(a):
                    raise ValueError(f"composition: input {a} table has {m.nnz} elements, fn has {fn.nnz_in(a)}")
            for k, m in enumerate(self.stage_out):
                if m.nnz != fn.nnz_out(k):
                    raise ValueError(f"composition: output {k} table has {m.nnz} elements, fn has {fn.nnz_out(k)}")
        for a, m in enumerate(self.stage_in):
            if m.table.shape[0] != self.n_stages:
                raise ValueError(f"composition: input {a} table has {m.table.shape[0]} stages, plan has {self.n_stages}")
            if not 0 <= m.arg < len(self.inputs):
                raise ValueError(f"composition: input {a} reads declared input {m.arg}, which does not exist")
            if m.table.max(initial=-1) >= self.inputs[m.arg].length or m.table.min(initial=0) < -1:
                raise ValueError(f"composition: input {a} table indexes outside declared input {m.arg}")
        for k, m in enumerate(self.stage_out):
            if m.table.shape[0] != self.n_stages:
                raise ValueError(f"composition: output {k} table has {m.table.shape[0]} stages, plan has {self.n_stages}")
            limit = self.scratch_per_instance if m.arg < 0 else self.outputs[m.arg].length
            if m.arg >= len(self.outputs):
                raise ValueError(f"composition: output {k} writes declared output {m.arg}, which does not exist")
            if m.table.max(initial=-1) >= limit or m.table.min(initial=0) < -1:
                raise ValueError(f"composition: output {k} table indexes outside its destination")
        for r in self.reduces:
            if len(r.segptr) != self.outputs[r.dst].length + 1:
                raise ValueError(f"composition: reduce for output {r.dst} has the wrong segment count")
            if r.segidx.max(initial=-1) >= self.scratch_per_instance or r.segidx.min(initial=0) < 0:
                raise ValueError(f"composition: reduce for output {r.dst} indexes outside the scratch stream")

    @property
    def in_lengths(self) -> List[int]:
        return [a.length for a in self.inputs]

    @property
    def out_lengths(self) -> List[int]:
        return [a.length for a in self.outputs]


# ---------------------------------------------------------------------------- builders
def legacy_composition(fn, n_stages: int = 1, in_mode=None, in_offset=None, in_span=None,
                       out_mode=None) -> Composition:
    """The plain stage specs as a composition: one declared buffer per function
    argument, laid out (instance, stage_count, nnz); the tables are the kernel's
    former stride arithmetic (stage s reads block s + offset, shared reads block 0)."""
    in_mode = in_mode or ["stage"] * fn.n_in()
    in_offset = in_offset or [0] * fn.n_in()
    in_span = in_span or [0] * fn.n_in()
    out_mode = out_mode or ["stage"] * fn.n_out()
    validate_specs(fn, n_stages, in_mode, in_offset, in_span, out_mode)
    inputs, stage_in = [], []
    for a in range(fn.n_in()):
        count = stage_count(in_mode[a], in_span[a], n_stages)
        inputs.append(DeclaredArg.from_sparsity(fn.name_in(a), fn.sparsity_in(a), mode=in_mode[a],
                                                offset=in_offset[a], span=in_span[a], count=count))
        nnz = fn.nnz_in(a)
        if in_mode[a] == "shared":
            table = np.tile(np.arange(nnz), (n_stages, 1))
        else:
            table = (np.arange(n_stages)[:, None] + in_offset[a]) * nnz + np.arange(nnz)[None, :]
        stage_in.append(StageMap(a, table))
    outputs, stage_out = [], []
    for k in range(fn.n_out()):
        nnz = fn.nnz_out(k)
        outputs.append(DeclaredArg.from_sparsity(fn.name_out(k), fn.sparsity_out(k), mode=out_mode[k],
                                                 count=stage_count(out_mode[k], 0, n_stages)))
        stage_out.append(StageMap(k, np.arange(n_stages)[:, None] * nnz + np.arange(nnz)[None, :]))
    return Composition(n_stages, inputs, outputs, stage_in, stage_out)


def gather_block(n_stages: int, nnz: int, base: int, stride: int, offset: int = 0,
                 count: Optional[int] = None) -> np.ndarray:
    """Table reading block `s + offset` of a knot-major region of `count` blocks of
    `stride` scalars starting at `base`: element b of stage s at
    base + (s + offset) * stride + b; -1 (zero) where the block index leaves [0, count)."""
    knot = np.arange(n_stages) + offset
    table = base + knot[:, None] * stride + np.arange(nnz)[None, :]
    if count is not None:
        table = np.where(((knot >= 0) & (knot < count))[:, None], table, -1)
    return table.astype(np.int32)


def gather_shared(n_stages: int, nnz: int, base: int) -> np.ndarray:
    """Every stage reads the same nnz scalars at `base`."""
    return np.tile(base + np.arange(nnz, dtype=np.int32), (n_stages, 1))


def gather_zero(n_stages: int, nnz: int) -> np.ndarray:
    return np.full((n_stages, nnz), -1, np.int32)


def compose(fn, n_stages: int, inputs: Sequence[DeclaredArg], gathers: Sequence[Tuple[int, np.ndarray]],
            outputs: Sequence[Tuple[str, int, int]], scatters: Sequence[Tuple[int, np.ndarray]]) -> Composition:
    """Build a composition of `fn` over `n_stages`.

    inputs:   the declared inputs, in call order.
    gathers:  per function input a: (declared input index, table (n_stages, nnz_in[a])).
    outputs:  the declared outputs as (name, rows, cols); their sparsity is the merged
              set of global (row, col) pairs the scatters send to them, column-major.
    scatters: per function output k: (declared output index, rc (n_stages, nnz_out[k], 2)
              of global (row, col), -1 to discard).

    A declared output every entry of which receives exactly one stage value is
    written directly; one with any duplicate goes through the scratch stream and a
    deterministic segmented sum."""
    inputs = list(inputs)
    stage_in = [StageMap(src, np.asarray(t)) for src, t in gathers]
    decl_out: List[DeclaredArg] = []
    stage_out: List[Optional[StageMap]] = [None] * len(scatters)
    reduces: List[Reduce] = []
    scratch = 0
    for d, (name, rows, cols) in enumerate(outputs):
        members = [(k, np.asarray(rc, np.int64)) for k, (dst, rc) in enumerate(scatters) if dst == d]
        keys, owners = [], []          # global key (col * rows + row) per contributing (k, s, b), in stream order
        for k, rc in members:
            if rc.shape != (n_stages, fn.nnz_out(k), 2):
                raise ValueError(f"compose: scatter {k} has shape {rc.shape}, expected ({n_stages}, {fn.nnz_out(k)}, 2)")
            valid = rc[..., 0] >= 0
            if np.any(valid & ((rc[..., 0] >= rows) | (rc[..., 1] >= cols) | (rc[..., 1] < 0))):
                raise ValueError(f"compose: scatter {k} has entries outside ({rows}, {cols})")
            key = np.where(valid, rc[..., 1] * rows + rc[..., 0], -1).reshape(-1)
            keys.append(key)
            owners.append(k)
        allkeys = np.concatenate(keys) if keys else np.zeros(0, np.int64)
        uniq, inverse, counts = np.unique(allkeys[allkeys >= 0], return_inverse=True, return_counts=True)
        decl = DeclaredArg.from_coo(name, rows, cols, uniq % rows, uniq // rows)
        decl_out.append(decl)
        G = decl.nnz
        direct = bool(np.all(counts == 1)) if G else True
        pos = np.full(len(allkeys), -1, np.int64)
        pos[allkeys >= 0] = inverse
        start = 0
        if direct:
            for k, key in zip(owners, keys):
                stage_out[k] = StageMap(d, pos[start:start + key.size].reshape(n_stages, -1))
                start += key.size
        else:
            raw_pos, raw_idx = [], []
            for k, key in zip(owners, keys):
                n = key.size
                ident = np.arange(n, dtype=np.int64)
                # this output's raw block in the scratch stream starts at `scratch`; stage-major identity
                stage_out[k] = StageMap(-1, np.where(key >= 0, scratch + ident, -1).reshape(n_stages, -1))
                raw_pos.append(pos[start:start + n])
                raw_idx.append(scratch + ident)
                scratch += n
                start += n
            p = np.concatenate(raw_pos)
            i = np.concatenate(raw_idx)
            keep = p >= 0
            p, i = p[keep], i[keep]
            order = np.lexsort((i, p))          # by entry, then stream order: fixed summation order
            segptr = np.zeros(G + 1, np.int64)
            np.add.at(segptr, p + 1, 1)
            reduces.append(Reduce(d, np.cumsum(segptr), i[order]))
    for k, m in enumerate(stage_out):
        if m is None:
            raise ValueError(f"compose: function output {k} is not sent to any declared output")
    comp = Composition(n_stages, inputs, decl_out, stage_in, list(stage_out), reduces, scratch)
    comp.validate(fn)
    return comp


# ---------------------------------------------------------------------------- host reference
def evaluate_composition(fn, comp: Composition, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Evaluate `fn` under `comp` on the host for one instance: declared input
    buffers in -> declared output buffers out, the way the kernel and the reduce
    launch do (zeros for -1 reads, discards for -1 writes, segment sums)."""
    inputs = [np.asarray(x, np.float64).reshape(-1) for x in inputs]
    for a, (buf, decl) in enumerate(zip(inputs, comp.inputs)):
        if buf.size != decl.length:
            raise ValueError(f"declared input {a} has {buf.size} scalars, expected {decl.length}")
    outs = [np.zeros(decl.length) for decl in comp.outputs]
    scratch = np.zeros(comp.scratch_per_instance)
    for s in range(comp.n_stages):
        args = []
        for m in comp.stage_in:
            t = m.table[s]
            v = np.where(t >= 0, inputs[m.arg][np.maximum(t, 0)], 0.0)
            args.append(v)
        res = fn(*args)
        res = res if isinstance(res, (list, tuple)) else [res]
        for m, r in zip(comp.stage_out, res):
            vals = np.asarray(r.nonzeros(), np.float64).reshape(-1) if hasattr(r, "nonzeros") else np.asarray(r).reshape(-1)
            t = m.table[s]
            keep = t >= 0
            dst = scratch if m.arg < 0 else outs[m.arg]
            dst[t[keep]] = vals[keep]
    for r in comp.reduces:
        out = outs[r.dst]
        for g in range(out.size):
            acc = 0.0
            for j in range(r.segptr[g], r.segptr[g + 1]):
                acc += scratch[r.segidx[j]]
            out[g] = acc
    return outs
