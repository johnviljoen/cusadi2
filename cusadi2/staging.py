"""Stage structure from a CasADi MX graph: what a stage-wise (composed) cusadi2 problem needs, derived
from the whole-horizon expressions instead of written by hand.

    st = cusadi2.staging.derive(x, p, sums={"f": f}, rows={"c": c, "d": d})
    st.specs, st.functions["f"], st.active["c"], st.user_rows["d"], ...

x is a list of MX symbols, one per variable, each (row size, horizon length): column k is knot k
(length N + 1) or interval k (length N). p is a list of MX symbols of any shape: a p leaf that a map
reads row by row (a reference trajectory, (row size, N or N + 1)) is a per-stage parameter, staged
like x (parameter stage arguments "xr", "xr+1"); the other leaves together are the one shared
parameter "p". `sums` are scalar outputs summed over the stages (an objective), `rows` vector
outputs whose rows are distributed over the stages (constraints).

Per-stage work is a call of a casadi.Function: `F.map(n)(...)` (one call node of a Map Function), or
one F called once per stage in a loop (n call nodes of the same Function). Everything else is glue
(x[:, 0] - x_hat, a terminal cost on x[:, N]); a Function called only once is glue too and is inlined.
JAX code reaches this through `cusadi2.jax.to_mx`, which turns lax.map / vmap into `F.map(n)`.

How. The graph's call nodes are found by walking its instructions. Each map call's outputs are
replaced by placeholder symbols; the call inputs and the remaining graph are then evaluated on one SX
symbol per x / p element, which gives every input element's provenance (which (variable, horizon
index) it reads) whatever MX operations produced it. Each map body is evaluated once, on stage
arguments.

Stage rule. Stage k is interval k with its end knot k + 1 (stage 0 also owns knot 0). A map iteration
(or a glue row / term) goes to the smallest interval index it reads, or, when it reads only knots, to
max(largest knot - 1, 0); a map that reads only knots and would put two iterations on one stage that
way (a map over the start knots, x[:-1]) goes by its smallest knot instead (knot k at stage k). Every
read is then at offset index - stage >= 0, and that offset, fixed per
element across a map's iterations, picks the stage argument (variable, offset): "q" (offset 0), an
alias "q+1" (offset 1), ...

Outputs. A `rows` entry is either a map output element or glue (no map output); rows are grouped into
blocks (a map output, or a run of glue rows on one stage) in order of first appearance, which gives
the stage output and its active mask (N, rows). A `sums` entry must be affine in the map outputs with
numeric weights; the remainder is split into additive terms, each placed like a glue row. Weights and
glue terms that do not hold at every stage are gated by per-stage constants (spec "consts").

The result (`Staging`) is what cusadi2's composition takes: specs (see cusadi2.compose.StageLayout),
one stage Function per output on the stage arguments in spec order, the active masks, and per output
the map user row -> (stage, row of the stage output). No CSE here: cusadi2.derivatives and compile
apply it (default on).

Not supported (StagingError): scans / maps with a carry (sequential), reduced map outputs
(`map(..., reduce_out)`), a map input built from another map's output, an unmapped map input that
depends on x, two iterations of one map on one stage.
A Function called once is inlined as glue, so a helper that wraps the whole horizon (maps inside)
loses the structure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import casadi as ca
import numpy as np



class StagingError(ValueError):
    """The graph has no stage structure this derivation understands."""


class NotStaged(StagingError):
    """The graph has no stage maps (no Map call, no Function called more than once)."""


@dataclass
class Staging:
    N: int                                   # stages
    specs: list                              # stage arguments (cusadi2.compose.StageLayout specs)
    functions: Dict[str, ca.Function]        # output name -> stage Function of the spec arguments
    active: Dict[str, np.ndarray]            # rows output name -> (N, stage rows) bool
    user_rows: Dict[str, list]               # rows output name -> [(stage, stage row)] per user row
    leaves: List[Tuple[str, tuple]]          # x variables: (name, (length, row size))
    p_size: int                              # the shared parameter's size
    n_maps: int
    p_leaves: list = field(default=None)     # per p leaf (user order): (name, staged?, (length, row size) or size)
    consts: np.ndarray = field(default=None)  # (N, n_consts), also in specs


_MAP = re.compile(r"^map(\d+)_")


def _is_sym(a) -> bool:
    return isinstance(a, np.ndarray) and a.dtype == object


def sx_inputs(name: str, shape) -> tuple:
    """One SX symbol vector of size prod(shape) and its row-major object-array view."""
    n = int(np.prod(shape, dtype=np.int64)) if len(shape) else 1
    sym = ca.SX.sym(name, n)
    arr = np.empty(n, dtype=object)
    for i in range(n):
        arr[i] = sym[i]
    return sym, arr.reshape(shape)


def _names(arr) -> List[str]:
    return [e.name() if isinstance(e, ca.SX) and e.is_symbolic() else None for e in np.asarray(arr).reshape(-1)]


class _Stager:
    def __init__(self, x_leaves, x_names, p_size, params=None):
        lens = sorted({int(np.shape(l)[0]) for l in x_leaves})
        if len(lens) == 2 and lens[1] == lens[0] + 1:
            self.N = lens[0]
        elif len(lens) == 1:
            self.N = lens[0]
        else:
            raise StagingError(f"x leaves need leading lengths N and N + 1, got {lens}")
        self.leaves = []                       # (name, shape, knot?)
        self.param = list(params) if params is not None else [False] * len(x_leaves)   # per-stage parameter leaf?
        self.lookup: Dict[str, Tuple[int, int, int]] = {}   # x symbol name -> (leaf, row, col)
        self.x_syms, self.x_arrs = [], []
        for i, (l, nm) in enumerate(zip(x_leaves, x_names)):
            shape = tuple(np.shape(l))
            knot = len(lens) == 2 and shape[0] == self.N + 1
            self.leaves.append((nm, shape, knot))
            sym, arr = sx_inputs(f"PS{i}" if self.param[i] else f"X{i}", shape)
            self.x_syms.append(sym)
            self.x_arrs.append(arr)
            rowsz = int(np.prod(shape[1:], dtype=np.int64))
            for j, n in enumerate(_names(arr)):
                self.lookup[n] = (i, j // rowsz, j % rowsz)
        self.p_sym, self.p_arr = sx_inputs("P", (p_size,))
        self.p_names = set(_names(self.p_arr))
        self.stage_args: Dict[Tuple[int, int], Tuple[ca.SX, np.ndarray]] = {}   # (leaf, offset) -> (sym, flat arr)
        self.const_blocks: List[Tuple[str, np.ndarray, ca.SX]] = []            # (name, values (N, size), sym)
        self.maps: List[dict] = []
        self.ph: Dict[str, Tuple[int, int, int, int]] = {}                      # placeholder -> (map, out, t, elem)

    # -- stage arguments and constants
    def stage_arg(self, leaf, offset):
        if (leaf, offset) not in self.stage_args:
            nm, shape, _ = self.leaves[leaf]
            sym, arr = sx_inputs(f"{nm}+{offset}" if offset else nm, (int(np.prod(shape[1:], dtype=np.int64)),))
            self.stage_args[(leaf, offset)] = (sym, arr)
        return self.stage_args[(leaf, offset)][1]

    def const(self, name, values):
        values = np.asarray(values, np.float64).reshape(self.N, -1)
        sym, arr = sx_inputs(f"k_{name}", (values.shape[1],))
        self.const_blocks.append((name, values, sym))
        return arr

    def stage_of(self, reads, what):
        """reads: list of (leaf, row); the stage and per-read offsets (see module docstring)."""
        if not reads:
            raise StagingError(f"{what} reads no x; cannot place it at a stage")
        interval = [r for l, r in reads if not self.leaves[l][2]]
        knots = [r for l, r in reads if self.leaves[l][2]]
        s = min(interval) if interval else max(max(knots) - 1, 0)
        if any(r - s < 0 for _, r in reads):
            raise StagingError(f"{what}: reads a row before its stage {s} (reads {sorted(set(reads))})")
        return s

    def to_stage(self, expr: ca.SX, what):
        """Glue expression on x / p symbols -> (stage, the expression on stage arguments)."""
        syms = ca.symvar(expr)
        reads, old, new = [], [], []
        for v in syms:
            n = v.name()
            if n in self.p_names:
                continue
            if n not in self.lookup:
                raise StagingError(f"{what} depends on {n}, which is neither x nor p")
            reads.append(self.lookup[n][:2])
        s = self.stage_of(reads, what)
        for v in syms:
            n = v.name()
            if n in self.lookup:
                l, r, c = self.lookup[n]
                old.append(v)
                new.append(self.stage_arg(l, r - s)[c])
        if old:
            expr = ca.substitute(expr, ca.vertcat(*old), ca.vertcat(*new))
        return s, expr

    def stage_map(self, consts, xs, n, body_eval):
        """A map of n iterations: consts are passed whole to every iteration, xs[j] has leading axis n
        (object arrays of x / p symbols or numbers). body_eval(body_in) -> list of output arrays.
        Evaluates the body once on stage arguments; returns per-output placeholder arrays (n, ...)."""
        m = len(self.maps)
        for a in consts:
            if _is_sym(a) and any(nm in self.lookup for nm in _names(a) if nm):
                raise StagingError(f"map {m}: an unmapped argument depends on x (the whole horizon in every stage)")
        # provenance of every mapped element, per iteration
        reads_t = [[] for _ in range(n)]
        for a in xs:
            a = np.asarray(a)
            if _is_sym(a):
                for t in range(n):
                    for nm in _names(a[t]):
                        if nm in self.lookup:
                            reads_t[t].append(self.lookup[nm][:2])
        stages = [self.stage_of(reads_t[t], f"map {m} iteration {t}") for t in range(n)]
        knot_only = all(reads_t[t] and all(self.leaves[l][2] for l, _ in reads_t[t]) for t in range(n))
        if len(set(stages)) != n and knot_only:
            # a map over start knots (x[:-1]: knots 0 .. N-1): knot k at stage k instead of k - 1
            alt = [min(r for _, r in reads_t[t]) for t in range(n)]
            if len(set(alt)) == n and max(alt) < self.N:
                stages = alt
        if len(set(stages)) != n:
            raise StagingError(f"map {m}: two iterations fall on one stage ({stages})")
        # body inputs on stage arguments / constants
        body_in = list(consts)
        for j, a in enumerate(xs):
            a = np.asarray(a)
            shape = a.shape[1:]
            flat = a.reshape(n, -1)
            out = np.empty(flat.shape[1], dtype=object)
            const_cols, const_vals = [], []
            for e in range(flat.shape[1]):
                col = flat[:, e]
                nms = _names(col) if _is_sym(a) else [None] * n
                if all(nm is not None and nm in self.lookup for nm in nms):
                    lrc = [self.lookup[nm] for nm in nms]
                    offs = {(l, r - stages[t], c) for t, (l, r, c) in enumerate(lrc)}
                    if len(offs) != 1:
                        raise StagingError(f"map {m} operand {j} element {e}: reads {sorted(offs)} across iterations")
                    l, o, c = offs.pop()
                    out[e] = self.stage_arg(l, o)[c]
                elif all(nm is not None and nm in self.p_names for nm in nms) and len(set(nms)) == 1:
                    out[e] = col[0]
                elif not _is_sym(np.asarray(col, dtype=object)) or all(not isinstance(v, ca.SX) for v in col):
                    vals = np.asarray(col, np.float64)
                    if np.all(vals == vals[0]):
                        out[e] = float(vals[0])
                    else:
                        const_cols.append(e)
                        const_vals.append(vals)
                else:
                    raise StagingError(f"map {m} operand {j} element {e}: neither a slice of x, a parameter nor a constant")
            if const_cols:
                full = np.zeros((self.N, len(const_cols)))
                full[stages] = np.stack(const_vals, axis=1)
                ks = self.const(f"m{m}_in{j}", full)
                for i, e in enumerate(const_cols):
                    out[e] = ks[i]
            body_in.append(out.reshape(shape))
        body_out = body_eval(body_in)
        rec = dict(stages=stages, n=n, outs=[])
        results = []
        for k, y in enumerate(body_out):
            y = np.asarray(y)
            rec["outs"].append([ca.SX(v) if isinstance(v, ca.SX) else ca.SX(float(v)) for v in y.reshape(-1)])
            ph_sym, ph_arr = sx_inputs(f"ph{m}_{k}", (n,) + y.shape)
            for t in range(n):
                for e, nm in enumerate(_names(ph_arr[t])):
                    self.ph[nm] = (m, k, t, e)
            results.append(ph_arr)
        self.maps.append(rec)
        return results



def _terms(e: ca.SX, sign=1.0):
    """Additive terms of a scalar SX expression."""
    if e.is_constant():
        return [(sign, e)] if float(e) != 0.0 else []
    op = e.op() if not e.is_symbolic() else None
    if op == ca.OP_ADD:
        return _terms(e.dep(0), sign) + _terms(e.dep(1), sign)
    if op == ca.OP_SUB:
        return _terms(e.dep(0), sign) + _terms(e.dep(1), -sign)
    if op == ca.OP_NEG:
        return _terms(e.dep(0), -sign)
    return [(sign, e)]


def _finish(S: _Stager, sums: Dict[str, object], rows_out: Dict[str, object]) -> "Staging":
    """Top-level outputs (SX in x / p symbols and map placeholders) -> blocks, specs, stage Functions."""
    N = S.N

    def body_expr(m, k, e):
        return S.maps[m]["outs"][k][e]

    # ---- c, d: blocks in order of first appearance
    def rows(out, what):
        out = np.asarray(out, dtype=object).reshape(-1)
        blocks, where, user = [], {}, []     # block: dict(key, rows (exprs), active (N, rows))
        glue_open = None                     # index of the glue block the previous row went to
        for u, v in enumerate(out):
            v = v if isinstance(v, ca.SX) else ca.SX(float(v))
            nm = v.name() if v.is_symbolic() else None
            if nm is not None and nm in S.ph:
                glue_open = None
                m, k, t, e = S.ph[nm]
                key = ("map", m, k)
                if key not in where:
                    size = len(S.maps[m]["outs"][k])
                    where[key] = len(blocks)
                    blocks.append(dict(key=key, rows=[body_expr(m, k, i) for i in range(size)],
                                       active=np.zeros((N, size), bool)))
                b = blocks[where[key]]
                s = S.maps[m]["stages"][t]
                if b["active"][s, e]:
                    raise StagingError(f"{what} row {u}: map {m} output {k} element {e} used twice")
                b["active"][s, e] = True
                user.append((where[key], s, e))
                continue
            if any(sv.name() in S.ph for sv in ca.symvar(v)):
                raise StagingError(f"{what} row {u} mixes a map output with other terms; compute it inside the map")
            s, ex = S.to_stage(v, f"{what} row {u}")
            if glue_open is None or blocks[glue_open]["key"][1] != s:
                blocks.append(dict(key=("glue", s, u), rows=[], active=np.zeros((N, 0), bool)))
                glue_open = len(blocks) - 1
            b = blocks[glue_open]
            b["rows"].append(ex)
            col = np.zeros((N, 1), bool)
            col[s, 0] = True
            b["active"] = np.concatenate([b["active"], col], axis=1)
            user.append((glue_open, s, len(b["rows"]) - 1))
        blocks, user = _merge_glue(S, blocks, user, what)
        # stage output: blocks concatenated; user row u -> (stage, row of the stage output)
        offs = np.cumsum([0] + [len(b["rows"]) for b in blocks])
        exprs = [r for b in blocks for r in b["rows"]]
        active = np.concatenate([b["active"] for b in blocks], axis=1) if blocks else np.zeros((N, 0), bool)
        user_rows = [(s, int(offs[bi] + e)) for bi, s, e in user]
        return exprs, active, user_rows

    row_res = {name: rows(out, name) for name, out in rows_out.items()}
    sum_res = {name: _sum(S, name, out, body_expr) for name, out in sums.items()}
    return _specs(S, sum_res, row_res)


def _glue_template(S: _Stager, rows, gen):
    """A glue block's rows with their p symbols replaced by the generic symbols gen[0], gen[1], ... (in
    order of appearance) -> (key comparing the structure, the template rows, the p symbols replaced)."""
    e = ca.vertcat(*rows)
    ps = [v for v in ca.symvar(e) if v.name() in S.p_names]
    while len(gen) < len(ps):
        gen.append(ca.SX.sym(f"g{len(gen)}"))
    t = ca.substitute(e, ca.vertcat(*ps), ca.vertcat(*gen[:len(ps)])) if ps else e
    ins = sorted(ca.symvar(t), key=lambda v: v.name())
    key = (len(rows), tuple(v.name() for v in ins),
           ca.Function("t", [ca.vertcat(*ins)] if ins else [], [t]).serialize())
    return key, [t[i] for i in range(t.numel())], ps


def _merge_glue(S: _Stager, blocks, user, what):
    """Glue blocks at different stages that are the same expression up to which p entries they read (a
    pin of a foot at every landing event, each to its own target) -> one block: the template once, its
    p entries selected per stage by one-hot stage constants. Otherwise every stage would compute every
    such block (masked to its one stage). Returns the new blocks and the user rows renumbered."""
    gen, groups = [], {}                     # key -> list of [member block indices] with distinct stages
    tmpl = {}
    for bi, b in enumerate(blocks):
        if b["key"][0] != "glue":
            continue
        key, t, ps = _glue_template(S, b["rows"], gen)
        tmpl[bi] = (t, ps)
        for grp in groups.setdefault(key, []):
            if all(blocks[j]["key"][1] != b["key"][1] for j in grp):
                grp.append(bi)
                break
        else:
            groups[key].append([bi])
    merged = {}                                # first member -> group, for groups of 2 or more
    for grps in groups.values():
        for grp in grps:
            if len(grp) > 1:
                merged[grp[0]] = grp
    if not merged:
        return blocks, user
    into = {}                                  # old block -> new block index
    out = []
    for bi, b in enumerate(blocks):
        if bi in into:
            continue
        if bi not in merged:
            into[bi] = len(out)
            out.append(b)
            continue
        grp = merged[bi]
        t, _ = tmpl[bi]
        npar = len(tmpl[bi][1])
        onehot = np.zeros((S.N, len(grp)))
        for m, j in enumerate(grp):
            onehot[blocks[j]["key"][1], m] = 1.0
        k = None
        sel = []
        for i in range(npar):
            cand = [tmpl[j][1][i] for j in grp]
            if all(ca.is_equal(c, cand[0]) for c in cand[1:]):
                sel.append(cand[0])
                continue
            if k is None:
                k = S.const(f"{what}_glue{bi}_sel", onehot)
            sel.append(sum((k[m] * c for m, c in enumerate(cand[1:], 1)), k[0] * cand[0]))
        rows = ca.substitute(ca.vertcat(*t), ca.vertcat(*gen[:npar]), ca.vertcat(*sel)) if npar else ca.vertcat(*t)
        active = np.zeros_like(b["active"])
        for j in grp:
            active |= blocks[j]["active"]
            into[j] = len(out)
        out.append(dict(key=("glue", "merged", bi), rows=[rows[i] for i in range(rows.numel())], active=active))
    return out, [(into[bi], s, e) for bi, s, e in user]


def _sum(S: _Stager, what: str, f_out, body_expr) -> ca.SX:
    """A scalar output: affine in the placeholders, the rest split into terms by stage."""
    N = S.N
    f = np.asarray(f_out, dtype=object).reshape(-1)[0]
    f = f if isinstance(f, ca.SX) else ca.SX(float(f))
    phs = [v for v in ca.symvar(f) if v.name() in S.ph]
    f_terms = []
    if phs:
        PH = ca.vertcat(*phs)
        g = ca.jacobian(f, PH)
        if ca.depends_on(g, ca.vertcat(PH, *S.x_syms, S.p_sym)):
            raise StagingError(f"{what} must be a weighted sum of map outputs plus glue (numeric weights)")
        g = np.asarray(ca.evalf(g)).reshape(-1)
        coef: Dict[Tuple[int, int, int], np.ndarray] = {}
        for v, w in zip(phs, g):
            m, k, t, e = S.ph[v.name()]
            c_ = coef.setdefault((m, k, e), np.zeros(N))
            c_[S.maps[m]["stages"][t]] += w
        for (m, k, e), w in coef.items():
            ex = body_expr(m, k, e)
            if np.all(w == w[0]):
                f_terms.append(ex if w[0] == 1.0 else w[0] * ex)
            else:
                f_terms.append(S.const(f"{what}_m{m}_{k}_{e}", w)[0] * ex)
        f = ca.substitute(f, PH, ca.DM.zeros(PH.numel()))
    glue_by_stage: Dict[int, List[ca.SX]] = {}
    for sign, t in _terms(f):
        if not ca.depends_on(t, ca.vertcat(*S.x_syms)):
            s, ex = 0, t
        else:
            s, ex = S.to_stage(t, f"a term of {what}")
        glue_by_stage.setdefault(s, []).append(sign * ex)
    for s, ts in sorted(glue_by_stage.items()):
        gate = np.zeros(N)
        gate[s] = 1.0
        f_terms.append(S.const(f"{what}_glue_s{s}", gate)[0] * sum(ts[1:], ts[0]))
    return sum(f_terms[1:], f_terms[0]) if f_terms else ca.SX(0)


def _specs(S: _Stager, sum_res, row_res) -> "Staging":
    """Specs (one per stage argument used, the shared p, the stage constants) and the stage Functions."""
    N = S.N
    specs, args = [], []
    for li, (nm, shape, knot) in enumerate(S.leaves):
        S.stage_arg(li, 0)
        span = shape[0] - N
        for o in sorted(o for (l, o) in S.stage_args if l == li):
            sym = S.stage_args[(li, o)][0]
            sp = dict(name=nm if o == 0 else f"{nm}+{o}", size=int(sym.numel()), mode="coupled", offset=o, span=span)
            if o:
                sp["alias"] = nm
            if S.param[li]:
                sp["param"] = True
            specs.append(sp)
            args.append(sym)
    if S.p_sym.numel():
        specs.append(dict(name="p", size=int(S.p_sym.numel()), mode="shared", param=True))
        args.append(S.p_sym)
    if S.const_blocks:
        vals = np.concatenate([v for _, v, _ in S.const_blocks], axis=1)
        specs.append(dict(name="consts", size=vals.shape[1], mode="stage", const=vals))
        args.append(ca.vertcat(*[s for _, _, s in S.const_blocks]))
    else:
        vals = np.zeros((N, 0))

    functions, active, user_rows = {}, {}, {}
    for name, y in sum_res.items():
        functions[name] = ca.Function(f"stage_{name}", args, [y])
    for name, (exprs, act, user) in row_res.items():
        functions[name] = ca.Function(f"stage_{name}", args, [ca.vertcat(*exprs) if exprs else ca.SX(0, 1)])
        active[name], user_rows[name] = act, user
    return Staging(N=N, specs=specs, functions=functions, active=active, user_rows=user_rows,
                   leaves=[(nm, tuple(shape)) for (nm, shape, _), prm in zip(S.leaves, S.param) if not prm],
                   p_size=int(S.p_sym.numel()), n_maps=len(S.maps))


def _elems(sx: ca.SX) -> List:
    """Nonzeros of an SX matrix as Python objects: SX scalars, or floats where constant."""
    out = []
    for i in range(sx.nnz()):
        e = sx.nz[i]
        out.append(float(e) if e.is_constant() else e)
    return out


def _to_obj(vals) -> np.ndarray:
    a = np.empty(len(vals), dtype=object)
    for i, v in enumerate(vals):
        a[i] = v
    return a


def _same(a, b) -> bool:
    """Element lists equal (SX by identity / structure, numbers by value)."""
    return len(a) == len(b) and all(
        (isinstance(u, ca.SX) and isinstance(v, ca.SX) and ca.is_equal(u, v)) or
        (not isinstance(u, ca.SX) and not isinstance(v, ca.SX) and u == v) for u, v in zip(a, b))


def _mat(rows, n) -> np.ndarray:
    """(n, len) object array from n equal-length lists (numpy must not look inside the SX scalars)."""
    w = len(rows[0]) if rows else 0
    a = np.empty((n, w), dtype=object)
    for t, r in enumerate(rows):
        for e, v in enumerate(r):
            a[t, e] = v
    return a


def _sx(sp: ca.Sparsity, vals) -> ca.SX:
    nz = [v if isinstance(v, ca.SX) else ca.SX(float(v)) for v in vals]
    return ca.SX(sp, ca.vertcat(*nz)) if nz else ca.SX(sp)


def _calls(G: ca.Function):
    """Call nodes of G's top level: list of (MX call node, Function)."""
    out = []
    for k in range(G.n_instructions()):
        if G.instruction_id(k) == ca.OP_CALL:
            node = G.instruction_MX(k)
            out.append((node, node.which_function()))
    return out


def derive(x: Sequence[ca.MX], p: Sequence[ca.MX], sums: Dict[str, ca.MX], rows: Dict[str, ca.MX], *,
           x_names: Sequence[str] = None) -> Staging:
    """The stage structure of the outputs `sums` (scalars) and `rows` (vectors), MX expressions of the
    MX symbols x (variables, (row size, horizon length) each) and p (parameters); see the module
    docstring. Raises NotStaged if the graph has no stage maps, StagingError if it cannot be staged."""
    x, p = list(x), list(p)
    x_names = list(x_names) if x_names else [v.name() for v in x]
    for v in x + p:
        if not v.is_symbolic():
            raise StagingError("x and p must be MX symbols")
    sum_names, row_names = list(sums), list(rows)
    outs = [ca.MX(sums[n]) for n in sum_names] + [ca.vec(ca.MX(rows[n])) for n in row_names]
    for n in sum_names:
        if ca.MX(sums[n]).numel() != 1:
            raise StagingError(f"{n} must be a scalar")
    G = ca.Function("G", x + p, outs)

    # ---- stage maps: Map calls, and Functions called more than once
    calls = _calls(G)
    count: Dict[str, int] = {}
    for _, fn in calls:
        count[fn.name()] = count.get(fn.name(), 0) + 1
    groups: Dict[str, dict] = {}                   # name -> dict(body, map n or None, nodes)
    for node, fn in calls:
        nm = fn.name()
        if _MAP.match(nm) and fn.class_name() in ("Map", "OmpMap", "ThreadMap"):
            body = fn.get_function(fn.get_function()[0])
            groups[f"{nm}#{len(groups)}"] = dict(body=body, n=int(_MAP.match(nm).group(1)), map=fn, nodes=[node])
        elif count[nm] > 1:
            groups.setdefault(nm, dict(body=fn, n=None, map=None, nodes=[]))["nodes"].append(node)
    if not groups:
        raise NotStaged("no stage maps: no F.map(n) call and no Function called more than once")
    # per-stage parameters: p leaves some map reads row by row (different elements per iteration)
    staged_p = _staged_parameters(x, p, groups)
    p_names = [v.name() for v in p]
    # the stager wants row-major (length, row size) leaves: the same flat order as CasADi's (row size, length)
    leaves = list(x) + [v for v, s in zip(p, staged_p) if s]
    S = _Stager([np.zeros((v.size2(), v.size1())) for v in leaves], x_names + [n for n, s in zip(p_names, staged_p) if s],
                int(sum(v.numel() for v, s in zip(p, staged_p) if not s)), params=[False] * len(x) + [True] * sum(staged_p))
    x_sx = [ca.reshape(s, v.size1(), v.size2()) for s, v in zip(S.x_syms[:len(x)], x)]
    p_sx, at, k = [], 0, len(x)
    for v, s in zip(p, staged_p):
        if s:
            p_sx.append(ca.reshape(S.x_syms[k], v.size1(), v.size2()))
            k += 1
        else:
            p_sx.append(ca.reshape(S.p_sym[at:at + v.numel()], v.size1(), v.size2()))
            at += v.numel()
    # replace every map call output by a placeholder symbol
    old, new, ph_mx = [], [], []
    for gi, (key, g) in enumerate(groups.items()):
        g["ph"] = []
        for ci, node in enumerate(g["nodes"]):
            row = []
            for k in range(node.n_out()):
                o = node.get_output(k)
                s = ca.MX.sym(f"ph{gi}_{ci}_{k}", o.sparsity())
                old.append(o); new.append(s); row.append(s)
            g["ph"].append(row)
            ph_mx += row
    sub = ca.graph_substitute(outs, old, new) if old else outs
    every = x + p + ph_mx
    for key, g in groups.items():
        for node in g["nodes"]:
            ins = [node.dep(j) for j in range(node.n_dep())]
            ins = ca.graph_substitute(ins, old, new) if old else ins
            if ph_mx and any(ca.depends_on(a, ca.veccat(*ph_mx)) for a in ins if a.numel()):
                raise StagingError(f"{key}: an input depends on another map's output; compose them in one Function")

    # ---- evaluate: map inputs (per group), then the body once on stage arguments, then the glue
    x_p_sx = x_sx + p_sx
    ph_sx: Dict[int, ca.SX] = {}                   # id in ph_mx -> SX value (placeholder symbols)
    for key, g in groups.items():
        body, nodes = g["body"], g["nodes"]
        nin, nout = body.n_in(), body.n_out()
        if g["map"] is not None:
            fm, n = g["map"], g["n"]
            for j in range(nout):
                if fm.size2_out(j) != n * body.size2_out(j):
                    raise StagingError(f"{key}: output {j} is reduced over the map; only plain maps are stages")
            ins = [nodes[0].dep(j) for j in range(nin)]
            vals = ca.Function("ins", x + p, ins).call(x_p_sx)
            consts, xs, mapped = [], [], []
            for j, v in enumerate(vals):
                bsp = body.sparsity_in(j)
                v = ca.densify(v)
                if v.size2() == bsp.size2() and n > 1:   # not mapped (repeated to every iteration)
                    consts.append(_to_obj(_elems(ca.project(v, bsp))))
                    mapped.append(False)
                    continue
                w = bsp.size2()
                per = [_elems(ca.project(v[:, t * w:(t + 1) * w], bsp)) for t in range(n)]
                if n > 1 and per[0] and all(_same(r, per[0]) for r in per[1:]) and any(
                        isinstance(e, ca.SX) and ca.depends_on(e, ca.vertcat(*S.x_syms)) for e in per[0]):
                    raise StagingError(f"{key}: input {j} is the same x slice in every iteration (an unmapped "
                                       "input that depends on x); map it per stage or move it out of the map")
                xs.append(_mat(per, n))
                mapped.append(True)
        else:
            n = len(nodes)
            consts, mapped = [], [True] * nin
            per_call = [ca.Function("ins", x + p, [nd.dep(j) for j in range(nin)]).call(x_p_sx) for nd in nodes]
            xs = []
            for j in range(nin):
                bsp = body.sparsity_in(j)
                xs.append(_mat([_elems(ca.project(ca.densify(pc[j]), bsp)) for pc in per_call], n))
        for a in consts:
            if any(isinstance(e, ca.SX) and ca.depends_on(e, ca.vertcat(*S.x_syms)) for e in a):
                raise StagingError(f"{key}: an unmapped input depends on x (the whole horizon in every stage)")

        def body_eval(body_in, body=body, mapped=mapped):
            it_c, it_x = iter(body_in[:len(consts)]), iter(body_in[len(consts):])
            args = [_sx(body.sparsity_in(j), next(it_x) if mapped[j] else next(it_c)) for j in range(nin)]
            return [_to_obj(_elems(y)) for y in body.call(args)]
        phs = S.stage_map(consts, xs, n, body_eval)     # per output k: (n, nnz_k) placeholder SX scalars
        if g["map"] is not None:
            for k in range(nout):
                ph_sx[id(g["ph"][0][k])] = _sx(g["ph"][0][k].sparsity(), list(np.asarray(phs[k]).reshape(-1)))
        else:
            for t in range(n):
                for k in range(nout):
                    ph_sx[id(g["ph"][t][k])] = _sx(g["ph"][t][k].sparsity(), list(np.asarray(phs[k])[t]))
    H = ca.Function("glue", every, sub)
    vals = H.call(x_p_sx + [ph_sx[id(s)] for s in ph_mx])
    k = len(sum_names)
    st = _finish(S, {n: _to_obj([ca.densify(v)[0]]) for n, v in zip(sum_names, vals[:k])},
                 {n: _to_obj(_elems_all(v)) for n, v in zip(row_names, vals[k:])})
    st.consts = next((sp["const"] for sp in st.specs if sp["name"] == "consts"), np.zeros((S.N, 0)))
    st.p_leaves = [(n, bool(s), (v.size2(), v.size1()) if s else int(v.numel())) for n, s, v in zip(p_names, staged_p, p)]
    return st


def _staged_parameters(x, p, groups) -> List[bool]:
    """Per p leaf: does some map read it row by row (an operand element that is a symbol of this leaf
    and differs between iterations)? Such leaves are per-stage parameters, staged like x."""
    if not p:
        return []
    tx = [ca.SX.sym(f"tx{i}", v.size1(), v.size2()) for i, v in enumerate(x)]
    tp = [ca.SX.sym(f"tp{i}", v.size1(), v.size2()) for i, v in enumerate(p)]
    owner = {tp[i].nz[j].name(): i for i in range(len(p)) for j in range(tp[i].nnz())}
    staged = [False] * len(p)

    def visit(columns):                        # columns: per iteration, the operand's elements
        for elems in zip(*columns):
            names = [e.name() if isinstance(e, ca.SX) and e.is_symbolic() else None for e in elems]
            if all(n in owner for n in names) and len(set(names)) > 1 and len({owner[n] for n in names}) == 1:
                staged[owner[names[0]]] = True
    for g in groups.values():
        body, nodes = g["body"], g["nodes"]
        nin = body.n_in()
        if g["map"] is not None:
            n = g["n"]
            vals = ca.Function("ins", list(x) + list(p), [nodes[0].dep(j) for j in range(nin)]).call(tx + tp)
            for j, v in enumerate(vals):
                w = body.sparsity_in(j).size2()
                v = ca.densify(v)
                if v.size2() == w * n:
                    visit([_elems(v[:, t * w:(t + 1) * w]) for t in range(n)])
        else:
            per_call = [ca.Function("ins", list(x) + list(p), [nd.dep(j) for j in range(nin)]).call(tx + tp) for nd in nodes]
            for j in range(nin):
                visit([_elems(ca.densify(pc[j])) for pc in per_call])
    return staged


def _elems_all(sx: ca.SX) -> List:
    """Every element (dense, column-major) of an SX column: SX scalars, or floats where constant."""
    return _elems(ca.densify(sx))
