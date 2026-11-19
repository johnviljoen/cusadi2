"""JAX -> CasADi MX, keeping the stage structure JAX states: the front end of `cusadi2.staging`.

    g = cusadi2.jax.to_mx(fun, x, p)        # fun(x, p) -> tuple of arrays; x, p example pytrees
    g.x, g.p, g.outs                        # MX symbols and the outputs (each flattened, row-major)

Every top-level `lax.map` (a carry-free `scan`) or `jax.vmap` (traced as a lax.map, `vmap_as_map`)
becomes `F.map(n)` with F the body as an SX `casadi.Function` (traced by `cusadi2.jax.interpret`).
The map inputs are nonzero gathers of the x leaves where they are plain slices / gathers of x (else
an SX helper Function call); the rest of the top level (the glue) is one SX Function call on x, p and
the map outputs. Nothing about stages is decided here: `cusadi2.staging.derive` reads them off the graph,
exactly as for a hand-written CasADi graph. Without any map the graph is a single glue call (a flat
problem).

Layout: an x leaf of JAX shape (L, *rest) is an MX symbol (prod(rest), L), column k = horizon index k
(the same flat order as JAX's row-major array); a 0-d / 1-d leaf of size n is (n, 1). All of p is one
MX column (its leaves flattened row-major, in tree order).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, List

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.core import Literal

from ..staging import StagingError
from .interpret import RULES, _arr, _bind_constants, _is_sym, eval_jaxpr, sx_inputs, to_sx_vector


def _names(arr) -> List[str]:
    return [e.name() if isinstance(e, ca.SX) and e.is_symbolic() else None for e in np.asarray(arr).reshape(-1)]


# ----------------------------------------------------------------------------- vmap -> lax.map
@contextlib.contextmanager
def vmap_as_map():
    """While tracing: every top-level jax.vmap is traced as lax.map over its mapped arguments (the
    body stays one sub-jaxpr instead of batched primitives). Nested vmaps (inside a map body, e.g.
    in library code) keep JAX's own batching. Works however vmap was imported: it replaces
    jax._src.interpreters.batching.batch, which jax.vmap looks up at call time."""
    from jax._src.interpreters import batching
    orig = batching.batch
    depth = [0]

    class _AsMap:
        def __init__(self, fun, axis_data, in_dims, out_dim_dests, sum_match=False):
            self.fun, self.in_dims, self.out_dim_dests = fun, in_dims, out_dim_dests

        def call_wrapped(self, *args):
            dims = self.in_dims() if callable(self.in_dims) else self.in_dims
            if any(d is not None and not isinstance(d, int) for d in dims):
                raise StagingError("vmap_as_map: only integer / None in_axes")
            mapped = [jnp.moveaxis(a, d, 0) for a, d in zip(args, dims) if d is not None]

            def body(xs):
                it = iter(xs)
                full = [next(it) if d is not None else a for a, d in zip(args, dims)]
                depth[0] += 1
                try:
                    return list(self.fun.call_wrapped(*full))
                finally:
                    depth[0] -= 1
            outs = lax.map(body, mapped)
            dests = self.out_dim_dests() if callable(self.out_dim_dests) else self.out_dim_dests
            if any(d is None or not isinstance(d, int) for d in dests):
                raise StagingError("vmap_as_map: only integer out_axes")
            return [jnp.moveaxis(o, 0, d) for o, d in zip(outs, dests)], list(dests)

    def batch(fun, axis_data, in_dims, out_dim_dests, sum_match=False):
        if depth[0] > 0:
            return orig(fun, axis_data, in_dims, out_dim_dests, sum_match=sum_match)
        return _AsMap(fun, axis_data, in_dims, out_dim_dests, sum_match)

    batching.batch = batch
    try:
        yield
    finally:
        batching.batch = orig



# ----------------------------------------------------------------------------- translator
class _Translator:
    def __init__(self, x_syms, p_sym, x_mx, p_mx):
        self.where: Dict[str, tuple] = {}            # SX symbol name -> (MX leaf vector, index)
        for s, m in zip(x_syms + [p_sym], x_mx + [p_mx]):
            for i in range(s.numel()):
                self.where[s[i].name()] = (m, i)
        self.x_syms, self.p_sym = x_syms, p_sym
        self.x_mx, self.p_mx = x_mx, p_mx
        self.maps: List[dict] = []                   # dict(F, n, ins (MX), ph (SX sym), out sizes)

    def to_mx(self, arr) -> ca.MX:
        """Object / numeric array (row-major) -> MX column: a nonzero gather when every element is an
        element of one x leaf or p, a constant when numeric, else a call of an SX helper Function."""
        flat = np.asarray(arr).reshape(-1)
        if not _is_sym(np.asarray(arr)):
            return ca.MX(ca.DM(np.asarray(flat, np.float64)))
        nms = _names(flat)
        if all(nm is not None and nm in self.where for nm in nms):
            leaves = {id(self.where[nm][0]) for nm in nms}
            if len(leaves) == 1:
                leaf = self.where[nms[0]][0]
                return leaf.nz[[self.where[nm][1] for nm in nms]]
        args = self.x_syms + [self.p_sym]
        h = ca.Function(f"in{len(self.maps)}", args, [to_sx_vector(flat)])
        return h.call(self.x_mx + [self.p_mx])[0]

    def scan(self, ins, eqn):
        p = eqn.params
        if "num_consts" in p:
            nc, nk = p["num_consts"], p["num_carry"]
            consts, carry, xs = list(ins[:nc]), list(ins[nc:nc + nk]), list(ins[nc + nk:])
        else:
            consts, carry, xs = [list(t) for t in p["ft_in"].update(ins).unpack()]
        if carry:
            raise StagingError("a scan with a carry is sequential, not a stage map; use lax.map / vmap")
        if p["reverse"]:
            raise StagingError("reverse scan")
        n, m = p["length"], len(self.maps)
        # the body as an SX Function of (consts..., one iteration of xs...), all flat (row-major)
        syms, arrs = [], []
        for j, a in enumerate(consts):
            s, ar = sx_inputs(f"c{j}", np.shape(a)); syms.append(s); arrs.append(ar)
        for j, a in enumerate(xs):
            s, ar = sx_inputs(f"x{j}", np.shape(a)[1:]); syms.append(s); arrs.append(ar)
        outs = eval_jaxpr(p["jaxpr"], arrs)
        F = ca.Function(f"stage{m}", syms, [to_sx_vector(o) for o in outs])
        # map inputs: consts whole (repeated), xs as (size, n): column t = iteration t
        mx_in = [self.to_mx(a) for a in consts]
        mx_in += [ca.reshape(self.to_mx(a), int(np.prod(np.shape(a)[1:], dtype=np.int64)), n) for a in xs]
        y = F.map(n)(*mx_in)
        y = list(y) if isinstance(y, (list, tuple)) else [y]
        results, ph = [], []
        for k, o in enumerate(outs):
            s, ar = sx_inputs(f"ph{m}_{k}", (n,) + np.shape(o))
            results.append(ar); ph.append(s)
        self.maps.append(dict(F=F, n=n, ph=ph, y=[ca.vec(v) for v in y]))
        return results

    def run(self, closed, args):
        jaxpr, env = closed.jaxpr, {}

        def read(v):
            return np.asarray(v.val) if isinstance(v, Literal) else env[v]
        for v, c in zip(jaxpr.constvars, closed.consts):
            env[v] = np.asarray(c)
        for v, a in zip(jaxpr.invars, args):
            env[v] = np.asarray(a)
        for eqn in jaxpr.eqns:
            name, ins = eqn.primitive.name, [read(v) for v in eqn.invars]
            if name == "scan":
                out = self.scan(ins, eqn)
            elif name in ("pjit", "jit"):
                out = self.run(eqn.params["jaxpr"], ins)
            elif name in RULES:
                out = RULES[name](ins, eqn)
            elif not any(_is_sym(a) for a in ins):
                out = _bind_constants(eqn, ins)
            else:
                raise StagingError(f"primitive {name!r} has no SX rule")
            out = out if isinstance(out, (list, tuple)) else [out]
            for v, o in zip(eqn.outvars, out):
                env[v] = _arr(o)
        return [read(v) for v in jaxpr.outvars]


@dataclass
class MXGraph:
    x: List[ca.MX]            # one MX symbol per x leaf
    p: List[ca.MX]            # [all of p as one column]
    outs: List[ca.MX]         # fun's outputs, each flattened (row-major) to a column
    x_names: List[str]        # the leaves' tree paths
    x_tree: Any
    p_tree: Any
    x_shapes: List[tuple]     # JAX shapes of the x leaves
    p_shapes: List[tuple]
    n_maps: int


def trace(fun, x, p):
    """The jaxpr of fun(x, p) as to_mx sees it (vmap traced as lax.map, outputs raveled), and a stable
    text of it (the jaxpr printed, plus every constant's bytes) to key caches on: one trace serves
    both the key and the translation. Returns (closed jaxpr, key text)."""
    x_leaves, x_tree = jax.tree_util.tree_flatten(x)
    p_leaves, p_tree = jax.tree_util.tree_flatten(p)

    def flat_fn(*leaves):
        xx = jax.tree_util.tree_unflatten(x_tree, leaves[:len(x_leaves)])
        pp = jax.tree_util.tree_unflatten(p_tree, leaves[len(x_leaves):])
        out = fun(xx, pp)
        out = tuple(out) if isinstance(out, (tuple, list)) else (out,)
        return tuple(jnp.ravel(o) for o in out)
    avals = [jax.ShapeDtypeStruct(np.shape(l), np.float64) for l in x_leaves + p_leaves]
    with vmap_as_map():
        closed = jax.make_jaxpr(flat_fn)(*avals)
    parts = [str(closed.jaxpr), str(x_tree), str(p_tree)]
    for c in closed.consts:
        a = np.asarray(c)
        parts.append(f"{a.dtype}{a.shape}:" + a.tobytes().hex())
    return closed, "\n".join(parts)


def to_mx(fun, x, p, traced=None) -> MXGraph:
    """Trace fun(x, p) -> array or tuple of arrays (x, p: example pytrees; only shapes are used) into a
    CasADi MX graph; see the module docstring. `traced`: the closed jaxpr from `trace(fun, x, p)`
    (skips tracing again)."""
    x_leaves, x_tree = jax.tree_util.tree_flatten(x)
    x_paths = [jax.tree_util.keystr(k) for k, _ in jax.tree_util.tree_flatten_with_path(x)[0]]
    x_names = [s.strip(".[]'") or f"x{i}" for i, s in enumerate(x_paths)]
    p_leaves, p_tree = jax.tree_util.tree_flatten(p)
    p_size = int(sum(np.size(l) for l in p_leaves))
    closed = trace(fun, x, p)[0] if traced is None else traced

    x_syms, x_arrs = zip(*[sx_inputs(f"X{i}", np.shape(l)) for i, l in enumerate(x_leaves)]) if x_leaves else ((), ())
    p_sym, p_arr = sx_inputs("P", (p_size,))

    def mx_shape(l):
        s = np.shape(l)
        return (int(np.prod(s[1:], dtype=np.int64)), s[0]) if len(s) >= 2 else (int(np.size(l)), 1)
    X = [ca.MX.sym(nm, *mx_shape(l)) for nm, l in zip(x_names, x_leaves)]
    Pm = ca.MX.sym("p", p_size)
    T = _Translator(list(x_syms), p_sym, [ca.vec(v) for v in X], Pm)
    p_parts, at = [], 0
    for l in p_leaves:
        k = int(np.size(l))
        p_parts.append(p_arr[at:at + k].reshape(np.shape(l)))
        at += k
    outs = T.run(closed, list(x_arrs) + p_parts)
    ph = [s for m in T.maps for s in m["ph"]]
    ys = [y for m in T.maps for y in m["y"]]
    glue = ca.Function("glue", list(x_syms) + [p_sym] + ph, [to_sx_vector(o) for o in outs])
    mx_outs = glue.call([ca.vec(v) for v in X] + [Pm] + ys)
    return MXGraph(x=X, p=[Pm], outs=list(mx_outs), x_names=x_names, x_tree=x_tree, p_tree=p_tree,
                   x_shapes=[tuple(np.shape(l)) for l in x_leaves], p_shapes=[tuple(np.shape(l)) for l in p_leaves],
                   n_maps=len(T.maps))
