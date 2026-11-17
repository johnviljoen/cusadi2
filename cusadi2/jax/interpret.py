"""jaxpr -> CasADi SX, one SX scalar per array element.

`jax.make_jaxpr` records the raw primitive stream of a JAX function (no XLA
fusion or hardware rewriting happens before lowering), so interpreting that
stream with scalar SX symbols reproduces the function exactly; CasADi's
constant folding at construction (``0 * sin(x) -> 0``) then gives symbolic
Jacobians and Hessians with the true, numerically generic sparsity. This is
what the GPU plans are built from.

Arrays are represented as numpy arrays: numeric dtype while every element is a
known constant, ``object`` dtype (elements are 1x1 ``casadi.SX`` or Python
numbers) once any element depends on an input. Index arithmetic (``iota``,
``gather`` indices, ``cond`` predicates) therefore stays concrete whenever JAX
could resolve it at trace time.

Boundaries (the same for any tracer): ``scan`` / ``fori_loop`` with a static
trip count are unrolled, ``cond`` with a symbolic predicate becomes an
elementwise ``if_else`` (union sparsity), and a ``while_loop`` or a ``gather``
/ ``scatter`` with symbolic indices has no SX form and raises
:class:`Unsupported`.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Literal


class Unsupported(NotImplementedError):
    """The jaxpr contains something SX cannot express."""


# --------------------------------------------------------------------------- helpers
def _is_sym(a) -> bool:
    return isinstance(a, np.ndarray) and a.dtype == object


def _arr(a) -> np.ndarray:
    """np.asarray that never hands a bare SX to numpy (which would try to convert it)."""
    if isinstance(a, ca.SX):
        out = np.empty((), dtype=object)
        out[()] = a
        return out
    return np.asarray(a)


def _obj(a) -> np.ndarray:
    """View as an object array (elements: SX or Python scalars)."""
    a = _arr(a)
    return a if a.dtype == object else a.astype(object)


def _vec(fn: Callable) -> Callable:
    return np.vectorize(fn, otypes=[object])


def _as_sx(e) -> ca.SX:
    return e if isinstance(e, ca.SX) else ca.SX(float(e))


def _static_int_array(a, what: str) -> np.ndarray:
    if _is_sym(a):
        raise Unsupported(f"{what} depends on the inputs; SX cannot index symbolically")
    return np.asarray(a).astype(np.int64)


# ------------------------------------------------------------------- scalar op tables
# (numpy form for the all-constant path, SX form for the symbolic path)
_UNARY = {
    "neg": (np.negative, lambda a: -a),
    "sign": (np.sign, ca.sign),
    "abs": (np.abs, ca.fabs),
    "floor": (np.floor, ca.floor),
    "ceil": (np.ceil, ca.ceil),
    "exp": (np.exp, ca.exp),
    "exp2": (np.exp2, lambda a: ca.exp(a * math.log(2.0))),
    "log": (np.log, ca.log),
    "log1p": (np.log1p, ca.log1p),
    "expm1": (np.expm1, ca.expm1),
    "sin": (np.sin, ca.sin),
    "cos": (np.cos, ca.cos),
    "tan": (np.tan, ca.tan),
    "asin": (np.arcsin, ca.asin),
    "acos": (np.arccos, ca.acos),
    "atan": (np.arctan, ca.atan),
    "sinh": (np.sinh, ca.sinh),
    "cosh": (np.cosh, ca.cosh),
    "tanh": (np.tanh, ca.tanh),
    "asinh": (np.arcsinh, ca.asinh),
    "acosh": (np.arccosh, ca.acosh),
    "atanh": (np.arctanh, ca.atanh),
    "sqrt": (np.sqrt, ca.sqrt),
    "rsqrt": (lambda a: 1.0 / np.sqrt(a), lambda a: 1.0 / ca.sqrt(a)),
    "cbrt": (np.cbrt, lambda a: ca.sign(a) * ca.fabs(a) ** (1.0 / 3.0)),
    "square": (np.square, lambda a: a * a),
    "logistic": (lambda a: 1.0 / (1.0 + np.exp(-a)), lambda a: 1.0 / (1.0 + ca.exp(-a))),
    "erf": (None, ca.erf),
    "erf_inv": (None, ca.erfinv),
    "not": (np.logical_not, ca.logic_not),
}
_BINARY = {
    "add": (np.add, lambda a, b: a + b),
    "sub": (np.subtract, lambda a, b: a - b),
    "mul": (np.multiply, lambda a, b: a * b),
    "div": (np.divide, lambda a, b: a / b),
    "rem": (np.fmod, ca.remainder),
    "pow": (np.power, lambda a, b: a ** b),
    "atan2": (np.arctan2, ca.atan2),
    "max": (np.maximum, ca.fmax),
    "min": (np.minimum, ca.fmin),
    "nextafter": (np.nextafter, None),
    "eq": (np.equal, lambda a, b: ca.eq(a, b)),
    "ne": (np.not_equal, lambda a, b: ca.ne(a, b)),
    "lt": (np.less, lambda a, b: ca.lt(a, b)),
    "le": (np.less_equal, lambda a, b: ca.le(a, b)),
    "gt": (np.greater, lambda a, b: ca.gt(a, b)),
    "ge": (np.greater_equal, lambda a, b: ca.ge(a, b)),
    "and": (np.logical_and, ca.logic_and),
    "or": (np.logical_or, ca.logic_or),
}


def _elementwise(name: str, np_fn, sx_fn):
    def rule(ins, eqn):
        if not any(_is_sym(a) for a in ins):
            if np_fn is None:
                raise Unsupported(f"{name} on constants is not implemented")
            return np_fn(*[np.asarray(a) for a in ins])
        if sx_fn is None:
            raise Unsupported(f"{name} has no SX form")
        return _vec(sx_fn)(*[_obj(a) for a in ins])
    return rule


RULES: Dict[str, Callable] = {}
for _n, (_np, _sx) in _UNARY.items():
    RULES[_n] = _elementwise(_n, _np, _sx)
for _n, (_np, _sx) in _BINARY.items():
    RULES[_n] = _elementwise(_n, _np, _sx)


def rule(*names):
    def deco(fn):
        for n in names:
            RULES[n] = fn
        return fn
    return deco


# ------------------------------------------------------------------- scalar extras
@rule("is_finite")
def _is_finite(ins, eqn):
    """SX symbols are assumed finite: a symbolic is_finite is the constant True, so a
    jnp.where(isfinite(x), a, b) picks `a` statically instead of an if_else."""
    a = ins[0]
    return np.ones(np.shape(a), dtype=bool) if _is_sym(a) else np.isfinite(np.asarray(a))


@rule("integer_pow")
def _integer_pow(ins, eqn):
    y = eqn.params["y"]
    if not _is_sym(ins[0]):
        return np.asarray(ins[0]) ** y
    return _vec(lambda a: a ** y)(_obj(ins[0]))


@rule("convert_element_type")
def _convert(ins, eqn):
    a = ins[0]
    if _is_sym(a):
        return a
    new = eqn.params["new_dtype"]
    return np.asarray(a).astype(np.dtype(new))


@rule("stop_gradient", "copy", "copy_p", "optimization_barrier", "reduce_precision")
def _identity(ins, eqn):
    return ins if len(ins) > 1 else ins[0]


@rule("clamp")
def _clamp(ins, eqn):
    lo, a, hi = ins
    if not any(_is_sym(v) for v in ins):
        return np.clip(np.asarray(a), np.asarray(lo), np.asarray(hi))
    return _vec(lambda l, x, h: ca.fmin(ca.fmax(x, l), h))(_obj(lo), _obj(a), _obj(hi))


@rule("select_n")
def _select_n(ins, eqn):
    pred, cases = ins[0], ins[1:]
    if not _is_sym(pred):
        idx = np.asarray(pred).astype(np.int64)
        if not any(_is_sym(c) for c in cases):
            return np.choose(idx, [np.asarray(c) for c in cases])
        return np.choose(idx, [np.broadcast_to(_obj(c), idx.shape) for c in cases]) if idx.ndim \
            else _obj(cases[int(idx)])
    cases = [_obj(c) for c in cases]
    # symbolic predicate: nested if_else over the case index (bool pred: cases[0] is False)
    def pick(p, *cs):
        out = cs[-1]
        for k in range(len(cs) - 2, -1, -1):
            out = ca.if_else(ca.eq(p, k), cs[k], out) if len(cs) > 2 else ca.if_else(p, cs[1], cs[0])
            if len(cs) == 2:
                break
        return out
    return _vec(pick)(_obj(pred), *cases)


# ------------------------------------------------------------------- shape ops
@rule("broadcast_in_dim")
def _broadcast_in_dim(ins, eqn):
    a = np.asarray(ins[0])
    shape, bdims = eqn.params["shape"], eqn.params["broadcast_dimensions"]
    new = [1] * len(shape)
    for i, d in enumerate(bdims):
        new[d] = a.shape[i]
    return np.broadcast_to(a.reshape(new), shape).copy()


@rule("reshape")
def _reshape(ins, eqn):
    return np.asarray(ins[0]).reshape(eqn.params["new_sizes"])


@rule("squeeze")
def _squeeze(ins, eqn):
    return np.squeeze(np.asarray(ins[0]), axis=tuple(eqn.params["dimensions"]))


@rule("expand_dims")
def _expand_dims(ins, eqn):
    return np.expand_dims(np.asarray(ins[0]), tuple(eqn.params["dimensions"]))


@rule("transpose")
def _transpose(ins, eqn):
    return np.transpose(np.asarray(ins[0]), eqn.params["permutation"])


@rule("rev")
def _rev(ins, eqn):
    return np.flip(np.asarray(ins[0]), axis=tuple(eqn.params["dimensions"])).copy()


@rule("concatenate")
def _concatenate(ins, eqn):
    arrs = [np.asarray(a) for a in ins]
    if any(_is_sym(a) for a in arrs):
        arrs = [_obj(a) for a in arrs]
    return np.concatenate(arrs, axis=eqn.params["dimension"])


@rule("stack")
def _stack(ins, eqn):
    arrs = [np.asarray(a) for a in ins]
    if any(_is_sym(a) for a in arrs):
        arrs = [_obj(a) for a in arrs]
    axis = eqn.params["axis"] if "axis" in eqn.params else eqn.params["dimension"]
    return np.stack(arrs, axis=axis)


@rule("unstack")
def _unstack(ins, eqn):
    a = np.asarray(ins[0])
    axis = eqn.params.get("axis", 0)
    return [np.take(a, i, axis=axis) for i in range(a.shape[axis])]


@rule("split")
def _split(ins, eqn):
    sizes, axis = eqn.params["sizes"], eqn.params["axis"]
    return list(np.split(np.asarray(ins[0]), np.cumsum(sizes[:-1]), axis=axis))


@rule("slice")
def _slice(ins, eqn):
    s, e = eqn.params["start_indices"], eqn.params["limit_indices"]
    st = eqn.params["strides"] or [1] * len(s)
    return np.asarray(ins[0])[tuple(slice(a, b, c) for a, b, c in zip(s, e, st))].copy()


@rule("dynamic_slice")
def _dynamic_slice(ins, eqn):
    a = np.asarray(ins[0])
    starts = [int(_static_int_array(i, "dynamic_slice start index")) for i in ins[1:]]
    sizes = eqn.params["slice_sizes"]
    starts = [min(max(s, 0), d - n) for s, d, n in zip(starts, a.shape, sizes)]  # JAX clamps
    return a[tuple(slice(s, s + n) for s, n in zip(starts, sizes))].copy()


@rule("dynamic_update_slice")
def _dynamic_update_slice(ins, eqn):
    a, upd = np.asarray(ins[0]), np.asarray(ins[1])
    starts = [int(_static_int_array(i, "dynamic_update_slice start index")) for i in ins[2:]]
    starts = [min(max(s, 0), d - n) for s, d, n in zip(starts, a.shape, upd.shape)]
    out = _obj(a).copy() if _is_sym(a) or _is_sym(upd) else a.copy()
    out[tuple(slice(s, s + n) for s, n in zip(starts, upd.shape))] = upd
    return out


@rule("pad")
def _pad(ins, eqn):
    a, val = np.asarray(ins[0]), np.asarray(ins[1])
    cfg = eqn.params["padding_config"]
    if any(lo < 0 or hi < 0 for lo, hi, _ in cfg):
        raise Unsupported("negative padding")
    if any(interior for _, _, interior in cfg):
        shape = [lo + (n - 1) * (it + 1) + 1 + hi for n, (lo, hi, it) in zip(a.shape, cfg)]
        out = np.full(shape, val.item(), dtype=object if _is_sym(a) else a.dtype)
        out[tuple(slice(lo, lo + n * (it + 1), it + 1) for n, (lo, hi, it) in zip(a.shape, cfg))] = a
        return out
    if _is_sym(a) or _is_sym(val):
        return np.pad(_obj(a), [(lo, hi) for lo, hi, _ in cfg], constant_values=val.item())
    return np.pad(a, [(lo, hi) for lo, hi, _ in cfg], constant_values=val.item())


@rule("empty")
def _empty(ins, eqn):
    return np.zeros(eqn.params["shape"], dtype=np.dtype(eqn.params["dtype"]))


@rule("tile")
def _tile(ins, eqn):
    return np.tile(np.asarray(ins[0]), eqn.params["reps"])


@rule("sort")
def _sort(ins, eqn):
    if any(_is_sym(a) for a in ins):
        raise Unsupported("sort of symbolic values has no SX form (only static index sorts)")
    return _bind_constants(eqn, ins)


def _bind_constants(eqn, ins):
    """Evaluate a primitive on concrete inputs with JAX itself, eagerly even when the
    interpreter runs inside an enclosing jit trace (jax_function called under jax.jit)."""
    with jax.ensure_compile_time_eval():
        out = eqn.primitive.bind(*[jnp.asarray(a) for a in ins], **eqn.params)
    return [np.asarray(o) for o in out] if isinstance(out, (list, tuple)) else np.asarray(out)


@rule("call_hi_primitive")
def _call_hi(ins, eqn):
    """JAX >= 0.12 'hi' primitives (e.g. jnp.nonzero): expand to lo-level jaxpr and interpret."""
    prim = eqn.params["_prim"]
    avals = [jax.ShapeDtypeStruct(np.shape(a), _np_dtype(a)) for a in ins]
    closed = jax.make_jaxpr(lambda *a: prim.expand(*a))(*avals)
    return eval_jaxpr(closed, ins)


def _np_dtype(a):
    return np.asarray(a).dtype if not _is_sym(a) else np.dtype("float64")


@rule("iota")
def _iota(ins, eqn):
    shape, dim = eqn.params["shape"], eqn.params["dimension"]
    view = [1] * len(shape)
    view[dim] = shape[dim]
    return np.broadcast_to(np.arange(shape[dim], dtype=np.dtype(eqn.params["dtype"])).reshape(view), shape).copy()


# ------------------------------------------------------------------- reductions
def _reduce(np_fn, sx_pair):
    def rule_(ins, eqn):
        a = np.asarray(ins[0])
        axes = tuple(eqn.params["axes"])
        if not _is_sym(a):
            return np_fn(a, axis=axes)
        if any(a.shape[ax] == 0 for ax in axes):
            raise Unsupported("reduction over an empty axis")
        keep = [d for d in range(a.ndim) if d not in axes]
        moved = np.transpose(_obj(a), keep + list(axes)).reshape([a.shape[d] for d in keep] + [-1])
        out = np.empty(moved.shape[:-1], dtype=object)
        for oi in np.ndindex(*out.shape):
            acc = moved[oi + (0,)]
            for k in range(1, moved.shape[-1]):
                acc = sx_pair(acc, moved[oi + (k,)])
            out[oi] = acc
        return out
    return rule_


RULES["reduce_sum"] = _reduce(np.sum, lambda a, b: a + b)
RULES["reduce_and"] = _reduce(np.all, ca.logic_and)
RULES["reduce_or"] = _reduce(np.any, ca.logic_or)
RULES["reduce_prod"] = _reduce(np.prod, lambda a, b: a * b)
RULES["reduce_max"] = _reduce(np.max, ca.fmax)
RULES["reduce_min"] = _reduce(np.min, ca.fmin)


@rule("cumsum")
def _cumsum(ins, eqn):
    a = np.asarray(ins[0])
    axis, reverse = eqn.params["axis"], eqn.params["reverse"]
    if not _is_sym(a):
        return np.flip(np.cumsum(np.flip(a, axis), axis), axis) if reverse else np.cumsum(a, axis)
    moved = np.moveaxis(_obj(a), axis, 0)
    out = np.empty(moved.shape, dtype=object)
    order = range(moved.shape[0] - 1, -1, -1) if reverse else range(moved.shape[0])
    prev = None
    for t in order:
        for oi in np.ndindex(*moved.shape[1:]):
            out[(t,) + oi] = moved[(t,) + oi] if prev is None else out[(prev,) + oi] + moved[(t,) + oi]
        prev = t
    return np.moveaxis(out, 0, axis)


@rule("dot_general")
def _dot_general(ins, eqn):
    a, b = np.asarray(ins[0]), np.asarray(ins[1])
    ((lc, rc), (lb, rb)) = eqn.params["dimension_numbers"]
    if not _is_sym(a) and not _is_sym(b):
        return np.einsum(_einsum_spec(a.ndim, b.ndim, lc, rc, lb, rb), a, b)
    a, b = _obj(a), _obj(b)
    la = [i for i in range(a.ndim) if i not in lc and i not in lb]
    ra = [i for i in range(b.ndim) if i not in rc and i not in rb]
    bsz = [a.shape[i] for i in lb]
    nb_, ni, nk = (int(np.prod(bsz, initial=1)), int(np.prod([a.shape[i] for i in la], initial=1)),
                   int(np.prod([a.shape[i] for i in lc], initial=1)))
    nj = int(np.prod([b.shape[i] for i in ra], initial=1))
    A = np.transpose(a, list(lb) + la + list(lc)).reshape(nb_, ni, nk)
    B = np.transpose(b, list(rb) + list(rc) + ra).reshape(nb_, nk, nj)
    out = np.empty((A.shape[0], A.shape[1], B.shape[2]), dtype=object)
    for n in range(A.shape[0]):
        for i in range(A.shape[1]):
            for j in range(B.shape[2]):
                acc = 0.0
                for k in range(A.shape[2]):
                    acc = acc + A[n, i, k] * B[n, k, j]
                out[n, i, j] = acc
    return out.reshape(bsz + [a.shape[i] for i in la] + [b.shape[i] for i in ra])


def _einsum_spec(na, nb, lc, rc, lb, rb):
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    sa = [next(letters) for _ in range(na)]
    sb = [None] * nb
    for i, j in zip(lb, rb):
        sb[j] = sa[i]
    for i, j in zip(lc, rc):
        sb[j] = sa[i]
    sb = [s or next(letters) for s in sb]
    out = [sa[i] for i in lb] + [sa[i] for i in range(na) if i not in lb and i not in lc] + \
          [sb[j] for j in range(nb) if j not in rb and j not in rc]
    return f"{''.join(sa)},{''.join(sb)}->{''.join(out)}"


@rule("add_any")
def _add_any(ins, eqn):
    return RULES["add"](ins, eqn)


@rule("zeros_like")
def _zeros_like(ins, eqn):
    a = np.asarray(ins[0])
    return np.zeros(a.shape, dtype=np.float64 if _is_sym(a) else a.dtype)


# ------------------------------------------------------------------- dense linear algebra
# Unrolled on SX scalars: jnp.linalg.cholesky and the triangular solves behind
# jax.scipy.linalg.cho_solve / solve_triangular. (jnp.linalg.solve / inv use a
# pivoted LU, which has no symbolic form: factor SPD systems with Cholesky.)
def _leading(a, nd):
    """Iterate over leading batch dims of an array whose last nd dims are the matrix."""
    return list(np.ndindex(*a.shape[:-nd]))


@rule("cholesky")
def _cholesky(ins, eqn):
    A = np.asarray(ins[0])
    if not _is_sym(A):
        return np.linalg.cholesky(A)
    A = _obj(A)
    out = np.zeros(A.shape, dtype=object)
    n = A.shape[-1]
    for b in _leading(A, 2):
        L = out[b]
        for j in range(n):
            d = A[b][j, j]
            for k in range(j):
                d = d - L[j, k] * L[j, k]
            L[j, j] = ca.sqrt(d)
            for i in range(j + 1, n):
                v = A[b][i, j]
                for k in range(j):
                    v = v - L[i, k] * L[j, k]
                L[i, j] = v / L[j, j]
    return out


@rule("triangular_solve")
def _triangular_solve(ins, eqn):
    a, b = np.asarray(ins[0]), np.asarray(ins[1])
    p = eqn.params
    lower, left, trans, unit = p["lower"], p["left_side"], p["transpose_a"], p["unit_diagonal"]
    sym = _is_sym(a) or _is_sym(b)
    a = _obj(a) if sym else a.astype(np.float64)
    b = _obj(b) if sym else b.astype(np.float64)
    if trans:                      # solve op(a) x = b with op = transpose
        a, lower = np.swapaxes(a, -1, -2), not lower
    if not left:                   # x a = b  <=>  a^T x^T = b^T
        a, b, lower = np.swapaxes(a, -1, -2), np.swapaxes(b, -1, -2), not lower
    out = np.zeros(b.shape, dtype=object if sym else np.float64)
    n = a.shape[-1]
    for bi in _leading(b, 2):
        A, B, X = a[bi], b[bi], out[bi]
        rows = range(n) if lower else range(n - 1, -1, -1)
        for i in rows:
            for c in range(B.shape[1]):
                v = B[i, c]
                ks = range(i) if lower else range(i + 1, n)
                for k in ks:
                    v = v - A[i, k] * X[k, c]
                X[i, c] = v if unit else v / A[i, i]
    if not left:
        out = np.swapaxes(out, -1, -2)
    return out


# ------------------------------------------------------------------- gather / scatter
@rule("gather")
def _gather(ins, eqn):
    operand = np.asarray(ins[0])
    idx = _static_int_array(ins[1], "gather indices")
    dn, sizes = eqn.params["dimension_numbers"], eqn.params["slice_sizes"]
    offset_dims, collapsed, start_map = dn.offset_dims, dn.collapsed_slice_dims, dn.start_index_map
    operand_bd = tuple(getattr(dn, "operand_batching_dims", ()))
    idx_bd = tuple(getattr(dn, "start_indices_batching_dims", ()))
    batch_shape = list(idx.shape[:-1])
    offset_shape = [sizes[d] for d in range(operand.ndim) if d not in collapsed and d not in operand_bd]
    out_rank = len(batch_shape) + len(offset_shape)
    batch_dims_out = [d for d in range(out_rank) if d not in offset_dims]
    out_shape = [0] * out_rank
    for d, n in zip(batch_dims_out, batch_shape):
        out_shape[d] = n
    for d, n in zip(offset_dims, offset_shape):
        out_shape[d] = n
    out = np.empty(out_shape, dtype=operand.dtype)
    window_operand_dims = [d for d in range(operand.ndim) if d not in collapsed and d not in operand_bd]
    for oi in np.ndindex(*out_shape):
        g = tuple(oi[d] for d in batch_dims_out)
        start = [0] * operand.ndim
        for k, od in enumerate(start_map):
            start[od] = int(idx[g + (k,)])
        for od, bd in zip(operand_bd, idx_bd):   # batching dims pair operand[od] with the index batch coordinate
            start[od] = g[bd]
        # clamp like XLA
        for d in range(operand.ndim):
            start[d] = min(max(start[d], 0), operand.shape[d] - sizes[d])
        full = list(start)
        for k, od in enumerate(window_operand_dims):
            full[od] += oi[offset_dims[k]]
        out[oi] = operand[tuple(full)]
    return out


@rule("scatter", "scatter-add", "scatter_add", "scatter-mul", "scatter_mul")
def _scatter(ins, eqn):
    operand, upd = np.asarray(ins[0]), np.asarray(ins[2])
    idx = _static_int_array(ins[1], "scatter indices")
    dn = eqn.params["dimension_numbers"]
    uwd, iwd, s2o = dn.update_window_dims, dn.inserted_window_dims, dn.scatter_dims_to_operand_dims
    operand_bd = tuple(getattr(dn, "operand_batching_dims", ()))
    idx_bd = tuple(getattr(dn, "scatter_indices_batching_dims", ()))
    name = eqn.primitive.name
    out = _obj(operand).copy() if (_is_sym(operand) or _is_sym(upd)) else operand.copy()
    scatter_dims = [d for d in range(upd.ndim) if d not in uwd]
    window_operand_dims = [d for d in range(operand.ndim) if d not in iwd and d not in operand_bd]
    for ui in np.ndindex(*upd.shape):
        g = tuple(ui[d] for d in scatter_dims)
        full = [0] * operand.ndim
        for k, od in enumerate(s2o):
            full[od] = int(idx[g + (k,)])
        for od, bd in zip(operand_bd, idx_bd):
            full[od] = g[bd]
        for k, od in enumerate(window_operand_dims):
            full[od] += ui[uwd[k]]
        if any(i < 0 or i >= n for i, n in zip(full, operand.shape)):
            continue  # FILL_OR_DROP semantics
        t = tuple(full)
        if name in ("scatter-add", "scatter_add"):
            out[t] = out[t] + upd[ui]
        elif name in ("scatter-mul", "scatter_mul"):
            out[t] = out[t] * upd[ui]
        else:
            out[t] = upd[ui]
    return out


# ------------------------------------------------------------------- higher order
def _sub(closed, ins):
    return eval_jaxpr(closed, ins)


@rule("pjit", "jit")
def _pjit(ins, eqn):
    return _sub(eqn.params["jaxpr"], ins)


@rule("closed_call", "core_call")
def _closed_call(ins, eqn):
    return _sub(eqn.params["call_jaxpr"], ins)


@rule("custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr")
def _custom(ins, eqn):
    cj = eqn.params.get("call_jaxpr") or eqn.params.get("fun_jaxpr")
    return _sub(cj, ins)


@rule("remat", "checkpoint", "remat2")
def _remat(ins, eqn):
    j = eqn.params["jaxpr"]
    closed = j if hasattr(j, "consts") else jax.extend.core.ClosedJaxpr(j, [])
    return _sub(closed, ins)


@rule("scan")
def _scan(ins, eqn):
    p = eqn.params
    body, n = p["jaxpr"], p["length"]
    if "num_consts" in p:
        nc, nk = p["num_consts"], p["num_carry"]
        consts, carry, xs = list(ins[:nc]), list(ins[nc:nc + nk]), list(ins[nc + nk:])
    else:  # jax >= 0.12: FlatTree layout
        consts, carry, xs = [list(t) for t in p["ft_in"].update(ins).unpack()]
    ys = []
    for t in range(n):
        tt = n - 1 - t if p["reverse"] else t
        outs = _sub(body, consts + carry + [np.asarray(x)[tt] for x in xs])
        carry, y = list(outs[:len(carry)]), list(outs[len(carry):])
        ys.append(y)
    if p["reverse"]:
        ys = ys[::-1]
    stacked = [np.stack([_obj(ys[t][k]) if any(_is_sym(np.asarray(ys[s][k])) for s in range(n))
                         else np.asarray(ys[t][k]) for t in range(n)]) for k in range(len(ys[0]))] if n else []
    return carry + stacked


@rule("cond")
def _cond(ins, eqn):
    branches = eqn.params["branches"]
    index, ops = ins[0], list(ins[1:])
    if not _is_sym(index):
        k = int(np.asarray(index))
        return _sub(branches[min(max(k, 0), len(branches) - 1)], ops)
    outs = [_sub(b, ops) for b in branches]
    idx = _obj(index).item()
    result = []
    for k_out in range(len(outs[0])):
        acc = _obj(outs[-1][k_out])
        for k in range(len(branches) - 2, -1, -1):
            cand = _obj(outs[k][k_out])
            cond = ca.eq(idx, k) if len(branches) > 2 else ca.logic_not(idx) if k == 0 else idx
            acc = _vec(lambda c, f: ca.if_else(cond, c, f))(cand, acc)
        result.append(acc)
    return result


@rule("while")
def _while(ins, eqn):
    raise Unsupported("while_loop with a data-dependent trip count has no SX form; "
                      "use lax.scan / fori_loop with static bounds")


# ------------------------------------------------------------------- the interpreter
def eval_jaxpr(closed, args: Sequence) -> List[np.ndarray]:
    """Evaluate a ClosedJaxpr on numpy / object arrays; returns one array per output."""
    jaxpr = closed.jaxpr
    env = {}

    def read(v):
        if isinstance(v, Literal):
            return np.asarray(v.val)
        return env[v]

    for v, c in zip(jaxpr.constvars, closed.consts):
        env[v] = np.asarray(c)
    for v, a in zip(jaxpr.invars, args):
        env[v] = np.asarray(a)
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        ins = [read(v) for v in eqn.invars]
        if name in RULES:
            out = RULES[name](ins, eqn)
        elif not any(_is_sym(a) for a in ins):   # constant subgraph: let JAX evaluate it
            out = _bind_constants(eqn, ins)
        else:
            raise Unsupported(f"JAX primitive {name!r} has no SX rule")
        if not isinstance(out, (list, tuple)):
            out = [out]
        if len(out) != len(eqn.outvars):
            raise RuntimeError(f"{name}: rule returned {len(out)} outputs, jaxpr expects {len(eqn.outvars)}")
        for v, o in zip(eqn.outvars, out):
            o = _arr(o)
            if o.shape != tuple(v.aval.shape):
                raise RuntimeError(f"{name}: shape {o.shape} != expected {tuple(v.aval.shape)}")
            env[v] = o
    return [read(v) for v in jaxpr.outvars]


def sx_inputs(name: str, shape) -> tuple:
    """One SX symbol vector of size prod(shape) and its row-major object-array view."""
    n = int(np.prod(shape, dtype=np.int64)) if len(shape) else 1
    sym = ca.SX.sym(name, n)
    arr = np.empty(n, dtype=object)
    for i in range(n):
        arr[i] = sym[i]
    return sym, arr.reshape(shape)


def to_sx_vector(a) -> ca.SX:
    """Flatten an interpreter output (row-major) into a dense SX column vector."""
    flat = np.asarray(a).reshape(-1)
    if flat.size == 0:
        return ca.SX(0, 1)
    return ca.vertcat(*[_as_sx(e) for e in flat])


def trace_to_sx(fun: Callable, in_shapes: Sequence[tuple], in_dtypes=None, names=None):
    """Trace `fun(*arrays)` and return (sx_syms, sx_outs, out_shapes, closed_jaxpr).

    `sx_syms[i]` is the flat SX symbol for argument i, `sx_outs[k]` the dense flat
    SX column for output k (row-major), `out_shapes[k]` its JAX shape.
    """
    in_dtypes = in_dtypes or [np.float64] * len(in_shapes)
    avals = [jax.ShapeDtypeStruct(tuple(s), d) for s, d in zip(in_shapes, in_dtypes)]
    closed = jax.make_jaxpr(fun)(*avals)
    names = names or [f"x{i}" for i in range(len(in_shapes))]
    syms, arrs = zip(*[sx_inputs(nm, tuple(s)) for nm, s in zip(names, in_shapes)]) if in_shapes else ((), ())
    outs = eval_jaxpr(closed, list(arrs))
    out_shapes = [tuple(v.aval.shape) for v in closed.jaxpr.outvars]
    return list(syms), [to_sx_vector(o) for o in outs], out_shapes, closed
