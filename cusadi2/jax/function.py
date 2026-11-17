"""cusadi2.jax.jax_function: a JAX function evaluated by cusadi2 on the GPU, with
exact-sparsity Jacobians and Hessians formed symbolically in CasADi.

    fn = jax_function(f, param_argnums=(1,))
    y = fn(x, p)                    # dense, same as f(x, p); works under jit and vmap
    jac = fn.jacobian()             # jac(x, p) -> values (nnz,) per output; jac.coo(x, p)
    hes = fn.hessian()              # true Hessian of each vector output, coords (k, i, j)
    hes = fn.hessian(weighted=True) # Lagrangian Hessian: hes(x, p, lam_0, ...) -> (nnz,)

Shapes are taken from the arguments the first time each variant is called
(concrete arrays or jit tracers both carry them) and every (variant,
signature) pair is traced, differentiated, planned and uploaded once, then
cached. Arguments not listed in `param_argnums` are the variables: flattened
row-major and concatenated in order into the single CasADi variable the
derivatives are taken with respect to. Parameters are runtime inputs that are
never differentiated.

dtype is decided on the JAX side: the plan uses the floating dtype of the
arguments at the call (float32 when x64 is disabled, float64 when the arrays
are float64), each dtype being its own kernel build; `dtype=` on
`jax_function` overrides. Coordinates are int32 either way.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Callable, Sequence

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

from cusadi2.jax.interpret import eval_jaxpr, sx_inputs, to_sx_vector
from cusadi2.jax.primitive import MAX_ARGS, cusadi2_eval, ext_for
from cusadi2.plan import device_max_live, plan_bytes

_plan_ids = itertools.count(1)


def _signature(leaves, dtype):
    return (tuple(tuple(int(s) for s in np.shape(a)) for a in leaves), np.dtype(dtype))


def _infer_dtype(leaves):
    """The floating dtype of the arguments as JAX sees them (honours jax_enable_x64)."""
    dt = jnp.result_type(*[jnp.asarray(a) for a in leaves]) if leaves else jnp.float64
    if not jnp.issubdtype(dt, jnp.floating):
        dt = jnp.float64
    return np.dtype(jax.dtypes.canonicalize_dtype(dt))


class _Plan:
    """One CasADi Function uploaded as one cusadi2 plan, callable on flat leaves."""

    def __init__(self, name: str, fun: ca.Function, out_shapes, coo, dtype, block_size, serial, max_live,
                 composition=None):
        self.plan_id = next(_plan_ids)
        self.fun = fun
        self.dtype = np.dtype(dtype)
        self.ext = ext_for(self.dtype)
        self.out_shapes = [tuple(s) for s in out_shapes]
        self.coo = coo                      # None (dense outputs) or list of int32 (nnz, k) arrays
        if composition is None:             # the function's own arguments, batch only
            self.nnz_in = [int(fun.nnz_in(i)) for i in range(fun.n_in())]
            self.nnz_out = [int(fun.nnz_out(k)) for k in range(fun.n_out())]
            self.scratch = 0
        else:                               # fun per stage behind the composition's declared arguments
            self.nnz_in, self.nnz_out = composition.in_lengths, composition.out_lengths
            self.scratch = int(composition.scratch_per_instance)
        if len(self.nnz_in) > MAX_ARGS or len(self.nnz_out) + (self.scratch > 0) > MAX_ARGS:
            raise ValueError(f"cusadi2.jax: at most {MAX_ARGS} inputs and outputs per function")
        if max_live is None:   # window the levels to the device's shared memory (plans of big Jacobians / Hessians)
            max_live = device_max_live("float" if self.dtype == np.float32 else "double")
        self.serial = bool(serial)
        kw = dict(block_size=block_size, composition=composition)
        if serial:
            data = plan_bytes(fun, **kw)
        else:
            try:
                data = plan_bytes(fun, max_live=max_live, **kw)
            except ValueError as e:      # no window fits shared memory: run CasADi's instruction stream
                warnings.warn(f"cusadi2.jax: {e}; using the serial (thread-per-instance) kernel for {name}")
                data = plan_bytes(fun, **kw)
                self.serial = True
        self.ext.install_plan(self.plan_id, data, block_size or 0, self.serial)

    def __call__(self, leaves):
        xs = []
        for a, n in zip(leaves, self.nnz_in):
            a = jnp.asarray(a).astype(self.dtype).reshape(1, n)
            xs.append(a)
        outs = cusadi2_eval(xs, plan_id=self.plan_id, batch=1, dtype=self.dtype, nnz_out=tuple(self.nnz_out),
                            scratch=self.scratch)
        if self.coo is None:
            return [o.reshape(s) for o, s in zip(outs, self.out_shapes)]
        return [o.reshape(n) for o, n in zip(outs, self.nnz_out)]

    def free(self):
        self.ext.free_plan(self.plan_id)


class JaxFunction:
    """See module docstring."""

    def __init__(self, fun: Callable, param_argnums: Sequence[int] = (), name: str | None = None,
                 dtype=None, block_size: int | None = None, serial: bool = False,
                 max_live: int | None = None):
        self.fun = fun
        self.dtype = None if dtype is None else np.dtype(jax.dtypes.canonicalize_dtype(dtype))
        self.param_argnums = tuple(sorted(int(i) for i in param_argnums))
        self.name = _casadi_name(name or getattr(fun, "__name__", "f"))
        self.block_size, self.serial, self.max_live = block_size, serial, max_live
        self._plans: dict = {}
        self._traces: dict = {}

    # ----------------------------------------------------------------- tracing
    def _flatten_args(self, args):
        """-> (leaves, in_tree, var_mask) ; var_mask[i] True if leaf i is a variable."""
        if any(i >= len(args) for i in self.param_argnums):
            raise ValueError(f"param_argnums {self.param_argnums} out of range for {len(args)} arguments")
        leaves, mask = [], []
        for i, a in enumerate(args):
            ls, _ = jax.tree_util.tree_flatten(a)
            leaves += ls
            mask += [i not in self.param_argnums] * len(ls)
        _, in_tree = jax.tree_util.tree_flatten(args)
        return leaves, in_tree, mask

    def _trace(self, args):
        """Trace f at the argument shapes -> SX symbols, SX outputs, output shapes, out tree."""
        leaves, in_tree, mask = self._flatten_args(args)
        dtype = self.dtype or _infer_dtype(leaves)
        sig = _signature(leaves, dtype)
        if sig in self._traces:
            return self._traces[sig]
        avals = [jax.ShapeDtypeStruct(np.shape(a), dtype) for a in leaves]

        def flat_f(*flat):
            return self.fun(*jax.tree_util.tree_unflatten(in_tree, flat))

        closed, out_shape = jax.make_jaxpr(flat_f, return_shape=True)(*avals)
        out_leaves, out_tree = jax.tree_util.tree_flatten(out_shape)
        syms, arrs = [], []
        for i, a in enumerate(avals):
            s, arr = sx_inputs(f"a{i}", a.shape)
            syms.append(s)
            arrs.append(arr)
        outs = eval_jaxpr(closed, arrs)
        ys = [to_sx_vector(o) for o in outs]
        var_syms = [s for s, m in zip(syms, mask) if m]
        if not var_syms:
            raise ValueError("cusadi2.jax: every argument is a parameter; nothing to differentiate")
        tr = dict(sig=sig, dtype=dtype, syms=syms, ys=ys, out_shapes=[tuple(o.shape) for o in out_leaves],
                  out_tree=out_tree, x=ca.vertcat(*var_syms), mask=mask)
        self._traces[sig] = tr
        return tr

    # ----------------------------------------------------------------- plans
    def _plan(self, kind: str, args):
        tr = self._trace(args)
        key = (kind, tr["sig"])
        if key in self._plans:
            return self._plans[key]
        syms, ys, x = tr["syms"], tr["ys"], tr["x"]
        name = f"{self.name}_{kind}"
        coo = None
        extra = []
        if kind == "value":
            outs = [ca.densify(y) for y in ys]
        elif kind == "jac":
            outs = [ca.sparsify(ca.jacobian(y, x)) for y in ys]
            coo = [_coo2(J) for J in outs]
        elif kind == "hess":
            outs, coo = [], []
            for y in ys:
                H = ca.sparsify(ca.jacobian(ca.jacobian(y, x), x))     # (n_y*n_x) x n_x, rows column-major
                outs.append(H)
                coo.append(_coo3(H, int(y.numel())))
        elif kind == "whess":
            lams = [ca.SX.sym(f"lam{k}", int(y.numel())) for k, y in enumerate(ys)]
            L = sum((ca.dot(l, y) for l, y in zip(lams, ys)), ca.SX(0))
            H = ca.sparsify(ca.hessian(L, x)[0])
            outs, coo, extra = [H], [_coo2(H)], lams
        else:
            raise ValueError(kind)
        fun = ca.Function(name, list(syms) + extra, outs)
        plan = _Plan(name, fun, tr["out_shapes"], coo, tr["dtype"], self.block_size, self.serial, self.max_live)
        self._plans[key] = plan
        return plan

    # ----------------------------------------------------------------- calls
    def __call__(self, *args):
        leaves, _, _ = self._flatten_args(args)
        tr = self._trace(args)
        outs = self._plan("value", args)(leaves)
        return jax.tree_util.tree_unflatten(tr["out_tree"], outs)

    def jacobian(self) -> "SparseDerivative":
        return SparseDerivative(self, "jac")

    def hessian(self, weighted: bool = False) -> "SparseDerivative":
        return SparseDerivative(self, "whess" if weighted else "hess")

    def free(self):
        for p in self._plans.values():
            p.free()
        self._plans.clear()


class SparseDerivative:
    """A Jacobian / Hessian of a JaxFunction as its own plan.

    `d(*args)` returns the nonzero values, one (nnz,) array per output of f
    (a single array if f returns one array), batched to (B, nnz) under vmap.
    `d.coo(*args)` returns the matching coordinates as int32 device arrays of
    shape (nnz, 2) [row, col] for Jacobians and weighted Hessians, or (nnz, 3)
    [output element, i, j] for true Hessians, in CasADi's column-major nonzero
    order. `d.shape(*args)` gives the dense shapes. Both are fixed per argument
    signature and involve no GPU work.
    """

    def __init__(self, parent: JaxFunction, kind: str):
        self.parent, self.kind = parent, kind

    def _leaves(self, args, lam):
        """Flat plan inputs: f's argument leaves, then (weighted Hessian only) one
        multiplier array per output of f, passed as `lam=` (an array, or a tuple
        of arrays when f has several outputs)."""
        leaves, _, _ = self.parent._flatten_args(args)
        if self.kind != "whess":
            if lam is not None:
                raise ValueError("lam= is only accepted by hessian(weighted=True)")
            return leaves
        ys = self.parent._trace(args)["ys"]
        if lam is None:
            raise ValueError("hessian(weighted=True) needs lam=<multipliers>, one array per output of f")
        lams = list(lam) if isinstance(lam, (tuple, list)) else [lam]
        if len(lams) != len(ys):
            raise ValueError(f"lam= has {len(lams)} arrays, f has {len(ys)} outputs")
        for l, y in zip(lams, ys):
            if int(np.prod(np.shape(l), dtype=np.int64)) != int(y.numel()):
                raise ValueError(f"multiplier of size {np.shape(l)} does not match an output of {int(y.numel())} elements")
        return leaves + [jnp.asarray(l).astype(self.parent._trace(args)["dtype"]) for l in lams]

    def __call__(self, *args, lam=None):
        leaves = self._leaves(args, lam)
        plan = self.parent._plan(self.kind, args)
        outs = plan(leaves)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def coo(self, *args):
        plan = self.parent._plan(self.kind, args)
        cs = [jnp.asarray(c) for c in plan.coo]
        return cs[0] if len(cs) == 1 else tuple(cs)

    def shape(self, *args):
        f_args = args
        tr = self.parent._trace(f_args)
        n_x = int(tr["x"].numel())
        if self.kind == "jac":
            shapes = [(int(y.numel()), n_x) for y in tr["ys"]]
        elif self.kind == "hess":
            shapes = [(int(y.numel()), n_x, n_x) for y in tr["ys"]]
        else:
            shapes = [(n_x, n_x)]
        return shapes[0] if len(shapes) == 1 else tuple(shapes)


def _casadi_name(name: str) -> str:
    """A CasADi-valid function name: letters, digits, single underscores, starts
    with a letter, not a reserved word ('<lambda>' and the like become 'f')."""
    import re
    n = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", str(name))).strip("_")
    if not n or not n[0].isalpha() or n in ("null", "jac", "hess"):
        n = "f_" + n if n else "f"
    return n


def _coo2(M: ca.SX) -> np.ndarray:
    rows, cols = M.sparsity().get_triplet()
    return np.stack([np.asarray(rows, np.int32), np.asarray(cols, np.int32)], axis=1).reshape(-1, 2)


def _coo3(H: ca.SX, n_y: int) -> np.ndarray:
    rows, cols = H.sparsity().get_triplet()
    rows = np.asarray(rows, np.int64)
    k, i = rows % n_y, rows // n_y            # row r = k + n_y * i  (column-major vec of J[k, i])
    return np.stack([k, i, np.asarray(cols, np.int64)], axis=1).astype(np.int32).reshape(-1, 3)


def jax_function(fun: Callable = None, *, param_argnums: Sequence[int] = (), **kw) -> JaxFunction:
    """Wrap `fun` (plain JAX code) for cusadi2 evaluation; usable as a decorator.
    Keyword options: dtype (default: from the arguments), block_size, serial, max_live."""
    if fun is None:
        return lambda f: JaxFunction(f, param_argnums=param_argnums, **kw)
    return JaxFunction(fun, param_argnums=param_argnums, **kw)
