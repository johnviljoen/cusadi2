"""Stage-parallel plans from a per-stage JAX function (cusadi2's `n_stages` mode).

An MPC's constraint vector is N copies of one stage function. Tracing the whole
horizon gives one huge graph whose Jacobian / Hessian plans no longer fit the
level-parallel kernel (Go2 N = 28: 371k / 594k live slots against 12,672).
Here the stage function is traced once and cusadi2 replicates it: the plan is
the per-stage graph, the kernel grid is (batch, n_stages), and each argument
declares how its buffer is indexed per stage:

    mode "stage"    one block per interval:   buffer (B, n_stages * size),     stage s reads block s
    mode "coupled"  knot quantities:          buffer (B, (n_stages + span) * size), stage s reads block s + offset
    mode "shared"   one block for all stages: buffer (B, size)

    sf = StagedFunction(c_stage, n_stages=N, specs=[
            dict(name="x_k",  size=37, mode="coupled", offset=0, span=1),
            dict(name="x_k1", size=37, mode="coupled", offset=1, span=1, alias="x_k"),   # same buffer as x_k
            dict(name="u_k",  size=42, mode="stage"),
            dict(name="p",    size=49, mode="shared", param=True),
            dict(name="c_k",  size=4,  mode="stage",  param=True)])
    y   = sf(X, X, U, P, C)          # (B, N * ny): stage-major
    jac = sf.jacobian();  J = jac(X, X, U, P, C)   # (B, N * nnz) values; jac.coo: stage-local (row, col)
    whs = sf.hessian(weighted=True);  H = whs(X, X, U, P, C, lam=LAM)   # lam: (B, N * ny)

Variables of a stage are the non-parameter arguments concatenated in order
(the stage window); an argument with `alias=<other>` reads the other's buffer
(the window x_k, x_{k+1} is one knot buffer passed twice) and shares its global block; `global_pattern` maps the stage-local COO to the global
(row, col) of the whole-horizon function for a knot-major global layout
(all knots of argument 0, then all stages of argument 1, ...), which is what a
consumer like cuipm needs to assemble the KKT system from the stage-major
value stream.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np

from cusadi2.compose import gather_block, gather_shared
from cusadi2.dag import STAGE_MODES, stage_count
from cusadi2.jax.function import _casadi_name, _coo2
from cusadi2.jax.interpret import eval_jaxpr, sx_inputs, to_sx_vector
from cusadi2.jax.primitive import MAX_ARGS, cusadi2_eval, ext_for
from cusadi2.plan import device_max_live, plan_bytes

_plan_ids = itertools.count(10_000_000)


class StageLayout:
    """The global layouts a list of stage specs implies, and the gather table of
    every argument. Variables (specs without `param`) form the knot-major global
    x: all knots of the first variable argument, then all of the second, ...;
    an argument with `alias=<name>` reads the block of the argument it aliases
    (x_k and x_{k+1} are one knot buffer read twice). Parameters form p the same
    way, shared ones first, then the per-stage ones, each argument's stages
    contiguous. An argument with `const=<array (n_stages, size), or (size,) if
    shared>` is neither: its values are baked into the plan (stage constants: a
    contact schedule, per-stage weights). `gather(i)` is (0, table) for a
    variable (read from x), (1, table) for a parameter (read from p) or
    (-1, table) for a constant (read from `consts`), table (n_stages, size) per
    `cusadi2.compose`."""

    def __init__(self, specs: Sequence[dict], n_stages: int):
        self.specs = [dict(s) for s in specs]
        self.n_stages = int(n_stages)
        names = [s["name"] for s in self.specs]
        for s in self.specs:
            s.setdefault("mode", "stage"); s.setdefault("offset", 0); s.setdefault("span", 0)
            s.setdefault("param", False); s.setdefault("alias", None); s.setdefault("const", None)
            if s["const"] is not None:
                if s["param"] or s["alias"]:
                    raise ValueError(f"{s['name']}: a constant argument is neither a parameter nor an alias")
                s["const"] = np.asarray(s["const"], np.float64).reshape(-1, s["size"])
            if s["mode"] not in STAGE_MODES:
                raise ValueError(f"{s['name']}: unknown mode {s['mode']!r}")
            if s["mode"] != "coupled" and (s["offset"] or s["span"]):
                raise ValueError(f"{s['name']}: offset / span are only valid for coupled mode")
        self.var_args = [i for i, s in enumerate(self.specs) if not s["param"] and s["const"] is None]
        self.param_args = [i for i, s in enumerate(self.specs) if s["param"]]
        self.const_args = [i for i, s in enumerate(self.specs) if s["const"] is not None]
        self.count = [stage_count(s["mode"], s["span"], self.n_stages) for s in self.specs]
        for i in self.const_args:
            s = self.specs[i]
            if s["const"].shape != (self.count[i], s["size"]):
                raise ValueError(f"{s['name']}: const needs shape ({self.count[i]}, {s['size']}), got {s['const'].shape}")

        def layout(order):
            base, at = {}, 0
            for i in order:
                if self.specs[i]["alias"]:
                    continue
                base[i] = at
                at += self.count[i] * self.specs[i]["size"]
            for i in order:
                s = self.specs[i]
                if s["alias"]:
                    j = names.index(s["alias"])
                    if j not in base or self.specs[j]["size"] != s["size"] or self.count[j] != self.count[i]:
                        raise ValueError(f"{s['name']}: alias must name a non-alias argument of the same kind, size and stage count")
                    base[i] = base[j]
            return base, at
        self.x_base, self.nx = layout(self.var_args)
        shared_first = ([i for i in self.param_args if self.specs[i]["mode"] == "shared"]
                        + [i for i in self.param_args if self.specs[i]["mode"] != "shared"])
        self.p_base, self.np = layout(shared_first)
        self.c_base, self.nconst = layout(self.const_args)
        self.consts = (np.concatenate([self.specs[i]["const"].reshape(-1) for i in self.const_args])
                       if self.const_args else np.zeros(0))
        self.stage_start = {}
        at = 0
        for i in self.var_args:
            self.stage_start[i] = at
            at += self.specs[i]["size"]
        self.nx_stage = at

    def gather(self, i: int):
        s = self.specs[i]
        src, base = ((-1, self.c_base[i]) if s["const"] is not None else (1, self.p_base[i]) if s["param"]
                     else (0, self.x_base[i]))
        if s["mode"] == "shared":
            return src, gather_shared(self.n_stages, s["size"], base)
        return src, gather_block(self.n_stages, s["size"], base, s["size"], s["offset"], self.count[i])

    def global_cols(self, local_cols, s: int) -> np.ndarray:
        """Global x columns of entries of the stage-s variable vector (the variable
        arguments concatenated in spec order); -1 where the argument's knot
        s + offset lies outside its buffer (such an argument reads zero there and
        its derivative columns are dropped, so the stage function must not depend
        on it at that stage: gate the term with a zero stage constant)."""
        local_cols = np.asarray(local_cols, np.int64)
        out = np.empty(len(local_cols), np.int64)
        for i in self.var_args:
            sp = self.specs[i]
            start, size = self.stage_start[i], sp["size"]
            sel = (local_cols >= start) & (local_cols < start + size)
            knot = 0 if sp["mode"] == "shared" else s + sp["offset"]
            out[sel] = (self.x_base[i] + knot * size + (local_cols[sel] - start)) if 0 <= knot < self.count[i] else -1
        return out

    def stage_x_cols(self) -> np.ndarray:
        """(n_stages, nx_stage) global column of every stage variable."""
        return np.stack([self.global_cols(np.arange(self.nx_stage), s) for s in range(self.n_stages)])

    def blocks(self, kind: str):
        """[(name, base, count, size)] of the non-alias arguments making up x ("x") or p ("p")."""
        args, base = (self.var_args, self.x_base) if kind == "x" else (self.param_args, self.p_base)
        out = [(self.specs[i]["name"], int(base[i]), self.count[i], self.specs[i]["size"])
               for i in args if not self.specs[i]["alias"]]
        return sorted(out, key=lambda t: t[1])

    def pack(self, kind: str, values: dict) -> np.ndarray:
        """Assemble x or p from per-argument arrays (count, size) (or (size,) for shared)."""
        n = self.nx if kind == "x" else self.np
        out = np.zeros(n)
        for name, base, count, size in self.blocks(kind):
            v = np.asarray(values[name], np.float64).reshape(-1)
            if v.size != count * size:
                raise ValueError(f"{name}: expected {count} x {size} values, got {v.size}")
            out[base:base + v.size] = v
        return out


class _StagedPlan:
    def __init__(self, name, fun, n_stages, specs, dtype, nnz_out, max_live, serial):
        self.plan_id = next(_plan_ids)
        self.fun, self.n_stages, self.specs = fun, n_stages, specs
        self.dtype = np.dtype(dtype)
        self.ext = ext_for(self.dtype)
        self.nnz_out = [int(n) for n in nnz_out]
        self.in_len = [stage_count(s["mode"], s.get("span", 0), n_stages) * s["size"] for s in specs]
        in_mode = [s["mode"] for s in specs]
        in_offset = [int(s.get("offset", 0)) for s in specs]
        in_span = [int(s.get("span", 0)) for s in specs]
        if max_live is None:
            max_live = device_max_live("float" if self.dtype == np.float32 else "double")
        self.serial = bool(serial)
        if serial:
            data = plan_bytes(fun, n_stages=n_stages, in_mode=in_mode, in_offset=in_offset, in_span=in_span)
        else:
            try:
                data = plan_bytes(fun, n_stages=n_stages, in_mode=in_mode, in_offset=in_offset, in_span=in_span,
                                  max_live=max_live)
            except ValueError:
                data = plan_bytes(fun, n_stages=n_stages, in_mode=in_mode, in_offset=in_offset, in_span=in_span)
                self.serial = True
        self.ext.install_plan(self.plan_id, data, 0, self.serial)

    def __call__(self, buffers):
        xs = []
        for a, n in zip(buffers, self.in_len):
            a = jnp.asarray(a).astype(self.dtype).reshape(1, n)
            xs.append(a)
        outs = cusadi2_eval(xs, plan_id=self.plan_id, batch=1, dtype=self.dtype,
                            nnz_out=tuple(self.n_stages * n for n in self.nnz_out))
        return [o.reshape(-1) for o in outs]

    def free(self):
        self.ext.free_plan(self.plan_id)


class StagedFunction:
    """See module docstring. `stage_fun(*args)` returns one array per stage; `specs` one dict
    per argument with name, size, mode, offset, span, param."""

    def __init__(self, stage_fun, n_stages: int, specs: Sequence[dict], name: Optional[str] = None,
                 dtype=np.float64, max_live: Optional[int] = None, serial: bool = False):
        if len(specs) > MAX_ARGS - 1:
            raise ValueError(f"at most {MAX_ARGS - 1} stage arguments")
        self.fun, self.n_stages, self.specs = stage_fun, int(n_stages), [dict(s) for s in specs]
        self.name = _casadi_name(name or getattr(stage_fun, "__name__", "staged"))
        self.dtype, self.max_live, self.serial = np.dtype(dtype), max_live, serial
        self._plans: Dict[str, _StagedPlan] = {}
        self._trace()

    def _trace(self):
        syms, arrs = [], []
        for s in self.specs:
            sym, arr = sx_inputs(s["name"], (s["size"],))
            syms.append(sym)
            arrs.append(arr)
        avals = [jax.ShapeDtypeStruct((s["size"],), self.dtype) for s in self.specs]
        closed = jax.make_jaxpr(self.fun)(*avals)
        outs = eval_jaxpr(closed, arrs)
        if len(outs) != 1:
            raise ValueError("staged: the stage function must return one array")
        self.y = to_sx_vector(outs[0])
        self.ny = int(self.y.numel())
        self.syms = syms
        self.layout = StageLayout(self.specs, self.n_stages)
        self.var_args = self.layout.var_args
        self.x = ca.vertcat(*[syms[i] for i in self.var_args])
        self.nx_stage = int(self.x.numel())
        self.nx_global = self.layout.nx

    def _plan(self, kind: str) -> _StagedPlan:
        if kind in self._plans:
            return self._plans[kind]
        extra, specs = [], list(self.specs)
        if kind == "value":
            out, coo = ca.densify(self.y), None
        elif kind == "jac":
            out = ca.sparsify(ca.jacobian(self.y, self.x))
            coo = _coo2(out)
        elif kind == "whess":
            lam = ca.SX.sym("lam", self.ny)
            out = ca.sparsify(ca.hessian(ca.dot(lam, self.y), self.x)[0])
            coo = _coo2(out)
            extra = [lam]
            specs = specs + [dict(name="lam", size=self.ny, mode="stage", param=True)]
        else:
            raise ValueError(kind)
        fun = ca.Function(f"{self.name}_{kind}", list(self.syms) + extra, [out])
        plan = _StagedPlan(f"{self.name}_{kind}", fun, self.n_stages, specs, self.dtype, [fun.nnz_out(0)],
                           self.max_live, self.serial)
        plan.coo = coo
        self._plans[kind] = plan
        return plan

    # ---- calls (vmap over a leading batch axis of every buffer)
    def __call__(self, *buffers):
        return self._plan("value")(buffers)[0]

    def jacobian(self):
        return _StagedDerivative(self, "jac")

    def hessian(self, weighted: bool = True):
        if not weighted:
            raise NotImplementedError("staged: only the Lagrangian Hessian")
        return _StagedDerivative(self, "whess")

    def free(self):
        for p in self._plans.values():
            p.free()
        self._plans.clear()

    # ---- patterns
    def global_pattern(self, kind: str):
        """(rows, cols) int32 of the whole-horizon derivative for the knot-major global layout,
        in the stage-major order the value stream has: entry j of stage s is (rows[s*nnz + j],
        cols[s*nnz + j]). Jacobian rows are s*ny + r; Hessian rows and cols are both global x."""
        plan = self._plan(kind)
        local = np.asarray(plan.coo)
        rows, cols = [], []
        for s in range(self.n_stages):
            if kind == "jac":
                rows.append(s * self.ny + local[:, 0])
                cols.append(self.layout.global_cols(local[:, 1], s))
            else:
                rows.append(self.layout.global_cols(local[:, 0], s))
                cols.append(self.layout.global_cols(local[:, 1], s))
        return np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32)

    def global_offsets(self):
        """Start of each variable argument's block in the global x, and (offset, size) per arg.
        Aliased arguments share the block of the argument they alias."""
        offs = {i: (int(self.specs[i].get("offset", 0)), self.specs[i]["size"]) for i in self.var_args}
        return dict(self.layout.x_base), offs

    def global_x(self, *buffers):
        """Concatenate the variable arguments' (B, ...) buffers into the knot-major global x."""
        return jnp.concatenate([jnp.asarray(buffers[i]).reshape(jnp.shape(buffers[i])[0], -1)
                                for i in self.var_args if not self.specs[i].get("alias")], axis=1)


class _StagedDerivative:
    def __init__(self, parent: StagedFunction, kind: str):
        self.parent, self.kind = parent, kind

    def __call__(self, *buffers, lam=None):
        plan = self.parent._plan(self.kind)
        bufs = list(buffers) + ([lam] if self.kind == "whess" else [])
        return plan(bufs)[0]

    @property
    def coo(self):
        return self.parent._plan(self.kind).coo

    def global_pattern(self):
        return self.parent.global_pattern(self.kind)
