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

from cusadi2.dag import stage_count
from cusadi2.jax.function import _casadi_name, _coo2
from cusadi2.jax.interpret import eval_jaxpr, sx_inputs, to_sx_vector
from cusadi2.jax.primitive import MAX_ARGS, cusadi2_eval, ext_for
from cusadi2.plan import device_max_live, plan_bytes

_plan_ids = itertools.count(10_000_000)


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
        self.var_args = [i for i, s in enumerate(self.specs) if not s.get("param", False)]
        self.x = ca.vertcat(*[syms[i] for i in self.var_args])
        self.nx_stage = int(self.x.numel())

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
        base, offs = self.global_offsets()
        local = np.asarray(plan.coo)
        rows, cols = [], []
        for s in range(self.n_stages):
            if kind == "jac":
                rows.append(s * self.ny + local[:, 0])
                cols.append(self._global_cols(local[:, 1], s, base, offs))
            else:
                rows.append(self._global_cols(local[:, 0], s, base, offs))
                cols.append(self._global_cols(local[:, 1], s, base, offs))
        return np.concatenate(rows).astype(np.int32), np.concatenate(cols).astype(np.int32)

    def global_offsets(self):
        """Start of each variable argument's block in the global x, and (offset, size) per arg.
        Aliased arguments share the block of the argument they alias."""
        names = [sp["name"] for sp in self.specs]
        base, offs, at = {}, {}, 0
        for i in self.var_args:
            sp = self.specs[i]
            offs[i] = (int(sp.get("offset", 0)), sp["size"])
            if sp.get("alias"):
                continue
            base[i] = at
            at += stage_count(sp["mode"], sp.get("span", 0), self.n_stages) * sp["size"]
        for i in self.var_args:
            sp = self.specs[i]
            if sp.get("alias"):
                base[i] = base[names.index(sp["alias"])]
        self.nx_global = at
        return base, offs

    def _global_cols(self, local_cols, s, base, offs):
        out = np.empty(len(local_cols), dtype=np.int64)
        start = 0
        for i in self.var_args:
            off, size = offs[i]
            sel = (local_cols >= start) & (local_cols < start + size)
            knot = 0 if self.specs[i]["mode"] == "shared" else s + off
            out[sel] = base[i] + knot * size + (local_cols[sel] - start)
            start += size
        return out

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
