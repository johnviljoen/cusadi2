"""Export a JAX-defined NLP to the nine cusadi2 plans cuipm loads.

    from cusadi2.jax.ipm import export_cuipm
    info = export_cuipm(f, c, d, x0, p0, "plans/")

cuipm (`~/code/cuipm`) solves  min f(x, p)  s.t.  c(x, p) = 0,  d(x, p) >= 0,  x_L <= x <= x_U
and reads the problem from nine plan files with fixed names and input signatures
(the convention of `cuipm/codegen/generate.py`; dims and sparsity are taken from
the plans themselves, bounds are runtime arrays):

    f.bin       f(x, p, eps)      -> scalar
    c.bin       c(x, p)           -> (nyc,)
    d.bin       d(x, p)           -> (nyd,)
    jac_f.bin   jac_f(x, p, eps)  -> grad f as a sparse column (nx x 1)
    jac_c.bin   jac_c(x, p)       -> (nyc x nx), structural nonzeros
    jac_d.bin   jac_d(x, p)       -> (nyd x nx)
    hess_f.bin  hess_f(x, p, eps) -> (nx x nx)
    hess_c.bin  hess_c(x, p, lam_c) -> Hessian of lam_c . c  (nx x nx)
    hess_d.bin  hess_d(x, p, lam_d) -> Hessian of lam_d . d  (nx x nx)

`eps` is a one-element input that cuipm's drivers pass as zero; the hand-written
codegen used it to inflate the cost's sparsity pattern to what jax2sympy
reported. The patterns here are exact, so eps is accepted and ignored (kept so
that the plans have the input count the drivers expect).

f, c and d are JAX callables of (x, p) with x and p flat 1-D arrays; their
jaxpr is interpreted into CasADi SX (`cusadi2.jax.interpret`) on shared
symbols, the derivatives are formed symbolically and each of the nine
functions is written as a level-parallel plan with `cusadi2.write_plan`.

Stage-wise problems (`StagedNLP`, `export_cuipm_staged`): f, c and d given as
per-stage JAX callables of the stage arguments (specs as in
`cusadi2.jax.staged`) are traced once and composed (`cusadi2.compose`) into
the same nine plans with the same whole-horizon signatures: the kernel runs
the stage graph per (instance, stage), gathers x_k / x_{k+1} / u_k from the
knot-major x and the parameters from p, scatters each stage's rows into the
global pattern, and sums the Hessian blocks that meet on a knot. cuipm needs
nothing but the plan files; the stage structure is inside them.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import casadi as ca
import jax
import numpy as np

from cusadi2 import device_max_live, extract_level_plan, write_plan
from cusadi2.compose import Composition, DeclaredArg, compose, evaluate_composition
from cusadi2.jax.function import _coo2
from cusadi2.jax.interpret import eval_jaxpr, sx_inputs, to_sx_vector
from cusadi2.jax.staged import StageLayout

NAMES = ("f", "c", "d", "jac_f", "jac_c", "jac_d", "hess_f", "hess_c", "hess_d")


def _trace(fun: Callable, arrs, avals) -> ca.SX:
    closed = jax.make_jaxpr(fun)(*avals)
    outs = eval_jaxpr(closed, list(arrs))
    if len(outs) != 1:
        raise ValueError("ipm: f, c and d must each return one array")
    return to_sx_vector(outs[0])


def trace_nlp(f: Callable, c: Callable, d: Optional[Callable], nx: int, np_: int, dtype=np.float64):
    """(x, p, f_sx, c_sx, d_sx): the NLP as CasADi SX on shared symbols x (nx) and p (np)."""
    x_sym, x_arr = sx_inputs("x", (nx,))
    p_sym, p_arr = sx_inputs("p", (np_,))
    avals = [jax.ShapeDtypeStruct((nx,), dtype), jax.ShapeDtypeStruct((np_,), dtype)]
    f_sx = _trace(f, (x_arr, p_arr), avals)
    if f_sx.numel() != 1:
        raise ValueError("ipm: f must be scalar")
    c_sx = _trace(c, (x_arr, p_arr), avals)
    d_sx = _trace(d, (x_arr, p_arr), avals) if d is not None else ca.SX.zeros(0, 1)
    return x_sym, p_sym, f_sx, c_sx, d_sx


def solve_ipopt(f: Callable, c: Callable, d: Optional[Callable], x0, p0, *, x_L=None, x_U=None,
                d_L=None, d_U=None, options: Optional[dict] = None, nlp=None) -> dict:
    """Solve the same NLP with IPOPT through CasADi's nlpsol (reference / baseline):
    min f  s.t.  c = 0,  d_L <= d <= d_U,  x_L <= x <= x_U. Derivatives are CasADi's
    own on the traced SX. Returns x, lam_c, lam_d, f, status, iterations, wall time."""
    import time
    x0, p0 = np.asarray(x0, float).reshape(-1), np.asarray(p0, float).reshape(-1)
    x, p, f_sx, c_sx, d_sx = nlp or trace_nlp(f, c, d, x0.size, p0.size)
    nyc, nyd = int(c_sx.numel()), int(d_sx.numel())
    g = ca.vertcat(c_sx, d_sx)
    opts = {"ipopt": {"print_level": 0, "sb": "yes"}, "print_time": False}
    for k, v in (options or {}).items():
        opts["ipopt"][k] = v
    solver = ca.nlpsol("solver", "ipopt", {"x": x, "p": p, "f": f_sx, "g": g}, opts)
    lbg = np.concatenate([np.zeros(nyc), np.zeros(nyd) if d_L is None else np.asarray(d_L, float)])
    ubg = np.concatenate([np.zeros(nyc), np.full(nyd, np.inf) if d_U is None else np.asarray(d_U, float)])
    t = time.perf_counter()
    res = solver(x0=x0, p=p0, lbx=-np.inf if x_L is None else np.asarray(x_L, float),
                 ubx=np.inf if x_U is None else np.asarray(x_U, float), lbg=lbg, ubg=ubg)
    wall = time.perf_counter() - t
    st = solver.stats()
    lam_g = np.asarray(res["lam_g"]).reshape(-1)
    return {"x": np.asarray(res["x"]).reshape(-1), "f": float(res["f"]), "lam_c": lam_g[:nyc], "lam_d": lam_g[nyc:],
            "g": np.asarray(res["g"]).reshape(-1), "status": st.get("return_status"), "success": bool(st.get("success")),
            "iterations": int(st.get("iter_count", -1)), "wall_s": wall}


def build_functions(f: Callable, c: Callable, d: Optional[Callable], nx: int, np_: int,
                    dtype=np.float64, nlp=None) -> Dict[str, ca.Function]:
    """The nine CasADi Functions in cuipm's signatures, from the JAX callables."""
    x_sym, p_sym, f_sx, c_sx, d_sx = nlp or trace_nlp(f, c, d, nx, np_, dtype)
    eps = ca.SX.sym("eps")
    lam_c = ca.SX.sym("lam_c", c_sx.numel())
    lam_d = ca.SX.sym("lam_d", d_sx.numel())
    x, p = x_sym, p_sym
    grad_f = ca.sparsify(ca.jacobian(f_sx, x).T)       # column with the structural pattern (ca.gradient densifies)
    return {
        "f":      ca.Function("f",      [x, p, eps],   [f_sx]),
        "c":      ca.Function("c",      [x, p],        [c_sx]),
        "d":      ca.Function("d",      [x, p],        [d_sx]),
        "jac_f":  ca.Function("jac_f",  [x, p, eps],   [grad_f]),
        "jac_c":  ca.Function("jac_c",  [x, p],        [ca.sparsify(ca.jacobian(c_sx, x))]),
        "jac_d":  ca.Function("jac_d",  [x, p],        [ca.sparsify(ca.jacobian(d_sx, x))]),
        "hess_f": ca.Function("hess_f", [x, p, eps],   [ca.sparsify(ca.hessian(f_sx, x)[0])]),
        "hess_c": ca.Function("hess_c", [x, p, lam_c], [ca.sparsify(ca.hessian(ca.dot(lam_c, c_sx), x)[0])]),
        "hess_d": ca.Function("hess_d", [x, p, lam_d], [ca.sparsify(ca.hessian(ca.dot(lam_d, d_sx), x)[0])]),
    }


def export_cuipm(f: Callable, c: Callable, d: Optional[Callable], x0, p0, out_dir: str, *,
                 x_L=None, x_U=None, d_L=None, d_U=None, max_live: Optional[int] = None,
                 block_size: Optional[int] = None, nlp=None) -> dict:
    """Write the nine plans to `out_dir` and `problem.json` next to them.

    x0, p0: example arrays (only their shapes matter for the plans; they are recorded
    in problem.json as the initial point / parameter). Bounds, if given, are recorded
    in problem.json for the driver (cuipm takes them at runtime, not from the plans).
    `max_live` windows the level plans to a shared-memory budget (see
    `cusadi2.device_max_live`); default is no windowing, as cuipm's codegen did.
    Returns dims, nnz per plan and the file paths."""
    x0, p0 = np.asarray(x0, dtype=np.float64).reshape(-1), np.asarray(p0, dtype=np.float64).reshape(-1)
    fns = build_functions(f, c, d, x0.size, p0.size, nlp=nlp)
    os.makedirs(out_dir, exist_ok=True)
    info = {"nx": int(x0.size), "np": int(p0.size), "nyc": int(fns["c"].nnz_out(0)), "nyd": int(fns["d"].nnz_out(0)),
            "plans": {}, "x0": x0.tolist(), "p0": p0.tolist()}
    for k, v in (("x_L", x_L), ("x_U", x_U), ("d_L", d_L), ("d_U", d_U)):
        if v is not None:
            info[k] = np.asarray(v, dtype=np.float64).reshape(-1).tolist()
    for name in NAMES:
        fn = fns[name]
        plan = extract_level_plan(fn, max_live=max_live)
        path = write_plan(fn, os.path.join(out_dir, f"{name}.bin"), block_size=block_size, plan=plan)
        sp = fn.sparsity_out(0)
        info["plans"][name] = {"path": path, "rows": sp.size1(), "cols": sp.size2(), "nnz": sp.nnz(),
                               "n_in": fn.n_in(), "nnz_in": [fn.nnz_in(a) for a in range(fn.n_in())],
                               "peak_live_slots": plan.peak_live_slots, "n_levels": plan.n_levels,
                               "n_nodes": plan.n_nodes}
    with open(os.path.join(out_dir, "problem.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    info["functions"] = fns
    return info


class GpuFunction:
    """One of the nine exported Functions uploaded as a cusadi2 plan: `g(x, p, ...)` on flat
    1-D arrays returns the structural nonzeros of the output (vmap / jit as usual);
    `coo` is the (nnz, 2) row/col pattern, `serial` whether the level kernel fit."""

    def __init__(self, name: str, fn, dtype=np.float64, max_live: Optional[int] = None,
                 block_size: Optional[int] = None, serial: bool = False):
        from cusadi2.jax.function import _Plan
        if isinstance(fn, ComposedFunction):
            self.coo, self.nnz, composition, fn = fn.coo, fn.nnz_out, fn.composition, fn.fn
        else:
            rows, cols = fn.sparsity_out(0).get_triplet()
            self.coo = np.stack([np.asarray(rows, np.int32), np.asarray(cols, np.int32)], axis=1).reshape(-1, 2)
            self.nnz, composition = int(fn.nnz_out(0)), None
        self.name = name
        self._plan = _Plan(name, fn, [(self.nnz,)], [self.coo], dtype, block_size, serial, max_live,
                           composition=composition)
        self.serial = self._plan.serial

    def __call__(self, *args):
        return self._plan(list(args))[0]

    def free(self):
        self._plan.free()


def gpu_functions(fns: Dict[str, object], dtype=np.float64, max_live: Optional[int] = None,
                  block_size: Optional[int] = None, serial: bool = False) -> Dict[str, GpuFunction]:
    """The Functions `export_cuipm` / `export_cuipm_staged` returned (`info["functions"]`) as
    cusadi2 plans on the GPU, without tracing or differentiating again; same signatures as the
    plans (f / jac_f / hess_f take the ignored `eps` as a third argument). Level plans are
    windowed to the device's shared memory (`max_live=None`) and fall back to the serial kernel
    when no window fits."""
    return {name: GpuFunction(name, fn, dtype, max_live, block_size, serial) for name, fn in fns.items()}


# ============================================================================ stage-wise problems
def rows_active(n_stages: int, blocks) -> np.ndarray:
    """(n_stages, total rows) bool: at which stages each row of a stage output exists.
    `blocks` lists (size, when) in the output's row order, `when` one of "all",
    "first", "last" or a bool sequence of n_stages. Rows inactive at a stage are
    computed and discarded, so a constraint that holds only at one knot (the
    initial state, a touchdown) is a block of the stage function and the global
    pattern has exactly the rows that exist."""
    cols = []
    for size, when in blocks:
        if isinstance(when, str):
            if when == "all":
                m = np.ones(n_stages, bool)
            elif when == "first":
                m = np.arange(n_stages) == 0
            elif when == "last":
                m = np.arange(n_stages) == n_stages - 1
            else:
                raise ValueError(f"rows_active: unknown block spec {when!r}")
        else:
            m = np.asarray(when, bool).reshape(-1)
            if m.size != n_stages:
                raise ValueError(f"rows_active: a bool block spec needs {n_stages} entries")
        cols.append(np.repeat(m[:, None], size, axis=1))
    return np.concatenate(cols, axis=1) if cols else np.zeros((n_stages, 0), bool)


def _row_numbers(active: np.ndarray, order: str = "stage") -> np.ndarray:
    """Global row of every active (stage, row): "stage" order numbers stage by stage
    (all of stage 0's active rows, then stage 1's, ...); "block" order takes the
    blocks of the stage output (runs of rows with the same activity over the
    stages) one after the other, each stage by stage, as a hand-written
    whole-horizon function listing its initial-state rows before the dynamics
    defects does; -1 where inactive."""
    n_stages, ny = active.shape
    rows = np.full(active.shape, -1, np.int64)
    if order == "stage":
        rows[active] = np.arange(int(active.sum()))
    elif order == "block":
        change = np.ones(ny, bool)
        change[1:] = np.any(active[:, 1:] != active[:, :-1], axis=0)
        block = np.cumsum(change) - 1                                   # block id per row
        s, r = np.nonzero(active)
        order_ = np.lexsort((r, s, block[r]))                           # by block, then stage, then row
        rows[s[order_], r[order_]] = np.arange(len(s))
    else:
        raise ValueError(f"row order must be 'stage' or 'block', got {order!r}")
    return rows


def _resolve_max_live(max_live):
    if max_live != "device":
        return max_live
    try:
        return device_max_live("double")
    except RuntimeError:
        return None


class ComposedFunction:
    """A per-stage CasADi Function behind a whole-horizon interface (`cusadi2.compose`):
    callable on the host with the declared buffers (one instance) for the declared
    output's structural nonzeros; `write` emits the plan file."""

    def __init__(self, name: str, fn: ca.Function, composition: Composition):
        self.name, self.fn, self.composition = name, fn, composition
        out = composition.outputs[0]
        rows, cols = out.coo()
        self.coo = np.stack([rows, cols], axis=1).astype(np.int32).reshape(-1, 2)
        self.nnz_out = out.nnz
        self.shape = (out.rows, out.cols)

    @property
    def n_in(self) -> int:
        return len(self.composition.inputs)

    @property
    def nnz_in(self) -> List[int]:
        return self.composition.in_lengths

    def __call__(self, *bufs) -> np.ndarray:
        return evaluate_composition(self.fn, self.composition, bufs)[0]

    def dense(self, *bufs) -> np.ndarray:
        M = np.zeros(self.shape)
        M[self.coo[:, 0], self.coo[:, 1]] = self(*bufs)
        return M

    def write(self, path: str, max_live="device", block_size: Optional[int] = None):
        """The plan file; `max_live` as in `export_cuipm_staged` ("device" by default)."""
        plan = extract_level_plan(self.fn, max_live=_resolve_max_live(max_live))
        write_plan(self.fn, path, composition=self.composition, block_size=block_size, plan=plan)
        return plan


class StagedNLP:
    """min sum_k f(args_k)  s.t.  c(args_k) = 0,  d(args_k) >= 0 over k = 0..n_stages-1,
    where the stage arguments are described by `specs` as in `cusadi2.jax.staged`
    (name, size, mode stage|coupled|shared, offset, span, param, alias, const): the
    non-parameter arguments form the knot-major global x (`StageLayout`), the
    parameters form p (shared first, then per-stage), `const` arguments (a contact
    schedule, per-stage weights) are baked into the plans. `stage_f` returns a
    scalar; `stage_c` / `stage_d` return one vector per stage, of which `c_active` /
    `d_active` ((n_stages, ny) bool, see `rows_active`; default all) say which rows
    exist at which stage, numbered in `c_order` / `d_order` ("stage": stage by
    stage; "block": row of the stage output by row, as a hand-written
    whole-horizon function that lists the initial-state rows first would).
    `functions()` gives the nine cuipm functions as `ComposedFunction`s;
    `whole_horizon()` the same NLP as whole-horizon SX for IPOPT and checks."""

    def __init__(self, stage_f: Callable, stage_c: Optional[Callable], stage_d: Optional[Callable], n_stages: int,
                 specs, *, c_active=None, d_active=None, c_order="stage", d_order="stage", dtype=np.float64,
                 relax_equality: bool = False):
        # relax_equality (MadNLP's RelaxEquality): the equality rows become inequality
        # rows with d_L = d_U = 0 (the caller passes those bounds; cuipm_jax's bounds_scale
        # then relaxes them to +-1e-8), so nyc == 0 and the condensed KKT system applies.
        # Per stage the relaxed rows come first; `relaxed` (nyd,) marks them in the global
        # d numbering.
        self.layout = StageLayout(specs, n_stages)
        self.n_stages = int(n_stages)
        self.specs = self.layout.specs
        self.syms, arrs = [], []
        for sp in self.specs:
            sym, arr = sx_inputs(sp["name"], (sp["size"],))
            self.syms.append(sym)
            arrs.append(arr)
        avals = [jax.ShapeDtypeStruct((sp["size"],), dtype) for sp in self.specs]

        def trace(fun):
            if fun is None:
                return ca.SX.zeros(0, 1)
            outs = eval_jaxpr(jax.make_jaxpr(fun)(*avals), arrs)
            if len(outs) != 1:
                raise ValueError("StagedNLP: a stage function must return one array")
            return to_sx_vector(outs[0])
        self.f_sx, self.c_sx, self.d_sx = trace(stage_f), trace(stage_c), trace(stage_d)
        if int(self.f_sx.numel()) != 1:
            raise ValueError("StagedNLP: stage_f must return a scalar")
        self.x_stage = ca.vertcat(*[self.syms[i] for i in self.layout.var_args])
        self.relax_equality = bool(relax_equality)
        n_relaxed_stage = 0
        if relax_equality:
            c_act = self._active(c_active, int(self.c_sx.numel()), "c")
            d_act = self._active(d_active, int(self.d_sx.numel()), "d")
            n_relaxed_stage = int(self.c_sx.numel())
            self.d_sx = ca.vertcat(self.c_sx, self.d_sx)
            self.c_sx = ca.SX.zeros(0, 1)
            d_active, c_active = np.concatenate([c_act, d_act], axis=1), None
        self.c_active = self._active(c_active, int(self.c_sx.numel()), "c")
        self.d_active = self._active(d_active, int(self.d_sx.numel()), "d")
        self.c_rows, self.d_rows = _row_numbers(self.c_active, c_order), _row_numbers(self.d_active, d_order)
        self.nyc, self.nyd = int(self.c_active.sum()), int(self.d_active.sum())
        self.relaxed = np.zeros(self.nyd, bool)
        if n_relaxed_stage:
            rows = self.d_rows[:, :n_relaxed_stage]
            self.relaxed[rows[rows >= 0]] = True
        self.nx, self.np = self.layout.nx, self.layout.np
        self._gx = self.layout.stage_x_cols()                                 # (n_stages, nx_stage)
        self._base_gathers = [self.layout.gather(i) for i in range(len(self.specs))]

    def _active(self, active, ny, what):
        if active is None:
            return np.ones((self.n_stages, ny), bool)
        active = np.asarray(active, bool)
        if active.shape != (self.n_stages, ny):
            raise ValueError(f"StagedNLP: {what}_active must be ({self.n_stages}, {ny}), got {active.shape}")
        return active

    # ---- composition
    def _rc(self, M: ca.SX, rows_of: np.ndarray, cols_of: np.ndarray) -> np.ndarray:
        """Global (row, col) per (stage, nonzero of M), -1 where the row is inactive or the
        column belongs to a knot outside its buffer (see `StageLayout.global_cols`)."""
        coo = _coo2(M)
        r = rows_of[:, coo[:, 0]]
        c = cols_of[:, coo[:, 1]]
        keep = (r >= 0) & (c >= 0)
        return np.stack([np.where(keep, r, -1), np.where(keep, c, -1)], axis=-1)

    def _make(self, name, out_sx, declared, extra_syms, extra_gathers, rows_of, cols_of, shape) -> ComposedFunction:
        fn = ca.Function(name, list(self.syms) + extra_syms, [out_sx])
        comp = compose(fn, self.n_stages, declared, self._base_gathers + extra_gathers, [(name, *shape)],
                       [(0, self._rc(out_sx, rows_of, cols_of))], consts=self.layout.consts)
        return ComposedFunction(name, fn, comp)

    def functions(self) -> Dict[str, ComposedFunction]:
        """The nine cuipm functions, composed."""
        N, nx, gx = self.n_stages, self.nx, self._gx
        x_in, p_in, eps_in = DeclaredArg.dense("x", nx), DeclaredArg.dense("p", self.np), DeclaredArg.dense("eps", 1)
        one = np.zeros((N, 1), np.int64)                       # the one row / column of a scalar or vector output
        xs, f, c, d = self.x_stage, self.f_sx, self.c_sx, self.d_sx
        lam_c, lam_d = ca.SX.sym("lam_c", int(c.numel())), ca.SX.sym("lam_d", int(d.numel()))
        grad_f = ca.sparsify(ca.jacobian(f, xs).T)
        fns = {
            "f":      self._make("f", f, [x_in, p_in, eps_in], [], [], one, one, (1, 1)),
            "jac_f":  self._make("jac_f", grad_f, [x_in, p_in, eps_in], [], [], gx, one, (nx, 1)),
            "hess_f": self._make("hess_f", ca.sparsify(ca.hessian(f, xs)[0]), [x_in, p_in, eps_in], [], [], gx, gx, (nx, nx)),
            "c":      self._make("c", ca.densify(c), [x_in, p_in], [], [], self.c_rows, one, (self.nyc, 1)),
            "jac_c":  self._make("jac_c", ca.sparsify(ca.jacobian(c, xs)), [x_in, p_in], [], [], self.c_rows, gx, (self.nyc, nx)),
            "hess_c": self._make("hess_c", ca.sparsify(ca.hessian(ca.dot(lam_c, c), xs)[0]),
                                 [x_in, p_in, DeclaredArg.dense("lam_c", self.nyc)], [lam_c], [(2, self.c_rows)], gx, gx, (nx, nx)),
            "d":      self._make("d", ca.densify(d), [x_in, p_in], [], [], self.d_rows, one, (self.nyd, 1)),
            "jac_d":  self._make("jac_d", ca.sparsify(ca.jacobian(d, xs)), [x_in, p_in], [], [], self.d_rows, gx, (self.nyd, nx)),
            "hess_d": self._make("hess_d", ca.sparsify(ca.hessian(ca.dot(lam_d, d), xs)[0]),
                                 [x_in, p_in, DeclaredArg.dense("lam_d", self.nyd)], [lam_d], [(2, self.d_rows)], gx, gx, (nx, nx)),
        }
        return fns

    # ---- the same NLP as whole-horizon SX (symbolic calls of the stage functions)
    def whole_horizon(self):
        """(x, p, f, c, d) SX on the global layout, usable as `nlp=` of `build_functions`
        and `solve_ipopt`: the reference the composed plans are checked against."""
        X, P = ca.SX.sym("x", self.nx), ca.SX.sym("p", self.np)
        fns = [ca.Function(n, list(self.syms), [y]) for n, y in (("fs", self.f_sx), ("cs", self.c_sx), ("ds", self.d_sx))]

        K = ca.DM(self.layout.consts) if self.layout.consts.size else ca.DM.zeros(0, 1)

        def pick(buf, t):
            return ca.vertcat(*[buf[int(j)] if j >= 0 else ca.SX(0) for j in t]) if len(t) else ca.SX.zeros(0, 1)
        F = ca.SX(0)
        C, D = [None] * self.nyc, [None] * self.nyd
        for s in range(self.n_stages):
            args = [pick(X if src == 0 else P if src == 1 else K, table[s]) for src, table in self._base_gathers]
            F = F + fns[0](*args)
            cs, ds = fns[1](*args), fns[2](*args)
            for r in range(int(self.c_sx.numel())):
                if self.c_active[s, r]:
                    C[self.c_rows[s, r]] = cs[r]
            for r in range(int(self.d_sx.numel())):
                if self.d_active[s, r]:
                    D[self.d_rows[s, r]] = ds[r]
        return X, P, F, (ca.vertcat(*C) if C else ca.SX.zeros(0, 1)), (ca.vertcat(*D) if D else ca.SX.zeros(0, 1))


def export_cuipm_staged(nlp: StagedNLP, x0, p0, out_dir: str, *, x_L=None, x_U=None, d_L=None, d_U=None,
                        max_live="device", block_size: Optional[int] = None) -> dict:
    """`export_cuipm` for a `StagedNLP`: the nine composed plans to `out_dir` plus
    problem.json (dims, nnz, x0, p0, bounds, and the x / p layouts: per argument
    name, base, stage count, size). The stage graphs are windowed to `max_live`
    slots: "device" (default) queries this machine's per-block shared memory
    (`cusadi2.device_max_live`, falling back to no windowing without a CUDA
    runtime), an int sets the budget, None disables windowing."""
    max_live = _resolve_max_live(max_live)
    x0, p0 = np.asarray(x0, dtype=np.float64).reshape(-1), np.asarray(p0, dtype=np.float64).reshape(-1)
    if x0.size != nlp.nx or p0.size != nlp.np:
        raise ValueError(f"export_cuipm_staged: x0 / p0 have {x0.size} / {p0.size} entries, the layout {nlp.nx} / {nlp.np}")
    fns = nlp.functions()
    os.makedirs(out_dir, exist_ok=True)
    info = {"nx": nlp.nx, "np": nlp.np, "nyc": nlp.nyc, "nyd": nlp.nyd, "plans": {}, "x0": x0.tolist(), "p0": p0.tolist(),
            "staged": {"n_stages": nlp.n_stages, "max_live": max_live,
                       "x_layout": [dict(name=n, base=b, count=c, size=z) for n, b, c, z in nlp.layout.blocks("x")],
                       "p_layout": [dict(name=n, base=b, count=c, size=z) for n, b, c, z in nlp.layout.blocks("p")],
                       "const_args": [nlp.specs[i]["name"] for i in nlp.layout.const_args],
                       "nyc_stage": int(nlp.c_sx.numel()), "nyd_stage": int(nlp.d_sx.numel())}}
    for k, v in (("x_L", x_L), ("x_U", x_U), ("d_L", d_L), ("d_U", d_U)):
        if v is not None:
            info[k] = np.asarray(v, dtype=np.float64).reshape(-1).tolist()
    for name in NAMES:
        fn = fns[name]
        path = os.path.join(out_dir, f"{name}.bin")
        plan = fn.write(path, max_live=max_live, block_size=block_size)
        info["plans"][name] = {"path": path, "rows": fn.shape[0], "cols": fn.shape[1], "nnz": fn.nnz_out,
                               "n_in": fn.n_in, "nnz_in": list(fn.nnz_in),
                               "peak_live_slots": plan.peak_live_slots, "n_levels": plan.n_levels, "n_nodes": plan.n_nodes,
                               "scratch_per_instance": fn.composition.scratch_per_instance}
    with open(os.path.join(out_dir, "problem.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    info["functions"] = fns
    return info
