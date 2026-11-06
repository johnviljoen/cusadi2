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
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional

import casadi as ca
import jax
import numpy as np

from cusadi2 import extract_level_plan, write_plan
from cusadi2.jax.interpret import eval_jaxpr, sx_inputs, to_sx_vector

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

    def __init__(self, name: str, fn: ca.Function, dtype=np.float64, max_live: Optional[int] = None,
                 block_size: Optional[int] = None, serial: bool = False):
        from cusadi2.jax.function import _Plan
        rows, cols = fn.sparsity_out(0).get_triplet()
        self.coo = np.stack([np.asarray(rows, np.int32), np.asarray(cols, np.int32)], axis=1).reshape(-1, 2)
        self.name, self.nnz = name, int(fn.nnz_out(0))
        self._plan = _Plan(name, fn, [(self.nnz,)], [self.coo], dtype, block_size, serial, max_live)
        self.serial = self._plan.serial

    def __call__(self, *args):
        return self._plan(list(args))[0]

    def free(self):
        self._plan.free()


def gpu_functions(fns: Dict[str, ca.Function], dtype=np.float64, max_live: Optional[int] = None,
                  block_size: Optional[int] = None, serial: bool = False) -> Dict[str, GpuFunction]:
    """The Functions `export_cuipm` returned (`info["functions"]`) as cusadi2 plans on the GPU,
    without tracing or differentiating again; same signatures as the plans (f / jac_f / hess_f
    take the ignored `eps` as a third argument). Level plans are windowed to the device's
    shared memory (`max_live=None`) and fall back to the serial kernel when no window fits."""
    return {name: GpuFunction(name, fn, dtype, max_live, block_size, serial) for name, fn in fns.items()}
