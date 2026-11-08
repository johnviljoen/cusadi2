#!/usr/bin/env python
"""Quadcopter navigation NMPC through the whole pipeline: JAX stage functions -> cusadi2
composed plans -> cuipm solving a batch on the GPU, called from JAX (cuipm_jax), checked
against IPOPT.

The problem is `tests/problems/quadcopter_nav_staged` (quadcopter_nav of jaxipm / cuipm's
test fixture, N knots of 13 states and N-1 of 4 rotor speeds, Euler dynamics defects,
four obstacle constraints per interval, start state as the parameter p). One trace of the
stage functions gives the nine whole-horizon plans (`StagedNLP`, `export_cuipm_staged`);
`cuipm_jax.Solver` loads them and `solve(x0, p)` is one XLA custom call that solves every
problem of the batch, each with its own start x0 and parameters p.

No command-line arguments: `quadcopter_nav_cuipm.json` next to this file lists the
experiments. Each writes `runs/quadcopter_nav_cuipm/<name>/` with config.json (settings,
commit, dirty flag, versions), the plans + problem.json, and results.json (per problem:
cuipm status, iterations, objective, feasibility; IPOPT objective and iterations; wall
times). The batch varies the start position: p[b, 0] = start_x[b].

    .venv/bin/python examples/jax/quadcopter_nav_cuipm.py
(not `uv run`: uv sync would drop the editable cuipm_jax install from the venv)
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # cusadi2 / cuipm allocate outside XLA's pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CONFIG = os.path.join(HERE, "quadcopter_nav_cuipm.json")
RUNS = os.path.join(ROOT, "runs", "quadcopter_nav_cuipm")
sys.path.insert(0, ROOT)                                           # tests.problems (the fixture NLP)

REQUIRED = {"N": int, "Ts": float, "start_x": list, "ir_nsteps": int, "max_iter": int, "ipopt": dict, "run": bool}


def load_config(path):
    with open(path) as fh:
        cfg = json.load(fh)
    for name, exp in cfg.items():
        for k, t in REQUIRED.items():
            if k not in exp:
                raise KeyError(f"{name}: missing key {k!r}")
            if t is float and isinstance(exp[k], int):
                exp[k] = float(exp[k])
            if not isinstance(exp[k], t):
                raise TypeError(f"{name}: key {k!r} must be {t.__name__}")
        unknown = set(exp) - set(REQUIRED)
        if unknown:
            raise KeyError(f"{name}: unknown keys {sorted(unknown)}")
    return cfg


def git_info():
    def run(*args):
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def run_experiment(name, exp):
    import casadi
    import jax
    jax.config.update("jax_enable_x64", True)
    import cuipm_jax
    from cusadi2.jax.ipm import StagedNLP, build_functions, export_cuipm_staged, solve_ipopt
    from tests.problems.quadcopter_nav import quadcopter_nav
    from tests.problems.quadcopter_nav_staged import quadcopter_nav_staged

    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    start = datetime.now().isoformat(timespec="seconds")
    N, Ts = exp["N"], exp["Ts"]

    # 1. the NLP from its stage functions, and the nine composed plans for cuipm
    t = time.perf_counter()
    stage_f, stage_c, stage_d, n_stages, specs, c_active, d_active, x0, p0 = quadcopter_nav_staged(N=N, Ts=Ts)
    nlp = StagedNLP(stage_f, stage_c, stage_d, n_stages, specs, c_active=c_active, d_active=d_active, c_order="block")
    plans_dir = os.path.join(out, "plans")
    info = export_cuipm_staged(nlp, x0, p0, plans_dir, d_L=np.zeros(nlp.nyd), d_U=np.full(nlp.nyd, np.inf))
    t_export = time.perf_counter() - t
    print(f"[{name}] nx={nlp.nx} np={nlp.np} nyc={nlp.nyc} nyd={nlp.nyd}; trace + export {t_export:.1f} s", flush=True)
    fns = build_functions(None, None, None, nlp.nx, nlp.np, nlp=nlp.whole_horizon())   # CasADi reference of f, c, d

    # 2. the batch: one problem per start position, each with its own x0 and p
    starts = [float(s) for s in exp["start_x"]]
    B = len(starts)
    P = np.tile(p0, (B, 1)); P[:, 0] = starts
    X0 = np.tile(x0, (B, 1)); X0[:, 0] = starts

    # 3. cuipm from JAX: first call compiles, second measures
    solver = cuipm_jax.Solver(plans_dir, max_iter=exp["max_iter"], ir_nsteps=exp["ir_nsteps"])
    solve = jax.jit(solver.solve)
    t = time.perf_counter(); sol = solve(X0, P); jax.block_until_ready(sol.x); t_first = time.perf_counter() - t
    t = time.perf_counter(); sol = solve(X0, P); jax.block_until_ready(sol.x); t_solve = time.perf_counter() - t
    X, status, iters = np.asarray(sol.x), np.asarray(sol.status), np.asarray(sol.iterations)

    # 4. IPOPT on each problem from the same start, and the quality of every cuipm iterate
    f, c, d, z0, _, dims = quadcopter_nav(N=N, Ts=Ts)
    results = []
    for b in range(B):
        t = time.perf_counter()
        ref = solve_ipopt(lambda z, p: f(z), c, lambda z, p: d(z), X0[b], P[b], d_L=np.zeros(dims["nyd"]),
                          d_U=np.full(dims["nyd"], np.inf), options=exp["ipopt"])
        t_ipopt = time.perf_counter() - t
        r = {"start_x": starts[b], "cuipm": {"status": cuipm_jax.STATUS[int(status[b])], "iterations": int(iters[b]),
                                             "f": float(fns["f"](X[b], P[b], np.zeros(1))),
                                             "c_inf": float(np.abs(np.asarray(fns["c"](X[b], P[b]))).max()),
                                             "d_min": float(np.asarray(fns["d"](X[b], P[b])).min())},
             "ipopt": {"success": bool(ref["success"]), "iterations": ref.get("iterations"), "f": float(ref["f"]),
                       "wall_s": t_ipopt}}
        results.append(r)
        print(f"[{name}] start {starts[b]:.2f}: cuipm {r['cuipm']['status']} in {r['cuipm']['iterations']} it, "
              f"f {r['cuipm']['f']:.7f}, |c| {r['cuipm']['c_inf']:.1e}, min d {r['cuipm']['d_min']:.1e} | "
              f"IPOPT f {r['ipopt']['f']:.7f} in {r['ipopt']['iterations']} it ({t_ipopt:.2f} s)", flush=True)
    print(f"[{name}] cuipm batch {B}: {t_solve * 1e3:.0f} ms per call ({t_first:.1f} s with compilation), "
          f"{t_solve * 1e3 / B:.0f} ms per problem", flush=True)

    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump({"batch": B, "cuipm_batch_s": t_solve, "cuipm_first_call_s": t_first, "problems": results}, fh, indent=1)
    config = {"experiment": name, "settings": exp, "start": start, "git": git_info(),
              "versions": {"jax": jax.__version__, "casadi": casadi.__version__, "python": sys.version.split()[0]},
              "host": os.uname().nodename, "dims": {k: info[k] for k in ("nx", "np", "nyc", "nyd")},
              "export_s": t_export}
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)
    return results


def main():
    cfg = load_config(CONFIG)
    for name, exp in cfg.items():
        if exp["run"]:
            run_experiment(name, exp)


if __name__ == "__main__":
    main()
