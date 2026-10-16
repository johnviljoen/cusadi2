"""Time one {problem} x {backend, platform, dtype} configuration over batch sizes.

    python profiling/bench.py --problem panda --backend qpax --platform gpu --dtype f64 \
        --batches 1 16 128 1024 4096

Each piece is JIT-compiled separately (jax.jit(jax.vmap(...)) on GPU; on CPU
batch 1 is run unbatched, like the frax demo). After warm-up, every call is
followed by jax.block_until_ready; the CSV has median / p99 over n calls
(1000 for B <= 128, 200 above). Compile time is measured with
`.lower().compile()` and reported separately.

Pieces:
  nominal     torque_control_matrices-path OSC controller: (z, z_des) -> tau_des
  qp_data     cbf.qp_data(z, u_des) -> (P, q, A, b, G, h)
  dyn_terms   (2a) M, M^-1, bias, task Jacobian(s)
  barriers    (2b) h, Lfh (jvp) and Lgh (vmapped jvp)   [recomputes f, g]
  h2_only     raw barrier values h_2(z), no derivatives  [extra]
  solve       the QP solver on precomputed (P, q, G, h), cbfpy settings
  end_to_end  nominal followed by cbf.safety_filter
Appends rows to results/raw_timings.csv and results/solver_stats.csv.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["panda", "g1"], required=True)
    ap.add_argument("--backend", choices=["elastiqp", "qpax"], required=True)
    ap.add_argument("--platform", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--dtype", choices=["f64", "f32"], default="f64")
    ap.add_argument("--batches", type=int, nargs="+", default=None)
    ap.add_argument("--n-samples", type=int, default=4096)
    ap.add_argument("--n-calls", type=int, default=None, help="override call count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(common.RESULTS / "raw_timings.csv"))
    ap.add_argument("--stats-out", default=str(common.RESULTS / "solver_stats.csv"))
    ap.add_argument("--pieces", nargs="+", default=None)
    ap.add_argument("--tag", default="")
    return ap.parse_args()


ARGS = parse()
if ARGS.backend == "elastiqp":
    assert ARGS.platform == "cpu" and ARGS.dtype == "f64", "elastiqp: CPU, fp64 only"
common.configure(ARGS.platform, x64=(ARGS.dtype == "f64"))
common.ensure_frax()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import qpax  # noqa: E402

if ARGS.backend == "elastiqp":
    import elastiqp.jax  # noqa: E402

from sampling import build_dataset  # noqa: E402

TIMING_FIELDS = [
    "problem", "backend", "platform", "dtype", "batch", "vmap", "piece",
    "median_us", "p99_us", "mean_us", "min_us", "max_us", "n_calls", "compile_s", "tag",
]
STATS_FIELDS = [
    "problem", "backend", "platform", "dtype", "batch", "n_instances",
    "iters_mean", "iters_median", "iters_p99", "iters_max", "iters_min",
    "batchmax_mean", "batchmax_p99", "batchmax_max", "n_batches",
    "converged_frac", "active_cbf_frac", "active_rows_mean", "filter_changed_frac",
    "active_frac_random", "active_frac_jointlimit", "active_frac_collision", "u_sat_frac",
    "n_vars", "n_rows", "num_cbf", "tag",
]


def append_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def make_problem():
    if ARGS.problem == "panda":
        from panda_problem import PandaProblem

        return PandaProblem(ARGS.backend)
    from g1_problem import G1Problem

    return G1Problem(ARGS.backend)


def make_solver(prob):
    penalty = jnp.asarray(prob.penalty)
    tol = prob.solver_tol
    if ARGS.backend == "qpax":

        def solve(P, q, G, h):
            return qpax.solve_qp_elastic_primal(P, q, G, h, penalty=penalty, solver_tol=tol)

        def solve_full(P, q, G, h):
            x, t, s1, s2, z1, z2, conv, iters = qpax.solve_qp_elastic(P, q, G, h, penalty, solver_tol=tol)
            return x, conv, iters

    else:

        def solve(P, q, G, h):
            return elastiqp.jax.solve(P, q, G, h, penalty=penalty, eps_abs=tol).x

        def solve_full(P, q, G, h):
            r = elastiqp.jax.solve(P, q, G, h, penalty=penalty, eps_abs=tol)
            return r.x, r.converged, r.iters

    return solve, solve_full


def main():
    t_start = time.time()
    prob = make_problem()
    dt = jnp.float64 if ARGS.dtype == "f64" else jnp.float32
    print(f"[{ARGS.problem}/{ARGS.backend}/{ARGS.platform}/{ARGS.dtype}] nv={prob.nv} rows={prob.n_rows} "
          f"num_cbf={prob.num_cbf} sc_pairs={prob.n_sc_pairs} backend={jax.default_backend()} x64={jax.config.jax_enable_x64}",
          flush=True)

    ds = build_dataset(prob, ARGS.n_samples, ARGS.seed)
    N = ARGS.n_samples
    z_all = jnp.asarray(ds["z"], dtype=dt)
    zdes_all = jnp.asarray(ds["z_des"], dtype=dt)

    solve, solve_full = make_solver(prob)

    # Whole-dataset pass: u_des, QP data, solution statistics (vmapped, chunked)
    chunk = 512 if ARGS.platform == "gpu" else 256
    nominal_v = jax.jit(jax.vmap(prob.nominal))
    qpdata_v = jax.jit(jax.vmap(prob.qp_data))
    solvefull_v = jax.jit(jax.vmap(solve_full))
    u_parts, P_parts, q_parts, G_parts, h_parts, x_parts, conv_parts, it_parts = ([] for _ in range(8))
    for s in range(0, N, chunk):
        zc, zdc = z_all[s : s + chunk], zdes_all[s : s + chunk]
        u = nominal_v(zc, zdc)
        P, q, _, _, G, h = qpdata_v(zc, u)
        x, conv, iters = solvefull_v(P, q, G, h)
        jax.block_until_ready((u, P, q, G, h, x, conv, iters))
        u_parts.append(u); P_parts.append(P); q_parts.append(q); G_parts.append(G); h_parts.append(h)
        x_parts.append(x); conv_parts.append(conv); it_parts.append(iters)
    u_all = jnp.concatenate(u_parts)
    P_all, q_all, G_all, h_all = (jnp.concatenate(p) for p in (P_parts, q_parts, G_parts, h_parts))
    x_all = np.asarray(jnp.concatenate(x_parts), dtype=np.float64)
    conv_all = np.asarray(jnp.concatenate(conv_parts)).astype(int)
    iters_all = np.asarray(jnp.concatenate(it_parts)).astype(int)
    print(f"  dataset: {N} states, v_rms={ds['v_rms']:.3f}; solver converged {conv_all.mean()*100:.1f}%, "
          f"iters mean {iters_all.mean():.2f} max {iters_all.max()}; dataset+stats pass {time.time()-t_start:.1f}s", flush=True)

    # Active-set statistics (CBF rows are the first num_cbf rows of G, h)
    Gn, hn, un = (np.asarray(a, dtype=np.float64) for a in (G_all, h_all, u_all))
    resid = hn - np.einsum("bij,bj->bi", Gn, x_all[:, : prob.m])
    cbf_resid = resid[:, : prob.num_cbf]
    tol_act = 1e-5 * np.maximum(1.0, np.abs(hn[:, : prob.num_cbf]))
    active = cbf_resid <= tol_act
    changed = np.max(np.abs(x_all[:, : prob.m] - un), axis=1) > 1e-6 * np.maximum(1.0, np.max(np.abs(un), axis=1))
    kind = ds["kind"]
    act_any = active.any(1)
    act_by_kind = [float(act_any[kind == k].mean()) if np.any(kind == k) else float("nan") for k in range(3)]
    u_sat = (np.abs(un) >= np.asarray(prob.cfg.u_max)[None, :] * (1 - 1e-9)).any(1)
    print(f"  active CBF row in {act_any.mean()*100:.1f}% of instances (mean {active.sum(1).mean():.2f} rows; "
          f"random {act_by_kind[0]*100:.0f}%, joint-limit {act_by_kind[1]*100:.0f}%, collision {act_by_kind[2]*100:.0f}%); "
          f"filter changed u in {changed.mean()*100:.1f}%; u_des beyond torque bound in {u_sat.mean()*100:.1f}%", flush=True)

    # Check the two qpax entry points agree (primal-only vs full output)
    if ARGS.backend == "qpax":
        xs = jax.jit(jax.vmap(solve))(P_all[:64], q_all[:64], G_all[:64], h_all[:64])
        print(f"  solve_qp_elastic_primal vs solve_qp_elastic max|dx| = {float(jnp.max(jnp.abs(xs - x_all[:64]))):.2e}")

    batches = ARGS.batches or ([1, 16, 128, 1024, 4096] if ARGS.platform == "gpu" else [1])
    pieces = ARGS.pieces or ["nominal", "qp_data", "dyn_terms", "barriers", "h2_only", "solve", "end_to_end"]
    rows, stat_rows = [], []
    for B in batches:
        assert B <= N
        n_calls = ARGS.n_calls or common.n_calls_for(B)
        use_vmap = not (ARGS.platform == "cpu" and B == 1)
        n_windows = min(N // B, n_calls)
        # per-batch argument tuples, device resident
        def win(a, k):
            s = a[k * B : (k + 1) * B]
            return s if use_vmap else s[0]
        z_b = [win(z_all, k) for k in range(n_windows)]
        zd_b = [win(zdes_all, k) for k in range(n_windows)]
        u_b = [win(u_all, k) for k in range(n_windows)]
        qp_b = [(win(P_all, k), win(q_all, k), win(G_all, k), win(h_all, k)) for k in range(n_windows)]
        jax.block_until_ready((z_b, zd_b, u_b, qp_b))

        def wrap(f):
            return jax.jit(jax.vmap(f)) if use_vmap else jax.jit(f)

        funcs = {
            "nominal": (wrap(prob.nominal), [(z, zd) for z, zd in zip(z_b, zd_b)]),
            "qp_data": (wrap(prob.qp_data), [(z, u) for z, u in zip(z_b, u_b)]),
            "dyn_terms": (wrap(prob.dyn_terms), [(z,) for z in z_b]),
            "barriers": (wrap(prob.barriers), [(z,) for z in z_b]),
            "h2_only": (wrap(prob.h2_only), [(z,) for z in z_b]),
            "solve": (wrap(solve), qp_b),
            "end_to_end": (wrap(prob.end_to_end), [(z, zd) for z, zd in zip(z_b, zd_b)]),
        }
        for piece in pieces:
            fn, argl = funcs[piece]
            c_s = common.compile_time(fn, *argl[0])
            res = common.time_calls(fn, argl, n_calls)
            print(f"  B={B:5d} {piece:11s} median {res['median_us']:10.1f} us  p99 {res['p99_us']:10.1f} us  "
                  f"compile {c_s:6.2f} s  (n={n_calls})", flush=True)
            rows.append(dict(problem=prob.name, backend=ARGS.backend, platform=ARGS.platform, dtype=ARGS.dtype,
                             batch=B, vmap=int(use_vmap), piece=piece, compile_s=c_s, tag=ARGS.tag, **res))

        # iteration statistics for this batch size: per-instance + max per batch window
        nb = N // B
        bmax = iters_all[: nb * B].reshape(nb, B).max(axis=1)
        stat_rows.append(dict(
            problem=prob.name, backend=ARGS.backend, platform=ARGS.platform, dtype=ARGS.dtype, batch=B,
            n_instances=N, iters_mean=iters_all.mean(), iters_median=float(np.median(iters_all)),
            iters_p99=float(np.percentile(iters_all, 99)), iters_max=int(iters_all.max()), iters_min=int(iters_all.min()),
            batchmax_mean=bmax.mean(), batchmax_p99=float(np.percentile(bmax, 99)), batchmax_max=int(bmax.max()), n_batches=nb,
            converged_frac=conv_all.mean(), active_cbf_frac=active.any(1).mean(), active_rows_mean=active.sum(1).mean(),
            filter_changed_frac=changed.mean(), active_frac_random=act_by_kind[0], active_frac_jointlimit=act_by_kind[1],
            active_frac_collision=act_by_kind[2], u_sat_frac=u_sat.mean(), n_vars=prob.m, n_rows=prob.n_rows, num_cbf=prob.num_cbf, tag=ARGS.tag,
        ))
        append_rows(ARGS.out, TIMING_FIELDS, rows); rows = []
        append_rows(ARGS.stats_out, STATS_FIELDS, stat_rows); stat_rows = []

    # problem description (dimensions) once per run
    common.save_json(common.RESULTS / f"problem_{prob.name}.json", {
        "problem": prob.name, "nv": prob.nv, "n_vars": prob.m, "n_rows": prob.n_rows, "num_cbf": prob.num_cbf,
        "num_rd1_cbf": prob.cfg.num_rd1_cbf, "num_rd2_cbf": prob.cfg.num_rd2_cbf,
        "barrier_groups": prob.barrier_groups, "n_self_collision_pairs": prob.n_sc_pairs,
        "n_collision_spheres": int(prob.robot.flat_collision_radii.shape[0]),
        "penalties": {"cbf": float(prob.penalty[0]), "control": float(prob.penalty[-1])},
        "solver_tol": prob.solver_tol, "state_dim": 2 * prob.nv, "z_des_dim": prob.z_des_dim,
        "elastic_qp_vars_incl_slack": prob.m + prob.n_rows,
    })
    print(f"  done in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
