"""CPU throughput: N single-instance processes, one per physical core.

    python profiling/cpu_throughput.py --problem panda --backend elastiqp --procs 8

Each worker (spawned, pinned to its own core, JAX_PLATFORMS=cpu with frax's
single-thread flags) builds the problem, JIT-compiles the piece at batch 1,
warms up, waits at a barrier, then runs calls back to back for `--seconds`
and reports its count. Throughput = total calls / wall time of the window.
Pieces timed: end_to_end (nominal + safety_filter) and solve (QP only).
Appends rows to results/cpu_throughput.csv.
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIELDS = ["problem", "backend", "dtype", "piece", "procs", "seconds", "total_calls",
          "calls_per_s", "per_proc_calls_per_s_mean", "per_proc_calls_per_s_min", "single_proc_us_equiv", "tag"]


def worker(rank, args, barrier, q):
    os.sched_setaffinity(0, {args.cores[rank]})
    import common

    common.configure("cpu", x64=(args.dtype == "f64"))
    common.ensure_frax()
    import jax
    import jax.numpy as jnp
    import numpy as np
    import qpax

    if args.backend == "elastiqp":
        import elastiqp.jax

    from sampling import build_dataset

    if args.problem == "panda":
        from panda_problem import PandaProblem as Prob
    else:
        from g1_problem import G1Problem as Prob
    prob = Prob(args.backend)
    dt = jnp.float64 if args.dtype == "f64" else jnp.float32
    ds = build_dataset(prob, args.n_samples, seed=args.seed + rank)
    z_all = jnp.asarray(ds["z"], dtype=dt)
    zd_all = jnp.asarray(ds["z_des"], dtype=dt)
    u_all = jax.jit(jax.vmap(prob.nominal))(z_all, zd_all)
    P, qv, _, _, G, h = jax.jit(jax.vmap(prob.qp_data))(z_all, u_all)
    penalty = jnp.asarray(prob.penalty)
    if args.backend == "qpax":
        def solve(P, q, G, h):
            return qpax.solve_qp_elastic_primal(P, q, G, h, penalty=penalty, solver_tol=prob.solver_tol)
    else:
        def solve(P, q, G, h):
            return elastiqp.jax.solve(P, q, G, h, penalty=penalty, eps_abs=prob.solver_tol).x

    pieces = {
        "end_to_end": (jax.jit(prob.end_to_end), [(z_all[i], zd_all[i]) for i in range(args.n_samples)]),
        "solve": (jax.jit(solve), [(P[i], qv[i], G[i], h[i]) for i in range(args.n_samples)]),
    }
    results = {}
    for name in args.pieces:
        fn, argl = pieces[name]
        for i in range(20):
            jax.block_until_ready(fn(*argl[i % len(argl)]))
        barrier.wait()
        t0 = time.perf_counter()
        n = 0
        k = len(argl)
        while True:
            jax.block_until_ready(fn(*argl[n % k]))
            n += 1
            if (n & 15) == 0 and time.perf_counter() - t0 >= args.seconds:
                break
        results[name] = (n, time.perf_counter() - t0)
    q.put((rank, results))


def physical_cores():
    out = os.popen("lscpu -p=CPU,Core,Socket | grep -v '^#'").read().strip().splitlines()
    seen, cpus = set(), []
    for line in out:
        cpu, core, sock = (int(x) for x in line.split(",")[:3])
        if (core, sock) not in seen:
            seen.add((core, sock))
            cpus.append(cpu)
    return cpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["panda", "g1"], required=True)
    ap.add_argument("--backend", choices=["elastiqp", "qpax"], required=True)
    ap.add_argument("--dtype", choices=["f64", "f32"], default="f64")
    ap.add_argument("--procs", type=int, default=None, help="default: number of physical cores")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pieces", nargs="+", default=["end_to_end", "solve"])
    ap.add_argument("--out", default=str(HERE / "results" / "cpu_throughput.csv"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cores = physical_cores()
    args.procs = args.procs or len(cores)
    assert args.procs <= len(cores), f"{args.procs} procs > {len(cores)} physical cores"
    args.cores = cores[: args.procs]

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.procs)
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, args, barrier, q)) for r in range(args.procs)]
    t0 = time.time()
    for p in procs:
        p.start()
    res = [q.get() for _ in range(args.procs)]
    for p in procs:
        p.join()
    rows = []
    for piece in args.pieces:
        counts = [r[1][piece][0] for r in res]
        durs = [r[1][piece][1] for r in res]
        per = [c / d for c, d in zip(counts, durs)]
        total = sum(counts) / max(durs)
        rows.append(dict(problem=args.problem, backend=args.backend, dtype=args.dtype, piece=piece, procs=args.procs,
                         seconds=args.seconds, total_calls=sum(counts), calls_per_s=total,
                         per_proc_calls_per_s_mean=sum(per) / len(per), per_proc_calls_per_s_min=min(per),
                         single_proc_us_equiv=1e6 / (sum(per) / len(per)), tag=args.tag))
        print(f"[{args.problem}/{args.backend}/{args.dtype}] {piece:11s} {args.procs} procs: {total:9.1f} calls/s "
              f"(per proc mean {sum(per)/len(per):8.1f}, min {min(per):8.1f}); setup+run {time.time()-t0:.0f}s", flush=True)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
