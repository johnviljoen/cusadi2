"""jax.profiler trace of end-to-end G1 / qpax on the GPU at batch 1024 (fp64).

    python profiling/trace_g1.py [--batch 1024] [--calls 5]

Attribution (per call, device stream events only):
  * the qpax while loop is bracketed by its condition evaluations, which show
    up as device events with hlo_op == "while*" (one per iteration + 1). All
    device events between the first and the last of those are "while loop"
    (body + condition); events before are nominal + qp_data + solver
    initialisation; events after are solver output extraction.
  * kernels per tick = device events per call, split into compute kernels and
    memcpys; the number of while.* events - 1 is the iteration count run.
Writes results/trace_g1_summary.json (and keeps the raw trace under
results/trace/, git-ignored).
"""

import argparse
import collections
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--calls", type=int, default=5)
ap.add_argument("--problem", default="g1")
ap.add_argument("--dtype", choices=["f64", "f32"], default="f64")
ARGS = ap.parse_args()

common.configure("gpu", x64=(ARGS.dtype == "f64"))
common.ensure_frax()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import qpax  # noqa: E402

from sampling import build_dataset  # noqa: E402


def main():
    if ARGS.problem == "g1":
        from g1_problem import G1Problem as Prob
    else:
        from panda_problem import PandaProblem as Prob
    prob = Prob("qpax")
    B = ARGS.batch
    ds = build_dataset(prob, B, 0)
    dt = jnp.float64 if ARGS.dtype == "f64" else jnp.float32
    z = jnp.asarray(ds["z"], dtype=dt)
    zd = jnp.asarray(ds["z_des"], dtype=dt)
    f = jax.jit(jax.vmap(prob.end_to_end))
    for _ in range(3):
        jax.block_until_ready(f(z, zd))

    # iteration count of this batch (batch max under vmap)
    penalty = jnp.asarray(prob.penalty)
    u = jax.jit(jax.vmap(prob.nominal))(z, zd)
    P, q, _, _, G, h = jax.jit(jax.vmap(prob.qp_data))(z, u)
    out = jax.jit(jax.vmap(lambda P, q, G, h: qpax.solve_qp_elastic(P, q, G, h, penalty, solver_tol=prob.solver_tol)))(P, q, G, h)
    iters = np.asarray(out[-1])
    conv = np.asarray(out[-2])

    tdir = common.RESULTS / "trace" / f"{prob.name}_qpax_gpu_{ARGS.dtype}_B{B}"
    tdir.mkdir(parents=True, exist_ok=True)
    wall = []
    with jax.profiler.trace(str(tdir), create_perfetto_trace=True):
        for _ in range(ARGS.calls):
            t0 = time.perf_counter()
            jax.block_until_ready(f(z, zd))
            wall.append(time.perf_counter() - t0)
        time.sleep(0.2)

    tf = sorted(tdir.glob("plugins/profile/*/*.trace.json.gz"))[-1]
    summary = analyse(tf, ARGS.calls)
    summary.update({
        "problem": prob.name, "batch": B, "dtype": ARGS.dtype, "n_calls_traced": ARGS.calls,
        "wall_ms_per_call": [w * 1e3 for w in wall],
        "qpax_iters_batch_max": int(iters.max()), "qpax_iters_mean": float(iters.mean()),
        "qpax_converged_frac": float(conv.mean()), "trace_file": str(tf),
    })
    common.save_json(common.RESULTS / f"trace_{prob.name}_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_hlo_op",)}, indent=1, default=str))


def analyse(trace_file: Path, n_calls: int) -> dict:
    tr = json.load(gzip.open(trace_file))
    ev = tr["traceEvents"]
    procs = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "process_name"}
    threads = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev if e.get("ph") == "M" and e["name"] == "thread_name"}
    X = [e for e in ev if e.get("ph") == "X"]
    gpu_pids = {p for p, n in procs.items() if "GPU" in n}
    dev = sorted((e for e in X if e["pid"] in gpu_pids), key=lambda e: e["ts"])
    host = [e for e in X if e["pid"] not in gpu_pids]

    # per-call windows: the top-level host events of the jitted function
    calls = sorted((e for e in host if e["name"].startswith("jit_") and threads.get((e["pid"], e["tid"]), "").startswith("python")),
                   key=lambda e: e["ts"])
    # keep the outermost (longest) event per cluster of nested ones
    windows = []
    for e in calls:
        if windows and e["ts"] < windows[-1][1]:
            continue
        windows.append((e["ts"], e["ts"] + e["dur"], e["name"]))
    if len(windows) != n_calls:
        print(f"WARNING: found {len(windows)} host call windows, expected {n_calls}", file=sys.stderr)

    def is_memcpy(e):
        return e["name"].startswith("Memcpy") or e["name"].startswith("memcpy") or "memcpy_details" in e.get("args", {})

    per_call = []
    op_stats = collections.defaultdict(lambda: {"count": 0, "dur_us": 0.0, "in_loop": 0})
    for (t0, t1, _) in windows:
        evs = [e for e in dev if t0 <= e["ts"] < t1]
        whiles = [e for e in evs if e["args"].get("hlo_op", "").startswith("while")]
        if whiles:
            ws, we = whiles[0]["ts"], whiles[-1]["ts"] + whiles[-1]["dur"]
        else:
            ws = we = float("inf")
        pre = [e for e in evs if e["ts"] < ws]
        loop = [e for e in evs if ws <= e["ts"] < we]
        post = [e for e in evs if e["ts"] >= we]
        busy = lambda L: sum(e["dur"] for e in L) / 1e3  # ms
        span = (evs[-1]["ts"] + evs[-1]["dur"] - evs[0]["ts"]) / 1e3 if evs else 0.0
        rec = {
            "n_device_events": len(evs),
            "n_kernels": sum(1 for e in evs if not is_memcpy(e)),
            "n_memcpy": sum(1 for e in evs if is_memcpy(e)),
            "n_memcpy_d2h": sum(1 for e in evs if e["name"].startswith("MemcpyD2H")),
            "n_while_cond_events": len(whiles),
            "device_busy_ms": busy(evs), "device_span_ms": span,
            "busy_pre_loop_ms": busy(pre), "busy_loop_ms": busy(loop), "busy_post_loop_ms": busy(post),
            "span_loop_ms": (we - ws) / 1e3 if whiles else 0.0,
            "kernels_pre_loop": len(pre), "kernels_loop": len(loop), "kernels_post_loop": len(post),
        }
        rec["loop_share_of_busy"] = rec["busy_loop_ms"] / rec["device_busy_ms"] if rec["device_busy_ms"] else None
        rec["loop_share_of_span"] = rec["span_loop_ms"] / rec["device_span_ms"] if rec["device_span_ms"] else None
        per_call.append(rec)
        for e in evs:
            k = e["args"].get("hlo_op", e["name"])
            op_stats[k]["count"] += 1
            op_stats[k]["dur_us"] += e["dur"]
            op_stats[k]["in_loop"] += int(ws <= e["ts"] < we)

    def mean(key):
        return float(np.mean([r[key] for r in per_call])) if per_call else None

    agg = {k: mean(k) for k in per_call[0]} if per_call else {}
    top = sorted(op_stats.items(), key=lambda kv: -kv[1]["dur_us"])[:25]
    return {
        "per_call": per_call,
        "mean_per_call": agg,
        "device_total_busy_ms": sum(e["dur"] for e in dev) / 1e3,
        "n_device_events_total": len(dev),
        "per_hlo_op": {k: v for k, v in op_stats.items()},
        "top_hlo_ops_by_time": [
            {"hlo_op": k, "count_per_call": v["count"] / max(1, len(per_call)), "us_per_call": v["dur_us"] / max(1, len(per_call)),
             "frac_in_loop": v["in_loop"] / v["count"]} for k, v in top],
        "device_name": [procs[p] for p in gpu_pids],
    }


if __name__ == "__main__":
    main()
