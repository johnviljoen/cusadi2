"""frax vs cusadi2 as the model-evaluation backend of the Panda OSC + CBF safety filter
(GPU, qpax, fp64), batch sizes 1, 4, 16, ..., 4096.

    python profiling/bench_cusadi2_swap.py [--batches 1,4,16,64,256,1024,4096] [--n-samples 4096]

Pieces timed per batch (jit, vmapped, device-resident inputs, block_until_ready,
median/p99 over 1000 calls (B <= 128) or 200 calls):
  model_frax     nominal OSC + cbf.qp_data with frax (jax)          -> u_nom, P, q, G, h
  model_cusadi2  the same quantities from the cusadi2 kernel (one XLA FFI custom call)
  solve          qpax.solve_qp_elastic_primal on precomputed (P, q, G, h) (shared by both)
  e2e_frax       nominal + cbf.safety_filter, frax model, as profiled before
  e2e_cusadi2    cusadi2 kernel -> qpax solve, in one jit
Also records compile times and the max |u_safe_frax - u_safe_cusadi2| per batch.
Appends to results/cusadi2_swap.csv; `--mode serial` adds cusadi2's serial-mode pieces.
report_cusadi2_swap.py renders results/cusadi2_swap.md and .png from the CSV.
"""
import argparse, csv, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common

p = argparse.ArgumentParser()
p.add_argument("--batches", default="1,4,16,64,256,1024,4096")
p.add_argument("--n-samples", type=int, default=4096)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=str(common.RESULTS / "cusadi2_swap.csv"))
p.add_argument("--mode", default="level", choices=["level", "serial"], help="cusadi2 execution mode")
p.add_argument("--max-live", type=int, default=0, help="cusadi2 live-slot budget; 0 = device max")
p.add_argument("--pieces", default=None, help="comma list; default all (level) or the two cusadi2 pieces (serial)")
args = p.parse_args()
batches = [int(b) for b in args.batches.split(",")]

common.configure("gpu", x64=True)
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np, qpax
from panda_problem import PandaProblem
from sampling import build_dataset
from cusadi2_oscbf import Cusadi2Oscbf

prob = PandaProblem("qpax")
cus = Cusadi2Oscbf(mode=args.mode, max_live=args.max_live)
print(f"cusadi2 plan: {cus.plan_stats['n_levels']} levels, peak live {cus.plan_stats['peak_live_slots']} slots, "
      f"block {cus.block_size}, shared {cus.shared_bytes} B, {cus.plan_stats['sx_nodes']} SX nodes", flush=True)
penalty = jnp.asarray(prob.penalty); tol = prob.solver_tol

def solve1(P, q, G, h):
    return qpax.solve_qp_elastic_primal(P, q, G, h, penalty=penalty, solver_tol=tol)[: prob.m]
def frax_model1(z, zd):
    u = prob.nominal(z, zd); P, q, _, _, G, h = prob.qp_data(z, u); return u, P, q, G, h
def cusadi2_e2e(z, zd):
    u, P, q, G, h = cus.model_part(z, zd); return jax.vmap(solve1)(P, q, G, h)

sfx = ("" if args.mode == "level" else "_serial") + (f"_ml{args.max_live}" if args.max_live else "")
pieces = {
    "model_frax": jax.jit(jax.vmap(frax_model1)),
    "model_cusadi2" + sfx: jax.jit(cus.model_part),
    "solve": jax.jit(jax.vmap(solve1)),
    "e2e_frax": jax.jit(jax.vmap(prob.end_to_end)),
    "e2e_cusadi2" + sfx: jax.jit(cusadi2_e2e),
}
sel = args.pieces.split(",") if args.pieces else (list(pieces) if (args.mode == "level" and not args.max_live) else ["model_cusadi2" + sfx, "e2e_cusadi2" + sfx])
timed = {k: pieces[k] for k in sel}

ds = build_dataset(prob, args.n_samples, args.seed)
Z = jnp.asarray(ds["z"]); ZD = jnp.asarray(ds["z_des"])
FIELDS = ["problem", "backend_model", "platform", "dtype", "batch", "piece", "median_us", "p99_us", "mean_us", "min_us", "max_us",
          "n_calls", "compile_s", "max_abs_du_vs_frax", "tag"]
rows = []
for B in batches:
    n_win = max(1, min(args.n_samples // B, 8))
    zs = [Z[i * B:(i + 1) * B] for i in range(n_win)]; zds = [ZD[i * B:(i + 1) * B] for i in range(n_win)]
    # precomputed QP data for the solve-only piece (frax path; identical to cusadi2's to ~1e-12)
    qp = [jax.block_until_ready(pieces["model_frax"](z, zd)) for z, zd in zip(zs, zds)]
    argsets = {k: ([(P, q, G, h) for (_, P, q, G, h) in qp] if k == "solve" else list(zip(zs, zds))) for k in pieces}
    du = float(jnp.max(jnp.abs(pieces["e2e_frax"](zs[0], zds[0]) - pieces["e2e_cusadi2" + sfx](zs[0], zds[0]))))
    for name, fn in timed.items():
        a0 = argsets[name][0]
        cs = common.compile_time(fn, *a0)
        t = common.time_calls(fn, argsets[name], common.n_calls_for(B))
        rows.append(dict(problem="panda", backend_model=("cusadi2" if "cusadi2" in name else "frax"), platform="gpu", dtype="f64",
                         batch=B, piece=name, compile_s=cs, max_abs_du_vs_frax=du, tag=f"cusadi2_swap_{args.mode}_ml{args.max_live}_jtimes_cse", **t))
        print(f"B={B:5d} {name:14s} median {t['median_us']:10.1f} us  p99 {t['p99_us']:10.1f}  compile {cs:5.1f} s  |du| {du:.1e}", flush=True)

out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
new = not out.exists()
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new: w.writeheader()
    for r in rows: w.writerow({k: r.get(k) for k in FIELDS})
print(f"appended {len(rows)} rows to {out}; run report_cusadi2_swap.py for the table and plot")
