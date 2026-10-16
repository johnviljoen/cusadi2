"""Three pipelines of the Panda backup-CBF safety filter, batch 1 ... 4096 on the GPU (fp64).

    python bench.py --N 20 [--batches 1,4,16,64,256,1024,4096] [--frax-variant unrolled-jvp]

Pieces (jit, device-resident inputs, block_until_ready, median / p99 over 1000 calls for B <= 128
else 200), all on the same 4096-state dataset:
  model_frax        nominal OSC + cbfpy rows + JAX rollout rows, frax model (fastest variant)
  model_cusadi2     the same quantities from the staged cusadi2 kernels (one FFI call)
  solve_qpax        qpax.solve_qp_elastic_primal (vmapped while_loop) on precomputed QP data
  solve_cuda        the CUDA interior-point solver on the same data
  e2e_frax_qpax     frax model -> qpax                 (the published stack)
  e2e_frax_cuda     frax model -> CUDA IPM             (the fairness row)
  e2e_cusadi2_cuda  cusadi2 model -> CUDA IPM          (ours)
Also records compile time per piece and the max |u_safe| difference of every end-to-end pipeline
against e2e_frax_qpax on the first window.  Appends to results/bench.csv.
"""
import argparse, csv, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=20)
ap.add_argument("--batches", default="1,4,16,64,256,1024,4096")
ap.add_argument("--frax-variant", default="unrolled-jvp")
ap.add_argument("--n-samples", type=int, default=4096)
ap.add_argument("--pieces", default=None)
ap.add_argument("--out", default=str(HERE / "results" / "bench.csv"))
ap.add_argument("--tag", default="")
args = ap.parse_args()
batches = [int(b) for b in args.batches.split(",")]

common.configure("gpu", x64=True)
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np, qpax
from backup_jax import BackupProblem
from cusadi2_staged import Cusadi2Staged
from ipm import CudaIpm
from sampling import build_dataset

loop, deriv = args.frax_variant.split("-")
prob = BackupProblem(args.N, loop=loop, deriv=deriv)
cus = Cusadi2Staged(args.N, max_batch=max(batches))
ipm = CudaIpm(prob.n_rows, max_batch=max(batches), penalty=prob.penalty)
pen = jnp.asarray(prob.penalty); tol = prob.solver_tol
print(f"N={args.N}: {prob.n_rows} rows; cusadi2 head {cus.stats['head']['n_levels']} levels / step {cus.stats['step']['n_levels']} levels, "
      f"blocks {cus.block_sizes}, shared {cus.shared_bytes} B; frax variant {args.frax_variant}", flush=True)


def qpax1(P, q, G, h):
    return qpax.solve_qp_elastic_primal(P, q, G, h, pen, solver_tol=tol)[: prob.m]
def frax_model(z, zd):
    return jax.vmap(prob.model_part)(z, zd)
def e2e_frax_qpax(z, zd):
    u, P, q, G, h = frax_model(z, zd); return jax.vmap(qpax1)(P, q, G, h)
def e2e_frax_cuda(z, zd):
    u, P, q, G, h = frax_model(z, zd); return ipm.solve(P, q, G, h)[0]
def e2e_cusadi2_cuda(z, zd):
    u, P, q, G, h = cus.model_part(z, zd); return ipm.solve(P, q, G, h)[0]

pieces = {
    "model_frax": jax.jit(frax_model),
    "model_cusadi2": jax.jit(cus.model_part),
    "solve_qpax": jax.jit(jax.vmap(qpax1)),
    "solve_cuda": jax.jit(lambda P, q, G, h: ipm.solve(P, q, G, h)[0]),
    "e2e_frax_qpax": jax.jit(e2e_frax_qpax),
    "e2e_frax_cuda": jax.jit(e2e_frax_cuda),
    "e2e_cusadi2_cuda": jax.jit(e2e_cusadi2_cuda),
}
sel = args.pieces.split(",") if args.pieces else list(pieces)

ds = build_dataset(prob.base, args.n_samples, 0)
Z = jnp.asarray(ds["z"]); ZD = jnp.asarray(ds["z_des"])
FIELDS = ["N", "n_rows", "batch", "piece", "median_us", "p99_us", "mean_us", "min_us", "max_us", "n_calls", "compile_s", "max_abs_du_vs_frax_qpax", "frax_variant", "tag"]
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
for B in batches:
    n_win = max(1, min(args.n_samples // B, 8))
    zs = [Z[i * B:(i + 1) * B] for i in range(n_win)]; zds = [ZD[i * B:(i + 1) * B] for i in range(n_win)]
    try:
        qp = [jax.block_until_ready(pieces["model_cusadi2"](z, zd)) for z, zd in zip(zs, zds)]
    except Exception as e:
        print(f"B={B}: model_cusadi2 failed: {type(e).__name__}: {str(e)[:200]}", flush=True); continue
    argsets = {k: ([(P, q, G, h) for (_, P, q, G, h) in qp] if k.startswith("solve") else list(zip(zs, zds))) for k in pieces}
    ref = None
    try:
        ref = np.asarray(jax.block_until_ready(pieces["e2e_frax_qpax"](zs[0], zds[0])))
    except Exception as e:
        print(f"B={B}: e2e_frax_qpax failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    for name in sel:
        fn = pieces[name]
        try:
            cs = common.compile_time(fn, *argsets[name][0])
            t = common.time_calls(fn, argsets[name], common.n_calls_for(B))
            du = float(np.abs(np.asarray(fn(zs[0], zds[0])) - ref).max()) if (name.startswith("e2e") and ref is not None) else None
        except Exception as e:
            print(f"B={B} {name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); continue
        row = dict(N=args.N, n_rows=prob.n_rows, batch=B, piece=name, compile_s=cs, max_abs_du_vs_frax_qpax=du, frax_variant=args.frax_variant, tag=args.tag, **t)
        print(f"B={B:5d} {name:17s} median {t['median_us']:10.1f} us  p99 {t['p99_us']:10.1f}  compile {cs:6.1f} s" + (f"  |du| {du:.1e}" if du is not None else ""), flush=True)
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new: w.writeheader()
            w.writerow({k: row.get(k) for k in FIELDS})
