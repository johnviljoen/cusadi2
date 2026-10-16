"""Time the frax/JAX model side of the backup-CBF filter at every implementation variant.

    python bench_frax_variants.py --N 8 12 16 --batches 256 [--variants scan-jvp,...]

Pieces per (N, variant): model_part = nominal + cbfpy rows + predictive rows (one jit, vmapped,
device-resident inputs, block_until_ready). Appends to results/frax_variants.csv with compile time.
"""
import argparse, csv, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, nargs="+", default=[8, 12, 16])
ap.add_argument("--batches", type=int, nargs="+", default=[256])
ap.add_argument("--variants", default="scan-jvp,unrolled-jvp,scan-jacfwd,unrolled-jacfwd")
ap.add_argument("--dtype", default="f64", choices=["f64", "f32"])
ap.add_argument("--matmul-precision", default=None, help="e.g. highest: disables TF32 for f32 matmuls (JAX_DEFAULT_MATMUL_PRECISION)")
ap.add_argument("--n-samples", type=int, default=4096)
ap.add_argument("--out", default=str(HERE / "results" / "frax_variants.csv"))
args = ap.parse_args()

if args.matmul_precision:
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = args.matmul_precision
common.configure("gpu", x64=(args.dtype == "f64"))
DTYPE_TAG = args.dtype + ("-" + args.matmul_precision if args.matmul_precision else "")
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np
from backup_jax import BackupProblem
from sampling import build_dataset

FIELDS = ["N", "variant", "dtype", "batch", "piece", "median_us", "p99_us", "mean_us", "min_us", "max_us", "n_calls", "compile_s", "n_rows"]
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
ds = None
for N in args.N:
    for var in args.variants.split(","):
        loop, deriv = var.split("-")
        prob = BackupProblem(N, loop=loop, deriv=deriv)
        if ds is None:
            ds = build_dataset(prob.base, args.n_samples, 0)
        Z = jnp.asarray(ds["z"]); ZD = jnp.asarray(ds["z_des"])
        fn = jax.jit(jax.vmap(prob.model_part))
        for B in args.batches:
            n_win = max(1, min(args.n_samples // B, 8))
            argsets = [(Z[i * B:(i + 1) * B], ZD[i * B:(i + 1) * B]) for i in range(n_win)]
            try:
                cs = common.compile_time(fn, *argsets[0])
                t = common.time_calls(fn, argsets, common.n_calls_for(B))
            except Exception as e:  # OOM etc.
                print(f"N={N} {var} B={B}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            row = dict(N=N, variant=var, dtype=DTYPE_TAG, batch=B, piece="model_frax", compile_s=cs, n_rows=prob.n_rows, **t)
            print(f"N={N:3d} {var:16s} B={B:5d} model_frax median {t['median_us']:10.1f} us  p99 {t['p99_us']:10.1f}  compile {cs:6.1f} s", flush=True)
            new = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new: w.writeheader()
                w.writerow({k: row.get(k) for k in FIELDS})
