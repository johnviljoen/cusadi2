"""Bare cusadi2 kernel time (CUDA events, no XLA) of the built backup-CBF plans.

    python time_plans.py --N 8 12 16 --max-live 0 [--batches 1 64 256 1024 4096]

Appends to results/plan_times.csv.
"""
import argparse, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling")); sys.path.insert(0, str(HERE.parent.parent))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, nargs="+", default=[8, 12, 16])
ap.add_argument("--max-live", type=int, nargs="+", default=[0])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256, 1024, 4096])
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--out", default=str(HERE / "results" / "plan_times.csv"))
args = ap.parse_args()

common.configure("gpu", x64=True)
import jax
from cusadi2_model import Cusadi2Backup

FIELDS = ["N", "max_live", "batch", "median_us", "min_us", "n_levels", "peak_live_slots", "shared_bytes", "block_size", "sx_nodes", "n_rows"]
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
for N in args.N:
    for ml in args.max_live:
        m = Cusadi2Backup(N, max_live=ml)
        st = m.plan_stats
        print(f"N={N} ml={ml}: {st['sx_nodes']} SX nodes, {st['n_levels']} levels, peak live {st['peak_live_slots']}, shared {m.shared_bytes} B, block {m.block_size}", flush=True)
        for B in args.batches:
            med, mn = m.time_kernel(B, args.n)
            print(f"   B={B:5d}  kernel median {med:9.1f} us  min {mn:9.1f} us", flush=True)
            new = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new: w.writeheader()
                w.writerow(dict(N=N, max_live=ml, batch=B, median_us=med, min_us=mn, n_levels=st["n_levels"], peak_live_slots=st["peak_live_slots"],
                                shared_bytes=m.shared_bytes, block_size=m.block_size, sx_nodes=st["sx_nodes"], n_rows=st["n_rows"]))
