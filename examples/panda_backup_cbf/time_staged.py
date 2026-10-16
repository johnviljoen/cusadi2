"""Bare head + N-step cusadi2 time (CUDA events) of the staged model.  python time_staged.py --N 16 20 24 --batches 1 64 256 1024 4096"""
import argparse, csv, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling")); sys.path.insert(0, str(HERE.parent.parent))
import common
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, nargs="+", default=[16, 20, 24]); ap.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256, 1024, 4096])
ap.add_argument("--max-live-head", type=int, default=0); ap.add_argument("--max-live-step", type=int, default=0)
ap.add_argument("--n", type=int, default=50); ap.add_argument("--out", default=str(HERE / "results" / "staged_times.csv"))
args = ap.parse_args()
common.configure("gpu", x64=True)
import jax
from cusadi2_staged import Cusadi2Staged
FIELDS = ["N", "max_live_head", "max_live_step", "batch", "total_us", "min_us", "head_us", "step_us", "head_levels", "step_levels", "head_shared", "step_shared", "head_block", "step_block"]
out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
m = None
for N in args.N:
    if m is not None:   # one handler per process: re-init with the new N
        m.lib.backup_init(str(m._head).encode(), str(m._step).encode(), N, max(args.batches))
        m.N = N
    else:
        m = Cusadi2Staged(N, max_batch=max(args.batches), max_live_head=args.max_live_head, max_live_step=args.max_live_step)
        from build_staged import plan_paths
        m._head, m._step = plan_paths(args.max_live_head, args.max_live_step)
        print(f"head: {m.stats['head']['sx_nodes']} nodes, {m.stats['head']['n_levels']} levels, {m.shared_bytes[0]} B shared, block {m.block_sizes[0]}; "
              f"step: {m.stats['step']['sx_nodes']} nodes, {m.stats['step']['n_levels']} levels, {m.shared_bytes[1]} B shared, block {m.block_sizes[1]}", flush=True)
    for B in args.batches:
        med, mn, hd = m.time_kernel(B, args.n)
        print(f"N={N:3d} B={B:5d}: head {hd:8.1f} us + {N} steps -> total {med:9.1f} us (min {mn:9.1f}), {(med - hd) / N:7.1f} us per step", flush=True)
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new: w.writeheader()
            w.writerow(dict(N=N, max_live_head=args.max_live_head, max_live_step=args.max_live_step, batch=B, total_us=med, min_us=mn, head_us=hd, step_us=(med - hd) / N,
                            head_levels=m.stats["head"]["n_levels"], step_levels=m.stats["step"]["n_levels"], head_shared=m.shared_bytes[0], step_shared=m.shared_bytes[1],
                            head_block=m.block_sizes[0], step_block=m.block_sizes[1]))
