"""Time the frax evaluator's step and T-step rollout under jit + vmap (the fastest frax can be, no PJRT boundary).

    python time_frax_step.py --dtype f32|f64 --batches 256 2048 4096 8192 --T 20 [--rounds 3] [--rollout]

f32 sets JAX_DEFAULT_MATMUL_PRECISION=highest (TF32 off, see panda_backup_cbf). Appends to results/frax_step_times.csv.
"""
import argparse, csv, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common
ap = argparse.ArgumentParser()
ap.add_argument("--dtype", default="f32"); ap.add_argument("--batches", type=int, nargs="+", default=[256, 2048, 4096, 8192])
ap.add_argument("--T", type=int, default=20); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--rollout", action="store_true")
ap.add_argument("--loops", nargs="+", default=["scan", "unrolled"]); ap.add_argument("--n-calls", type=int, default=50)
a = ap.parse_args()
if a.dtype == "f32":
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
common.configure("gpu", x64=(a.dtype == "f64"))
import jax, jax.numpy as jnp, numpy as np
from frax_model import FraxStep
from dataset import make_dataset
fs = FraxStep(a.T, rounds=a.rounds)
ds = make_dataset(fs.base, 4096, 0, a.T)
out = HERE / "results" / "frax_step_times.csv"; out.parent.mkdir(exist_ok=True)
new = not out.exists()
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["piece", "loop", "dtype", "rounds", "T", "batch", "median_us", "min_us", "p99_us", "compile_s", "nonfinite"])
    pieces = [("step", None, jax.jit(jax.vmap(fs.step)))]
    if a.rollout:
        pieces += [("rollout", loop, jax.jit(jax.vmap(lambda x, Y, rp, par, loop=loop: fs.rollout(x, Y, rp, par, loop)))) for loop in a.loops]
    for piece, loop, fn in pieces:
        for B in a.batches:
            reps = -(-B // 4096)
            args = [jnp.asarray(np.tile(ds[k], (reps, 1))[:B]) for k in ("x", "Y", "rp", "par")]
            t0 = time.perf_counter()
            try:
                jax.block_until_ready(fn(*args))
            except Exception as e:
                print(piece, loop, B, "FAILED", type(e).__name__, str(e)[:120], flush=True); continue
            cs = time.perf_counter() - t0
            ts = []
            for _ in range(a.n_calls):
                t0 = time.perf_counter(); r = fn(*args); jax.block_until_ready(r); ts.append(time.perf_counter() - t0)
            r0 = r[0] if isinstance(r, tuple) else r
            nonfinite = int((~np.isfinite(np.asarray(r0))).sum())
            ts = np.array(ts) * 1e6
            w.writerow([piece, loop or "", a.dtype, a.rounds, a.T, B, f"{np.median(ts):.1f}", f"{ts.min():.1f}", f"{np.percentile(ts, 99):.1f}", f"{cs:.1f}", nonfinite]); f.flush()
            print(f"{piece:8s} {loop or '':9s} {a.dtype} B={B:6d}  median {np.median(ts)/1e3:8.2f} ms  min {ts.min()/1e3:8.2f}  compile {cs:5.1f} s  nonfinite {nonfinite}", flush=True)
