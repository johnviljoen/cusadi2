"""How good is the frax model side in float32 for this filter?  (footnote to the fp32 timing)

    python fp32_check.py --N 20 --n 256          # runs itself twice: an fp32 process dumps the QP data, an fp64 process compares

fp32 process: frax model (unrolled-jvp variant) in float32 on the first n dataset states -> results/fp32_qp.npz.
fp64 process: the same states in float64; relative errors of P, q, G, h; both solved by the CUDA IPM (fp64);
|Δu| between the two torques; and the fp32-derived torque checked against the fp64 constraints
(max violation of G64 u32 <= h64 over the CBF rows, i.e. how far the fp32 filter would let the barrier
condition slip, in units of the row scale).
"""
import argparse, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=20); ap.add_argument("--n", type=int, default=256)
ap.add_argument("--stage", default="driver", choices=["driver", "f32", "f64"])
args = ap.parse_args()
NPZ = HERE / "results" / "fp32_qp.npz"

if args.stage == "driver":
    for st in ("f32", "f64"):
        subprocess.run([sys.executable, __file__, "--N", str(args.N), "--n", str(args.n), "--stage", st], check=True)
    sys.exit(0)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import common
common.configure("gpu", x64=(args.stage == "f64"))
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np
from backup_jax import BackupProblem
from sampling import build_dataset

prob = BackupProblem(args.N, loop="unrolled", deriv="jvp")
ds = build_dataset(prob.base, max(args.n, 256), 0)
Z = jnp.asarray(ds["z"][: args.n]); ZD = jnp.asarray(ds["z_des"][: args.n])
u, P, q, G, h = [np.asarray(x) for x in jax.jit(jax.vmap(prob.model_part))(Z, ZD)]
if args.stage == "f32":
    print("fp32 model dtype:", G.dtype)
    np.savez(NPZ, u=u, P=P, q=q, G=G, h=h)
    sys.exit(0)

from ipm import CudaIpm
f = np.load(NPZ)
rel = lambda a, b: float(np.abs(a.astype(np.float64) - b).max() / max(1.0, np.abs(b).max()))
print(f"N={args.N}, n={args.n}: fp32 vs fp64 model, max relative error:", {k: f"{rel(f[k], v):.1e}" for k, v in zip(["u", "P", "q", "G", "h"], [u, P, q, G, h])})
ncbf = 64 * (args.N + 1)
Gc, hc = G[:, :ncbf], h[:, :ncbf]
print("   G rows (CBF part only): max rel err", f"{rel(f['G'][:, :ncbf], Gc):.1e}", " h:", f"{rel(f['h'][:, :ncbf], hc):.1e}")
ipm = CudaIpm(prob.n_rows, max_batch=args.n, penalty=prob.penalty)
solve = jax.jit(ipm.solve)
x64, it64, c64 = [np.asarray(a) for a in solve(jnp.asarray(P), jnp.asarray(q), jnp.asarray(G), jnp.asarray(h))]
x32, it32, c32 = [np.asarray(a) for a in solve(*[jnp.asarray(f[k].astype(np.float64)) for k in ("P", "q", "G", "h")])]
print(f"   CUDA IPM (fp64) on fp64 data: converged {c64.mean():.3f}; on fp32 data: converged {c32.mean():.3f}")
du = np.abs(x64 - x32).max(1)
print(f"   |u64 - u32|: max {du.max():.3e} N m, median {np.median(du):.3e}, 99 % {np.percentile(du, 99):.3e}  (|u| ~ {np.abs(x64).mean():.1f})")
viol64 = np.maximum(np.einsum("bij,bj->bi", Gc, x64) - hc, 0.0).max(1)
viol32 = np.maximum(np.einsum("bij,bj->bi", Gc, x32) - hc, 0.0).max(1)
scale = np.abs(hc).mean()
print(f"   CBF-row violation G64 u - h64 (fp64 solution: max {viol64.max():.2e}; fp32-model solution: max {viol32.max():.2e}, "
      f"median {np.median(viol32):.2e}; instances > 1e-3: {(viol32 > 1e-3).mean():.3f}; row scale |h| ~ {scale:.1f})")
