"""Validate the CUDA interior-point solver against qpax on the real QP data.

    python validate_ipm.py --N 0 20 --n 1024 [--time]

For each N: QP data from the frax/JAX model on the first n dataset states, solved by
qpax.solve_qp_elastic (per instance, vmapped) and by the CUDA IPM; reports max |dx|, objective
difference, convergence fractions and the iteration distributions.  --time also times both
solvers at batch 256 (solve only, precomputed data).
"""
import argparse, os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, nargs="+", default=[0, 20])
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--time", action="store_true")
ap.add_argument("--no-screen", action="store_true", help="disable the exact row screening in the CUDA IPM")
args = ap.parse_args()

common.configure("gpu", x64=True)
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np, qpax
from backup_jax import BackupProblem
from sampling import build_dataset
from ipm import CudaIpm

for N in args.N:
    prob = BackupProblem(N)
    ds = build_dataset(prob.base, max(args.n, 256), 0)
    Z = jnp.asarray(ds["z"][: args.n]); ZD = jnp.asarray(ds["z_des"][: args.n])
    P, q, G, h = [jax.block_until_ready(x) for x in jax.jit(jax.vmap(prob.model_part))(Z, ZD)[1:]]
    pen = jnp.asarray(prob.penalty)
    ipm = CudaIpm(prob.n_rows, max_batch=4096, penalty=prob.penalty)
    ipm.set_screen(not args.no_screen)

    def qpax1(P, q, G, h):
        x, t, s1, s2, z1, z2, conv, it = qpax.solve_qp_elastic(P, q, G, h, pen, solver_tol=prob.solver_tol, max_iter=30)
        return x, t, conv, it
    xq, tq, cq, iq = [np.asarray(a) for a in jax.jit(jax.vmap(qpax1))(P, q, G, h)]
    xc, ic, cc = [np.asarray(a) for a in jax.jit(ipm.solve)(P, q, G, h)]
    Pn, qn, Gn, hn = [np.asarray(a) for a in (P, q, G, h)]
    penn = np.asarray(pen)

    def objective(x):
        t = np.maximum(Gn @ x[..., None] - hn[..., None], 0.0)[..., 0]     # optimal slack for given x
        return 0.5 * np.einsum("bi,bij,bj->b", x, Pn, x) + np.einsum("bi,bi->b", qn, x) + t @ penn
    fq, fc = objective(xq), objective(xc)
    dx = np.abs(xq - xc).max(1)
    both = (cq == 1) & (cc == 1)
    print(f"N={N} rows={prob.n_rows} n={args.n}: qpax converged {cq.mean():.4f} (iters mean {iq.mean():.1f}, p99 {np.percentile(iq, 99):.0f}, max {iq.max()}), "
          f"cuda converged {cc.mean():.4f} (iters mean {ic.mean():.1f}, p99 {np.percentile(ic, 99):.0f}, max {ic.max()})")
    if both.sum() == 0: both[:] = True
    print(f"   max|dx| over both-converged {dx[both].max():.2e} (median {np.median(dx[both]):.2e}), over all {dx.max():.2e}; "
          f"objective cuda - qpax: max {np.max(fc - fq):+.2e} min {np.min(fc - fq):+.2e}  (|f| ~ {np.abs(fq).mean():.1e})", flush=True)
    if args.time:
        B = 256
        argsets = [(P[i * B:(i + 1) * B], q[i * B:(i + 1) * B], G[i * B:(i + 1) * B], h[i * B:(i + 1) * B]) for i in range(max(1, min(args.n // B, 4)))]
        fq_ = jax.jit(jax.vmap(lambda P, q, G, h: qpax.solve_qp_elastic_primal(P, q, G, h, pen, solver_tol=prob.solver_tol)))
        fc_ = jax.jit(ipm.solve)
        for name, fn in (("qpax", fq_), ("cuda_ipm", fc_)):
            cs = common.compile_time(fn, *argsets[0]); t = common.time_calls(fn, argsets, 200)
            print(f"   solve B={B} {name:9s} median {t['median_us']:9.1f} us  p99 {t['p99_us']:9.1f}  compile {cs:.1f} s", flush=True)
