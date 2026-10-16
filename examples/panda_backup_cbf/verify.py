"""Cross-check the CasADi and the JAX/frax model sides.

    python verify.py [--N 4] [--n 32]

  * N = 0: CasADi vs frax + cbfpy (the verified one-step filter), all five outputs;
  * N > 0: CasADi vs the JAX rollout (both variants of the derivative), all five outputs,
    plus the rollout itself (phi_N) and the backup policy's stopping behaviour.
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "profiling"))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=4)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

common.configure("cpu", x64=True)
common.ensure_frax()
import jax, jax.numpy as jnp, numpy as np
from backup_casadi import build_backup_function
from backup_jax import BackupProblem

fn, robot, info = build_backup_function(args.N)
print("casadi:", info, flush=True)
rng = np.random.default_rng(args.seed)
qs = common.sample_q(robot, rng, args.n); vs = common.sample_v(rng, args.n, 7)
zs = np.concatenate([qs, vs], 1)


def compare(tag, prob):
    model = jax.jit(prob.model_part)
    zdes = np.asarray(prob.desired_from_q(common.sample_q(robot, rng, args.n)))
    worst = {}
    for i in range(args.n):
        outs_c = [np.asarray(x) for x in fn(zs[i], zdes[i])]
        outs_j = [np.asarray(x) for x in model(jnp.asarray(zs[i]), jnp.asarray(zdes[i]))]
        for k, a, b in zip(["u_nom", "P", "q", "G", "h"], outs_c, outs_j):
            a = a.reshape(b.shape)
            rel = np.abs(a - b).max() / max(1.0, np.abs(b).max())
            worst[k] = max(worst.get(k, 0.0), rel)
    print(f"{tag}: max relative error casadi vs jax over {args.n} states:", {k: f"{v:.2e}" for k, v in worst.items()}, flush=True)
    return worst


if args.N == 0:
    from panda_problem import PandaProblem
    w = compare("N=0 vs cbfpy", BackupProblem(0))
else:
    w = compare(f"N={args.N} jvp/scan", BackupProblem(args.N, loop="scan", deriv="jvp"))
    w2 = compare(f"N={args.N} jacfwd/unrolled", BackupProblem(args.N, loop="unrolled", deriv="jacfwd"))
    prob = BackupProblem(args.N)
    ro = jax.jit(jax.vmap(prob.rollout_states))(jnp.asarray(zs))
    v0 = np.linalg.norm(np.asarray(ro[:, 0, 7:]), axis=1); vN = np.linalg.norm(np.asarray(ro[:, -1, 7:]), axis=1)
    print(f"rollout N={args.N}, dt={prob.dt}: |v| start mean {v0.mean():.3f} rad/s, end mean {vN.mean():.3f}, max {vN.max():.3f}")
bad = {k: v for k, v in w.items() if v > (1e-6 if k == "q" else 1e-7)}
print("OK" if not bad else f"MISMATCH {bad}")
