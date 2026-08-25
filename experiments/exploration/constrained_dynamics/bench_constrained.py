"""
Benchmark the level-parallel constrained ABA-Jacobian kernel.

Uses cusadi.pytorch.lp_unified.CusadiFunction2 which auto-routes small DAGs to
the LP kernel and large DAGs to the tiled LP kernel. Reports the mode selected.

Timing: CUDA events inside the kernel wrapper, 10 warmup + 100 timed, median.

Usage:
    CUDA_VISIBLE_DEVICES=5 python exploration/constrained_dynamics/bench_constrained.py \
        --robot a1 --out results/a1_constrained_lp.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import casadi
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cusadi import CUSADI_FUNCTION_DIR  # noqa: E402
from cusadi.pytorch.lp_unified import CusadiFunction2  # noqa: E402


N_WARMUP = 10
N_EVALS = 100
BATCH_SIZES = [1, 8, 32, 128, 256]


def _random_inputs(fn: casadi.Function, bs: int, seed: int = 0) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    nq, nv = fn.nnz_in(0), fn.nnz_in(1)
    q = np.zeros((bs, nq))
    for k in range(bs):
        qk = rng.normal(0, 0.3, size=nq)
        qw, qx, qy, qz = rng.normal(size=4)
        nrm = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
        qk[3:7] = [qx / nrm, qy / nrm, qz / nrm, qw / nrm]
        q[k] = qk
    v = rng.normal(0, 0.2, size=(bs, nv))
    t = rng.normal(0, 1.0, size=(bs, nv))
    return [
        torch.tensor(q, dtype=torch.double, device="cuda").contiguous(),
        torch.tensor(v, dtype=torch.double, device="cuda").contiguous(),
        torch.tensor(t, dtype=torch.double, device="cuda").contiguous(),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["a1", "h1"], required=True)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES)
    ap.add_argument("--n_warmup", type=int, default=N_WARMUP)
    ap.add_argument("--n_evals", type=int, default=N_EVALS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fn_name = f"{args.robot}_constrained_aba_jac"
    f = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, f"{fn_name}.casadi"))

    gfn = CusadiFunction2(f)
    print(f"[lp-constrained] {fn_name}  n_instr={f.n_instructions():,}  sz_w={f.sz_w()}")
    print(f"[lp-constrained] {gfn.dag_info}")

    # For each batch size, time on the auto-selected mode.
    results: dict[int, dict] = {}
    for bs in args.batch_sizes:
        inputs = _random_inputs(f, bs)
        for _ in range(args.n_warmup):
            gfn.evaluate(inputs)
        torch.cuda.synchronize()
        s = np.empty(args.n_evals, dtype=np.float64)
        for i in range(args.n_evals):
            gfn.evaluate(inputs)
            s[i] = gfn.eval_time
        torch.cuda.synchronize()
        med = float(np.median(s))
        mean = float(np.mean(s))
        p25 = float(np.percentile(s, 25))
        p75 = float(np.percentile(s, 75))
        print(f"[lp-constrained] batch={bs:>4d}  median={med * 1e6:.1f} us  "
              f"mean={mean * 1e6:.1f} us  IQR [{p25 * 1e6:.1f}, {p75 * 1e6:.1f}]  "
              f"mode={gfn.dag_info.default_mode}")
        results[bs] = {
            "batch": bs, "median_s": med, "mean_s": mean,
            "p25_s": p25, "p75_s": p75, "min_s": float(np.min(s)),
            "n_samples": int(len(s)), "mode": gfn.dag_info.default_mode,
        }

    out = {
        "robot": args.robot,
        "dtype": "fp64",
        "mode_used": gfn.dag_info.default_mode,
        "n_instr": int(f.n_instructions()),
        "sz_w": int(f.sz_w()),
        "peak_live_slots": int(gfn.dag_info.peak_live_slots),
        "shared_mem_bytes": int(gfn.dag_info.shared_mem_bytes),
        "fits_in_smem": bool(gfn.dag_info.fits_in_smem),
        "tiled_n_chunks": int(gfn.dag_info.tiled_n_chunks) if not gfn.dag_info.fits_in_smem else 0,
        "device": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batch_sizes": args.batch_sizes,
        "n_warmup": args.n_warmup,
        "n_evals": args.n_evals,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\n[lp-constrained] wrote {args.out}")


if __name__ == "__main__":
    main()
