"""
Level-parallel (LP) and sequential ABA-Jacobian benchmark — output JSON in a
format that matches bench_mjx_jacobian.py so the two can be joined.

For each robot in {a1, h1} and each method in {level_parallel, sequential}:
  Measures per-call time across batch sizes, median of N_EVALS after N_WARMUP.
  Uses the .so timing returned by the kernel wrapper (CUDA events inside).

Always fp64 — that's what the compiled kernels produce.

Usage:
    CUDA_VISIBLE_DEVICES=5 python benchmark/mjx-comparison/bench_lp_jacobian.py \
        --robot a1 --method level_parallel --out results/a1_lp_fp64.json
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
from src import CUSADI_BUILD_DIR, CUSADI_FUNCTION_DIR  # noqa: E402
from src.CusadiFunction import CusadiFunction  # noqa: E402
from src.level_parallel.dag import extract_level_plan  # noqa: E402
from src.level_parallel.function import CusadiLevelParallelFunction  # noqa: E402


N_WARMUP = 10
N_EVALS = 100
BATCH_SIZES = [1, 8, 32, 128, 256, 512]

ROBOT_TO_FN = {
    "a1": "a1_aba_jac_analytical",
    "h1": "h1_aba_jac_analytical",
}


def _random_inputs(fn: casadi.Function, bs: int, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out = []
    for i in range(fn.n_in()):
        nnz = fn.nnz_in(i)
        nm = fn.name_in(i).lower()
        if nm.startswith("q"):
            scale = 0.3
        elif nm.startswith("v"):
            scale = 0.2
        else:
            scale = 1.0
        t = scale * torch.randn(bs, nnz, device="cuda", dtype=torch.double, generator=gen)
        out.append(t.contiguous())
    return out


def _time(evaluator, inputs, n_warmup: int, n_evals: int) -> np.ndarray:
    for _ in range(n_warmup):
        evaluator.evaluate(inputs)
    torch.cuda.synchronize()
    out = np.empty(n_evals, dtype=np.float64)
    for i in range(n_evals):
        evaluator.evaluate(inputs)
        out[i] = evaluator.eval_time  # seconds
    torch.cuda.synchronize()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=["a1", "h1"], required=True)
    ap.add_argument("--method", choices=["level_parallel", "sequential"], required=True)
    ap.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES)
    ap.add_argument("--n_warmup", type=int, default=N_WARMUP)
    ap.add_argument("--n_evals", type=int, default=N_EVALS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    fn_name = ROBOT_TO_FN[args.robot]
    fn = casadi.Function.load(os.path.join(CUSADI_FUNCTION_DIR, f"{fn_name}.casadi"))
    print(f"[lp] {fn_name}: n_in={fn.n_in()}  n_out={fn.n_out()}  "
          f"n_instr={fn.n_instructions():,}  sz_w={fn.sz_w()}")

    plan = extract_level_plan(fn)
    print(f"[lp] DAG: ops={plan.total_compute_ops:,}  levels={plan.n_compute_levels}  "
          f"peak_live={plan.peak_live_slots} ({plan.peak_live_slots * 8 / 1024:.1f} KB)")

    # Check .so existence
    if args.method == "level_parallel":
        so = os.path.join(CUSADI_BUILD_DIR, "liblevel_parallel_kernel.so")
    else:
        so = os.path.join(CUSADI_BUILD_DIR, f"lib{fn_name}.so")
    if not os.path.exists(so):
        print(f"[lp] ERROR: {so} not built", file=sys.stderr)
        sys.exit(2)

    results = {}
    for bs in args.batch_sizes:
        print(f"\n[lp] batch={bs}")
        if args.method == "level_parallel":
            ev = CusadiLevelParallelFunction(fn, bs, lib_name="level_parallel_kernel")
        else:
            ev = CusadiFunction(fn, bs)
        inputs = _random_inputs(fn, bs)
        samples = _time(ev, inputs, args.n_warmup, args.n_evals)
        med = float(np.median(samples))
        mean = float(np.mean(samples))
        p25 = float(np.percentile(samples, 25))
        p75 = float(np.percentile(samples, 75))
        mn = float(np.min(samples))
        print(f"[lp]   median = {med * 1e6:.1f} us  mean = {mean * 1e6:.1f} us  "
              f"min = {mn * 1e6:.1f} us  IQR [{p25 * 1e6:.1f}, {p75 * 1e6:.1f}]")

        results[bs] = {
            "batch": bs,
            "jit_compile_plus_run_s": None,
            "median_s": med, "mean_s": mean,
            "p25_s": p25, "p75_s": p75, "min_s": mn,
            "n_samples": len(samples),
            "jac_shapes": {
                "Jq": list(fn.sparsity_out(0).shape),
                "Jv": list(fn.sparsity_out(1).shape),
                "Jt": list(fn.sparsity_out(2).shape),
            },
            "jac_nnz": {
                "Jq": int(fn.nnz_out(0)),
                "Jv": int(fn.nnz_out(1)),
                "Jt": int(fn.nnz_out(2)),
            },
        }

    out_dict = {
        "robot": args.robot,
        "method": args.method,
        "dtype": "fp64",
        "n_instr": int(fn.n_instructions()),
        "sz_w": int(fn.sz_w()),
        "dag_ops": int(plan.total_compute_ops),
        "dag_levels": int(plan.n_compute_levels),
        "peak_live_slots": int(plan.peak_live_slots),
        "n_warmup": args.n_warmup,
        "n_evals": args.n_evals,
        "device": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "batch_sizes": args.batch_sizes,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2, default=str)
    print(f"\n[lp] wrote {args.out}")


if __name__ == "__main__":
    main()
