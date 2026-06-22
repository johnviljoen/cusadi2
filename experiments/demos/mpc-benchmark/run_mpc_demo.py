"""
MPC-pattern timing harness for the CusADi level-parallel demo.

Instead of constructing a full symbolic SRBM MPC (which produces a deep/narrow
DAG that hides LP's advantage), this harness reuses the already-compiled
Pinocchio analytical-derivative kernels (h1_aba_jac_analytical, a1_*) and
drives them in the *access pattern* of a real-time MPC / MPPI loop:

  H1 whole-body MPC pattern (iLQR / SQP)
    for K iterations:
      evaluate dynamics Jacobian at all N_stages stages in parallel (batch=N_stages)

  MPPI sampling pattern
    for T sequential timesteps:
      evaluate dynamics at all N_samples rollouts in parallel (batch=N_samples)

Timing: CUDA-event ms inside each kernel (`eval_time`), N_warmup + N_evals,
median and IQR reported. Sequential and level-parallel paths share identical
input tensors per batch size.

No MPC is actually solved — KKT assembly and linear solves are explicitly NOT
measured. Reported MPC frequencies assume dynamics evaluation is ~70% of the
total iteration cost (a standard empirical split for analytical-derivative
MPC pipelines on GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

import casadi
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cusadi import CUSADI_LP_BUILD_DIR, CUSADI_PYTORCH_BUILD_DIR, CUSADI_FUNCTION_DIR  # noqa: E402
from cusadi.CusadiFunction import CusadiFunction  # noqa: E402
from cusadi.level_parallel.dag import extract_level_plan  # noqa: E402
from cusadi.level_parallel.function import CusadiLevelParallelFunction  # noqa: E402


N_WARMUP = 10
N_EVALS = 100
DYNAMICS_FRACTION = 0.70  # assumed fraction of MPC step cost spent in dynamics


@dataclass
class KernelTiming:
    method: str  # "sequential" or "level_parallel"
    fn_name: str
    batch: int
    median_s: float
    mean_s: float
    p25_s: float
    p75_s: float
    min_s: float
    max_s: float
    n_samples: int


def _load_fn(name: str) -> casadi.Function:
    path = os.path.join(CUSADI_FUNCTION_DIR, f"{name}.casadi")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing CasADi function: {path}")
    return casadi.Function.load(path)


def _check_so(name: str, require_lp: bool = False) -> None:
    if require_lp:
        path = os.path.join(CUSADI_LP_BUILD_DIR, "liblevel_parallel_kernel.so")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "liblevel_parallel_kernel.so not built. Run "
                "`python tests/test_level_parallel_cuda.py --fn panda_aba_jac_analytical` first."
            )
    else:
        path = os.path.join(CUSADI_PYTORCH_BUILD_DIR, f"lib{name}.so")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"lib{name}.so not built. Run `python run_codegen.py --fn={name}` "
                f"and rebuild with `cd build && make -j`."
            )


def _make_inputs(fn: casadi.Function, batch: int, seed: int = 0) -> list[torch.Tensor]:
    """Physically plausible random (q, v, tau) for rigid-body dynamics."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ins = []
    for i in range(fn.n_in()):
        nnz = fn.nnz_in(i)
        name = fn.name_in(i).lower()
        if "q" == name or name.startswith("q"):
            t = 0.5 * torch.randn(batch, nnz, device="cuda", dtype=torch.double, generator=g)
        elif name.startswith("v"):
            t = 0.3 * torch.randn(batch, nnz, device="cuda", dtype=torch.double, generator=g)
        elif name.startswith("tau") or name.startswith("u"):
            t = 5.0 * torch.randn(batch, nnz, device="cuda", dtype=torch.double, generator=g)
        else:
            t = torch.randn(batch, nnz, device="cuda", dtype=torch.double, generator=g)
        ins.append(t.contiguous())
    return ins


def _time_kernel(evaluator, inputs, n_warmup: int, n_evals: int) -> np.ndarray:
    for _ in range(n_warmup):
        evaluator.evaluate(inputs)
    torch.cuda.synchronize()
    samples = np.empty(n_evals, dtype=np.float64)
    for i in range(n_evals):
        evaluator.evaluate(inputs)
        samples[i] = evaluator.eval_time  # seconds
    torch.cuda.synchronize()
    return samples


def benchmark_batch(
    fn: casadi.Function, batch: int, seed: int = 0,
    n_warmup: int = N_WARMUP, n_evals: int = N_EVALS,
) -> dict[str, KernelTiming]:
    inputs = _make_inputs(fn, batch, seed=seed)

    seq = CusadiFunction(fn, batch)
    seq_samples = _time_kernel(seq, inputs, n_warmup, n_evals)

    lp = CusadiLevelParallelFunction(fn, batch, lib_name="level_parallel_kernel")
    lp_samples = _time_kernel(lp, inputs, n_warmup, n_evals)

    def _summarize(method: str, s: np.ndarray) -> KernelTiming:
        return KernelTiming(
            method=method,
            fn_name=fn.name(),
            batch=batch,
            median_s=float(np.median(s)),
            mean_s=float(np.mean(s)),
            p25_s=float(np.percentile(s, 25)),
            p75_s=float(np.percentile(s, 75)),
            min_s=float(np.min(s)),
            max_s=float(np.max(s)),
            n_samples=len(s),
        )

    return {
        "sequential": _summarize("sequential", seq_samples),
        "level_parallel": _summarize("level_parallel", lp_samples),
    }


def h1_mpc_pattern(
    n_stages_list: list[int] | None = None,
    k_iters: int = 5,
) -> dict:
    if n_stages_list is None:
        n_stages_list = [10, 20, 50]
    print("\n" + "=" * 100)
    print(f"  H1 whole-body MPC pattern (h1_aba_jac_analytical)")
    print(f"  Per MPC step: K={k_iters} SQP iterations, each batches dynamics at N_stages")
    print("=" * 100)

    fn = _load_fn("h1_aba_jac_analytical")
    _check_so("h1_aba_jac_analytical")
    _check_so("", require_lp=True)
    plan = extract_level_plan(fn)
    print(f"  function: n_instr={fn.n_instructions():,}  sz_w={fn.sz_w()}  "
          f"DAG ops={plan.total_compute_ops:,}  levels={plan.n_compute_levels}  "
          f"peak_live={plan.peak_live_slots} ({plan.peak_live_slots * 8 / 1024:.1f} KB)")

    results = {}
    for n_stages in n_stages_list:
        print(f"\n  --- N_stages = {n_stages}  (batch={n_stages}) ---")
        t = benchmark_batch(fn, n_stages)
        seq_one = t["sequential"].median_s
        lp_one = t["level_parallel"].median_s
        # Total dynamics wall-clock per MPC step = K * single-iteration evaluation
        seq_step = k_iters * seq_one
        lp_step = k_iters * lp_one
        speedup = seq_step / lp_step if lp_step > 0 else float("inf")

        # MPC step frequency assuming dynamics is DYNAMICS_FRACTION of total cost
        seq_full = seq_step / DYNAMICS_FRACTION
        lp_full = lp_step / DYNAMICS_FRACTION
        seq_hz = 1.0 / seq_full if seq_full > 0 else 0.0
        lp_hz = 1.0 / lp_full if lp_full > 0 else 0.0

        print(f"    single-iter  seq = {seq_one * 1e6:8.1f} us   lp = {lp_one * 1e6:8.1f} us "
              f"(speedup {seq_one / lp_one:.2f}x)")
        print(f"    K={k_iters} iters  seq = {seq_step * 1e6:8.1f} us  lp = {lp_step * 1e6:8.1f} us "
              f"(speedup {speedup:.2f}x)")
        print(f"    implied MPC freq (assuming dyn={DYNAMICS_FRACTION:.0%} of step cost): "
              f"seq = {seq_hz:7.1f} Hz   lp = {lp_hz:7.1f} Hz")

        # Slack at 100 Hz (10 ms budget)
        budget_10ms = 10e-3
        seq_slack = budget_10ms - seq_step
        lp_slack = budget_10ms - lp_step
        print(f"    at 100 Hz MPC (10 ms budget): slack for linear solve+KKT: "
              f"seq = {seq_slack * 1000:+.2f} ms   lp = {lp_slack * 1000:+.2f} ms")

        results[n_stages] = {
            "timings": {k: asdict(v) for k, v in t.items()},
            "k_iters": k_iters,
            "seq_step_s": seq_step, "lp_step_s": lp_step,
            "speedup": speedup,
            "seq_mpc_hz": seq_hz, "lp_mpc_hz": lp_hz,
            "seq_slack_at_100hz_s": seq_slack, "lp_slack_at_100hz_s": lp_slack,
        }

    # Evaluations-within-budget summary (how many K*N_stages fits inside B)
    print(f"\n  --- Largest N_stages whose K={k_iters}-iter dynamics cost fits inside a time budget ---")
    print(f"  {'budget':>10}  {'seq N_stages':>14}  {'lp N_stages':>14}  {'ratio':>10}")
    for budget_ms in [1.0, 2.0, 5.0, 10.0]:
        seq_best: int | None = None
        lp_best: int | None = None
        for n_stages, r in sorted(results.items()):
            if r["seq_step_s"] * 1000 <= budget_ms:
                seq_best = n_stages if seq_best is None else max(seq_best, n_stages)
            if r["lp_step_s"] * 1000 <= budget_ms:
                lp_best = lp_best if lp_best is not None and lp_best > n_stages else n_stages
        seq_cell = f"{seq_best}" if seq_best is not None else "(none fit)"
        lp_cell = f"{lp_best}" if lp_best is not None else "(none fit)"
        if seq_best and lp_best:
            ratio_cell = f"{lp_best / seq_best:.2f}x"
        elif lp_best:
            ratio_cell = "lp-only"
        else:
            ratio_cell = "-"
        print(f"  {budget_ms:>8.1f} ms  {seq_cell:>14}  {lp_cell:>14}  {ratio_cell:>10}")

    return results


def mppi_pattern(
    n_samples_list: list[int] | None = None,
    horizon: int = 20,
) -> dict:
    if n_samples_list is None:
        n_samples_list = [128, 256]
    print("\n" + "=" * 100)
    print(f"  A1 MPPI pattern (a1_aba_jac_analytical)")
    print(f"  Per MPPI control step: T={horizon} sequential timesteps, "
          f"each evaluates dynamics for N_samples rollouts in parallel")
    print("=" * 100)

    fn = _load_fn("a1_aba_jac_analytical")
    _check_so("a1_aba_jac_analytical")
    _check_so("", require_lp=True)
    plan = extract_level_plan(fn)
    print(f"  function: n_instr={fn.n_instructions():,}  sz_w={fn.sz_w()}  "
          f"DAG ops={plan.total_compute_ops:,}  levels={plan.n_compute_levels}  "
          f"peak_live={plan.peak_live_slots} ({plan.peak_live_slots * 8 / 1024:.1f} KB)")

    results = {}
    for N in n_samples_list:
        print(f"\n  --- N_samples = {N}  (batch={N}, T={horizon} timesteps) ---")
        t = benchmark_batch(fn, N)
        seq_one = t["sequential"].median_s
        lp_one = t["level_parallel"].median_s
        seq_roll = horizon * seq_one
        lp_roll = horizon * lp_one
        speedup = seq_roll / lp_roll if lp_roll > 0 else float("inf")
        seq_hz = 1.0 / seq_roll if seq_roll > 0 else 0.0
        lp_hz = 1.0 / lp_roll if lp_roll > 0 else 0.0

        print(f"    per-timestep   seq = {seq_one * 1e6:8.1f} us  lp = {lp_one * 1e6:8.1f} us "
              f"(speedup {seq_one / lp_one:.2f}x)")
        print(f"    full T={horizon}       seq = {seq_roll * 1e6:8.1f} us  lp = {lp_roll * 1e6:8.1f} us "
              f"(speedup {speedup:.2f}x)")
        print(f"    MPPI freq (rollout only): "
              f"seq = {seq_hz:7.1f} Hz   lp = {lp_hz:7.1f} Hz")

        results[N] = {
            "timings": {k: asdict(v) for k, v in t.items()},
            "horizon": horizon,
            "seq_rollout_s": seq_roll, "lp_rollout_s": lp_roll,
            "speedup": speedup,
            "seq_mppi_hz": seq_hz, "lp_mppi_hz": lp_hz,
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1_stages", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--h1_k_iters", type=int, default=5)
    ap.add_argument("--mppi_samples", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--mppi_horizon", type=int, default=20)
    ap.add_argument("--out_dir", default="demo/mpc-benchmark/results")
    ap.add_argument("--skip_h1", action="store_true")
    ap.add_argument("--skip_mppi", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[run_mpc_demo] CUDA device = {torch.cuda.get_device_name(0)}")
    print(f"[run_mpc_demo] N_warmup={N_WARMUP}  N_evals={N_EVALS}  "
          f"dyn_fraction={DYNAMICS_FRACTION}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"device": torch.cuda.get_device_name(0),
           "n_warmup": N_WARMUP, "n_evals": N_EVALS,
           "dynamics_fraction": DYNAMICS_FRACTION,
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if not args.skip_h1:
        out["h1_mpc"] = h1_mpc_pattern(args.h1_stages, args.h1_k_iters)
    if not args.skip_mppi:
        out["mppi_a1"] = mppi_pattern(args.mppi_samples, args.mppi_horizon)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(args.out_dir, f"mpc_demo_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[run_mpc_demo] Saved raw results to {out_path}")


if __name__ == "__main__":
    main()
