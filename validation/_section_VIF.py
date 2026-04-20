"""Table VIII — MPPI rollout timing (A1, T=20 timesteps × N_samples)."""
from __future__ import annotations

from validation._helpers import (benchmark_kernel_only, build_lp_fn,
                                  load_casadi_fn, make_random_inputs,
                                  print_header, repo_path, SequentialFn)

T_HORIZON = 20
N_SAMPLES = [128, 256]
SEQ_BUILD_DIR = repo_path("build")


def run_section_VIF(n_warmup: int = 10, n_timed: int = 100) -> dict:
    print_header("SECTION VI-F / TABLE VIII : A1 MPPI rollout timing (T=20)")
    f = load_casadi_fn("a1_aba_jac_analytical")
    print(f"  Function: a1_aba_jac_analytical  n_instructions={f.n_instructions():,}")
    print(f"  {'N_samples':>10} {'Seq T=20 ms':>14} {'LP T=20 ms':>14} "
          f"{'Speedup':>10} {'Seq Hz':>10} {'LP Hz':>10}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*10} {'-'*10} {'-'*10}")

    out = {}
    for n_samples in N_SAMPLES:
        import torch
        inputs = make_random_inputs(f, n_samples, seed=42)
        # LP — kernel-only time
        lp, _plan, _mode = build_lp_fn(f, n_samples)
        lp_iter_us = benchmark_kernel_only(lambda: lp.evaluate(inputs),
                                            lambda: lp.eval_time,
                                            n_warmup, n_timed)
        lp_total_ms = lp_iter_us * T_HORIZON / 1000.0
        del lp
        torch.cuda.empty_cache()
        # Seq — kernel-only
        seq = SequentialFn(f, n_samples, SEQ_BUILD_DIR)
        seq.set_inputs(inputs)
        seq_iter_us = benchmark_kernel_only(seq.evaluate, lambda: seq.eval_time,
                                             n_warmup, n_timed)
        seq_total_ms = seq_iter_us * T_HORIZON / 1000.0
        del seq
        torch.cuda.empty_cache()

        speedup = seq_total_ms / lp_total_ms if lp_total_ms > 0 else float("nan")
        seq_hz = 1000.0 / seq_total_ms if seq_total_ms > 0 else float("nan")
        lp_hz = 1000.0 / lp_total_ms if lp_total_ms > 0 else float("nan")
        print(f"  {n_samples:>10} {seq_total_ms:>14.2f} {lp_total_ms:>14.2f} "
              f"{speedup:>9.2f}x {seq_hz:>10.0f} {lp_hz:>10.0f}")
        out[str(n_samples)] = {
            "seq_ms": seq_total_ms, "lp_ms": lp_total_ms,
            "speedup": speedup, "seq_hz": seq_hz, "lp_hz": lp_hz,
        }
    return out
