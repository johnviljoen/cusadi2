"""Table VII — MPC-pattern timing (H1 iLQR-style, K=5 iters, various N_stages)."""
from __future__ import annotations

from validation._helpers import (benchmark_kernel_only, build_lp_fn,
                                  load_casadi_fn, make_random_inputs,
                                  print_header, repo_path, SequentialFn)

K_ITERS = 5
N_STAGES = [10, 20, 50]
SEQ_BUILD_DIR = repo_path("build")
DYNAMICS_FRACTION = 0.70   # fraction of MPC step budget spent in dynamics


def run_section_VIE(n_warmup: int = 10, n_timed: int = 100) -> dict:
    print_header("SECTION VI-E / TABLE VII : H1 iLQR MPC-pattern timing")
    f = load_casadi_fn("h1_aba_jac_analytical")
    print(f"  Function: h1_aba_jac_analytical  n_instructions={f.n_instructions():,}")
    print(f"  {'N_stages':>10} {'SeqK=5 ms':>12} {'LP K=5 ms':>12} "
          f"{'Speedup':>10} {'Seq Hz':>10} {'LP Hz':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    out = {}
    for n_stages in N_STAGES:
        import torch
        inputs = make_random_inputs(f, n_stages, seed=42)
        # LP — kernel-only time
        lp, _plan, _mode = build_lp_fn(f, n_stages)
        lp_iter_us = benchmark_kernel_only(lambda: lp.evaluate(inputs),
                                            lambda: lp.eval_time,
                                            n_warmup, n_timed)
        lp_total_ms = lp_iter_us * K_ITERS / 1000.0
        del lp
        torch.cuda.empty_cache()
        # Seq — kernel-only
        seq = SequentialFn(f, n_stages, SEQ_BUILD_DIR)
        seq.set_inputs(inputs)
        seq_iter_us = benchmark_kernel_only(seq.evaluate, lambda: seq.eval_time,
                                             n_warmup, n_timed)
        seq_total_ms = seq_iter_us * K_ITERS / 1000.0
        del seq
        torch.cuda.empty_cache()

        speedup = seq_total_ms / lp_total_ms if lp_total_ms > 0 else float("nan")
        seq_hz = 1000.0 / (seq_total_ms / DYNAMICS_FRACTION) if seq_total_ms > 0 else float("nan")
        lp_hz = 1000.0 / (lp_total_ms / DYNAMICS_FRACTION) if lp_total_ms > 0 else float("nan")

        print(f"  {n_stages:>10} {seq_total_ms:>12.2f} {lp_total_ms:>12.2f} "
              f"{speedup:>9.2f}x {seq_hz:>10.0f} {lp_hz:>10.0f}")

        out[str(n_stages)] = {
            "seq_ms": seq_total_ms, "lp_ms": lp_total_ms,
            "speedup": speedup, "seq_hz": seq_hz, "lp_hz": lp_hz,
            "lp_single_iter_us": lp_iter_us,
            "seq_single_iter_us": seq_iter_us,
        }
    return out
