"""Table IV — Unconstrained ABA Jacobian, LP vs Sequential on A1/H1."""
from __future__ import annotations

from validation._helpers import (benchmark_kernel_only, build_lp_fn,
                                  load_casadi_fn, make_random_inputs,
                                  print_header, repo_path, SequentialFn,
                                  fmt_us)

ROBOTS = ["a1", "h1"]
BATCHES = [1, 128, 256]
SEQ_BUILD_DIR = repo_path("build")   # a1/h1_aba_jac_analytical.so lives in build/


def run_section_VIC(n_warmup: int = 10, n_timed: int = 100) -> dict:
    print_header("SECTION VI-C / TABLE IV : Unconstrained ABA Jacobian LP vs Seq")
    print(f"  {'Robot':<6} {'Kernel':<8}  {'Batch=1':>14} {'Batch=128':>14} {'Batch=256':>14}")
    print(f"  {'-'*6} {'-'*8}  {'-'*14} {'-'*14} {'-'*14}")

    out: dict = {}
    for robot in ROBOTS:
        fn_name = f"{robot}_aba_jac_analytical"
        try:
            f = load_casadi_fn(fn_name)
        except FileNotFoundError as e:
            print(f"  {robot}: MISSING ({e})")
            out[robot] = {"error": "file_missing"}
            continue

        per_robot = {"lp": {}, "seq": {}, "n_instructions": int(f.n_instructions())}
        for batch in BATCHES:
            import torch
            inputs = make_random_inputs(f, batch, seed=42)
            # LP — kernel-only time
            lp, _plan, lp_mode = build_lp_fn(f, batch)
            lp_us = benchmark_kernel_only(lambda: lp.evaluate(inputs),
                                           lambda: lp.eval_time,
                                           n_warmup, n_timed)
            per_robot["lp"][batch] = {"median_us": lp_us, "mode": lp_mode}
            del lp
            torch.cuda.empty_cache()
            # Seq — kernel-only
            try:
                seq = SequentialFn(f, batch, SEQ_BUILD_DIR)
                seq.set_inputs(inputs)
                seq_us = benchmark_kernel_only(seq.evaluate, lambda: seq.eval_time,
                                                n_warmup, n_timed)
                per_robot["seq"][batch] = {"median_us": seq_us}
                del seq
            except FileNotFoundError as e:
                per_robot["seq"][batch] = {"error": str(e)}
            torch.cuda.empty_cache()

        def _s(d, k):
            return d.get(k, {}).get("median_us", float("nan"))
        row_lp = [_s(per_robot["lp"], b) for b in BATCHES]
        row_seq = [_s(per_robot["seq"], b) for b in BATCHES]
        print(f"  {robot.upper():<6} {'LP':<8}  "
              f"{fmt_us(row_lp[0]):>14} {fmt_us(row_lp[1]):>14} {fmt_us(row_lp[2]):>14}")
        print(f"  {robot.upper():<6} {'Seq':<8}  "
              f"{fmt_us(row_seq[0]):>14} {fmt_us(row_seq[1]):>14} {fmt_us(row_seq[2]):>14}")

        out[robot] = per_robot

    return out
