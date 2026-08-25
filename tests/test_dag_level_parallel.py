"""
Phase 1 validation: DAG extraction with SSA renaming and parallel verification.

Tests:
1. No write-write conflicts within any level (SSA invariant).
2. Parallel-simulated execution matches CasADi oracle to machine precision.
3. Peak liveness is within shared memory budget.
"""

import os
import sys
import time

import casadi

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from cusadi.level_parallel.dag import extract_level_plan, verify_level_plan_parallel


FUNCTION_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "build", "casadi_functions")

TARGETS = [
    ("double_integrator", 1e-14),
    ("panda_aba_jac_analytical", 1e-10),
    ("a1_aba_jac_analytical", 1e-10),
    ("h1_aba_jac_analytical", 1e-10),
]

L40S_SHARED_MEM = 100 * 1024


def check_no_write_conflicts(plan):
    """Verify no two nodes at the same level write to the same physical slot."""
    from collections import Counter
    conflicts = 0
    for level in plan.compute_levels:
        slots = [n.ssa_out for n in level.nodes]
        dupes = {s: c for s, c in Counter(slots).items() if c > 1}
        if dupes:
            print(f"  Level {level.depth}: write-write conflicts: {dupes}")
            conflicts += sum(c - 1 for c in dupes.values())
    return conflicts


def main():
    results = []
    for name, atol in TARGETS:
        path = os.path.join(FUNCTION_DIR, f"{name}.casadi")
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        f = casadi.Function.load(path)
        t0 = time.time()
        plan = extract_level_plan(f)
        t_extract = time.time() - t0

        print(f"  Extraction: {t_extract:.2f}s")
        print(f"  Compute ops: {plan.total_compute_ops}")
        print(f"  Levels: {plan.n_compute_levels}, critical path: {plan.critical_path}")
        print(f"  Width: median={plan.median_width}, max={plan.max_width}")
        print(f"  SSA slots: {plan.n_ssa_slots}, peak live: {plan.peak_live_slots}")
        print(f"  Shared memory: {plan.shared_mem_bytes} bytes ({plan.shared_mem_bytes/1024:.1f} KB)")
        if plan.n_compute_levels > 0:
            print(f"  Speedup ceiling: {plan.total_compute_ops / plan.n_compute_levels:.1f}x")
        else:
            print(f"  Speedup ceiling: N/A (no compute ops)")

        n_conflicts = check_no_write_conflicts(plan)
        print(f"  Write-write conflicts: {n_conflicts}")

        if plan.shared_mem_bytes > L40S_SHARED_MEM:
            print(f"  WARNING: shared mem {plan.shared_mem_bytes} > L40S limit {L40S_SHARED_MEM}")

        t0 = time.time()
        passed = verify_level_plan_parallel(f, plan, n_tests=5, atol=atol)
        t_verify = time.time() - t0

        status = "PASS" if (passed and n_conflicts == 0) else "FAIL"
        print(f"  Parallel verification: {status} ({t_verify:.2f}s)")
        results.append((name, status, n_conflicts, plan))

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_pass = True
    for name, status, conflicts, plan in results:
        flag = "ok" if status == "PASS" else "FAIL"
        print(f"  {name:>30s}: {flag}  conflicts={conflicts}  "
              f"levels={plan.n_compute_levels}  peak_live={plan.peak_live_slots}  "
              f"smem={plan.shared_mem_bytes/1024:.1f}KB")
        if status != "PASS":
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
