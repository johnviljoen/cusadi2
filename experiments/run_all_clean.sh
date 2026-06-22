#!/bin/bash
# Sequential rerun of all constrained-dynamics measurements.
# Order: fast LP runs first (so partial results exist even if we ctrl-C later),
# then the expensive MJX-with-contact runs.
# Paths below are relative to this script's directory (experiments/).
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
PY=/home/john/.conda/envs/cusadi2/bin/python

run() {
    local out="$1"; shift
    echo ""
    echo "========================================"
    echo "[run ] $out"
    echo "========================================"
    "$PY" -u "$@" > "${out%.json}.log" 2>&1
    echo "  exit=$? ; wrote $out"
    grep -E "batch=|median" "${out%.json}.log" | tail -12 || true
}

# --- LP constrained (fast, ~5 min total) ---
run exploration/constrained_dynamics/results/a1_constrained_lp.json \
    exploration/constrained_dynamics/bench_constrained.py --robot a1 \
    --out exploration/constrained_dynamics/results/a1_constrained_lp.json
run exploration/constrained_dynamics/results/h1_constrained_lp.json \
    exploration/constrained_dynamics/bench_constrained.py --robot h1 \
    --out exploration/constrained_dynamics/results/h1_constrained_lp.json

# --- MJX contact, fp64 only (fp32 is secondary — can add later) ---
run benchmarks/mjx-comparison/results/a1_mjx_fp64_contact.json \
    benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot a1 --dtype fp64 --with_contact \
    --out benchmarks/mjx-comparison/results/a1_mjx_fp64_contact.json
run benchmarks/mjx-comparison/results/h1_mjx_fp64_contact.json \
    benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot h1 --dtype fp64 --with_contact \
    --out benchmarks/mjx-comparison/results/h1_mjx_fp64_contact.json

echo "[done] all reruns complete"
