#!/bin/bash
# Run MJX Jacobian benchmarks WITH CONTACT ENABLED for A1 + H1, fp32 + fp64.
# Swaps unsupported CYLINDER geoms to CAPSULE so MJX's contact pipeline can run.
# Each config takes ~20-60 minutes depending on batch sizes (contact iterations
# dominate at large batch).
#
# Assumes GPU 5 is idle. Skips configs whose JSON already exists.
set -u
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=5
PY=/home/john/.conda/envs/cusadi2/bin/python
RES=experiments/benchmarks/mjx-comparison/results
mkdir -p "$RES"

run() {
    local out="$1"; shift
    if [ -f "$out" ]; then
        echo "[skip] $out exists"
        return 0
    fi
    echo "[run ] $out"
    "$PY" -u "$@" > "${out%.json}.log" 2>&1
}

run "$RES/a1_mjx_fp64_contact.json"  experiments/benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot a1 --dtype fp64 --with_contact --out "$RES/a1_mjx_fp64_contact.json"
run "$RES/h1_mjx_fp64_contact.json"  experiments/benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot h1 --dtype fp64 --with_contact --out "$RES/h1_mjx_fp64_contact.json"
run "$RES/a1_mjx_fp32_contact.json"  experiments/benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot a1 --dtype fp32 --with_contact --out "$RES/a1_mjx_fp32_contact.json"
run "$RES/h1_mjx_fp32_contact.json"  experiments/benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot h1 --dtype fp32 --with_contact --out "$RES/h1_mjx_fp32_contact.json"

echo "[done] contact-on MJX benchmarks complete"
