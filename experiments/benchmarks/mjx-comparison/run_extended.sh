#!/bin/bash
# One-shot driver for the post-defense-prep additions:
#   1. Extended-batch forward dynamics sweep (B up to 10000)
#   2. Constrained ABA Jacobian: LP + MJX with contact (A1 4-foot, H1 2-foot)
# Runs serially on GPU 1. Skips configs whose JSON already exists.
set -u
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=1
PY=/home/john/.conda/envs/cusadi2/bin/python
RES=experiments/benchmarks/mjx-comparison/results
mkdir -p "$RES"

run() {
    local out="$1"; shift
    if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip] $out exists"
        return 0
    fi
    echo "[run ] $out"
    echo "[cmd ] $PY -u $*"
    "$PY" -u "$@" 2>&1 | tee "${out%.json}.log"
}

# ============================================================================
# Phase 1 — Extended-batch forward dynamics (B up to 10000)
# Re-runs the existing 10 fwd configs at the larger batch range, overwriting
# the B<=512 JSONs from the main sweep. FORCE=1 so they re-run.
# ============================================================================
EXT_BATCH="--batch_sizes 1 8 32 128 256 512 1024 4096 10000"

# We remove only the fwd JSONs (NOT the Jacobian ones) so the Jacobian plot
# keeps its existing GPU-1 data.
echo "=== Phase 1: extended fwd dynamics ==="
for f in $RES/{a1,h1}_fwd_{lp,seq,mjx_fp32,mjx_fp64,mjw}.json; do
    [ -f "$f" ] && rm -f "$f"
done

run "$RES/a1_fwd_lp.json"        experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend level_parallel --out "$RES/a1_fwd_lp.json" $EXT_BATCH
run "$RES/a1_fwd_seq.json"       experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend sequential --out "$RES/a1_fwd_seq.json" $EXT_BATCH
run "$RES/h1_fwd_lp.json"        experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend level_parallel --out "$RES/h1_fwd_lp.json" $EXT_BATCH
run "$RES/h1_fwd_seq.json"       experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend sequential --out "$RES/h1_fwd_seq.json" $EXT_BATCH
run "$RES/a1_fwd_mjw.json"       experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend mujoco_warp --out "$RES/a1_fwd_mjw.json" $EXT_BATCH
run "$RES/h1_fwd_mjw.json"       experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend mujoco_warp --out "$RES/h1_fwd_mjw.json" $EXT_BATCH
run "$RES/a1_fwd_mjx_fp32.json"  experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend mjx --dtype fp32 --out "$RES/a1_fwd_mjx_fp32.json" $EXT_BATCH
run "$RES/a1_fwd_mjx_fp64.json"  experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend mjx --dtype fp64 --out "$RES/a1_fwd_mjx_fp64.json" $EXT_BATCH
run "$RES/h1_fwd_mjx_fp32.json"  experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend mjx --dtype fp32 --out "$RES/h1_fwd_mjx_fp32.json" $EXT_BATCH
run "$RES/h1_fwd_mjx_fp64.json"  experiments/benchmarks/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend mjx --dtype fp64 --out "$RES/h1_fwd_mjx_fp64.json" $EXT_BATCH

# ============================================================================
# Phase 2 — LP constrained ABA Jacobian (A1 4-foot, H1 2-foot)
# Same batch range as the main Jacobian comparison (max B=512 since
# contact-on MJX gets prohibitively slow past that).
# ============================================================================
JAC_BATCH="--batch_sizes 1 8 32 128 256 512"
echo
echo "=== Phase 2: LP constrained Jacobian ==="
run "$RES/a1_lp_fp64_contact.json"  experiments/benchmarks/mjx-comparison/bench_lp_jacobian.py \
    --robot a1 --method level_parallel --with_contact \
    --out "$RES/a1_lp_fp64_contact.json" $JAC_BATCH
run "$RES/h1_lp_fp64_contact.json"  experiments/benchmarks/mjx-comparison/bench_lp_jacobian.py \
    --robot h1 --method level_parallel --with_contact \
    --out "$RES/h1_lp_fp64_contact.json" $JAC_BATCH

# ============================================================================
# Phase 3 — MJX with contact enabled (foot contacts only)
# 4 configs: a1+h1 × fp32+fp64. Each config has heavy JIT + heavy runtime
# (the contact LCP solver is the bottleneck inside autodiff). Cap at B=512.
# ============================================================================
echo
echo "=== Phase 3: MJX with contact ==="
run "$RES/a1_mjx_fp32_contact.json"  experiments/benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot a1 --dtype fp32 --with_contact \
    --out "$RES/a1_mjx_fp32_contact.json" $JAC_BATCH
run "$RES/a1_mjx_fp64_contact.json"  experiments/benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot a1 --dtype fp64 --with_contact \
    --out "$RES/a1_mjx_fp64_contact.json" $JAC_BATCH
run "$RES/h1_mjx_fp32_contact.json"  experiments/benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot h1 --dtype fp32 --with_contact \
    --out "$RES/h1_mjx_fp32_contact.json" $JAC_BATCH
run "$RES/h1_mjx_fp64_contact.json"  experiments/benchmarks/mjx-comparison/bench_mjx_jacobian.py \
    --robot h1 --dtype fp64 --with_contact \
    --out "$RES/h1_mjx_fp64_contact.json" $JAC_BATCH

echo
echo "[done] extended sweep complete"
