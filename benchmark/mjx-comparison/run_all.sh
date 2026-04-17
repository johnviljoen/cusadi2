#!/bin/bash
# Run every benchmark configuration serially (GPU is shared).
# Skips any config whose output JSON already exists — set --force to rerun.

set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=5
PY=/home/john/.conda/envs/cusadi2/bin/python
RES=benchmark/mjx-comparison/results
mkdir -p "$RES"

# Normalize any legacy filenames from early iterations
for legacy in a1_fp32 a1_fp64 h1_fp32 h1_fp64; do
    if [ -f "$RES/$legacy.json" ] && [ ! -f "$RES/${legacy/_/_mjx_}.json" ]; then
        mv -v "$RES/$legacy.json"  "$RES/${legacy/_/_mjx_}.json"
        [ -f "$RES/$legacy.log" ]  && mv "$RES/$legacy.log"  "$RES/${legacy/_/_mjx_}.log"
    fi
done

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

# ---- MJX autodiff ABA Jacobians ----
run "$RES/a1_mjx_fp32.json"  benchmark/mjx-comparison/bench_mjx_jacobian.py \
    --robot a1 --dtype fp32 --out "$RES/a1_mjx_fp32.json"
run "$RES/a1_mjx_fp64.json"  benchmark/mjx-comparison/bench_mjx_jacobian.py \
    --robot a1 --dtype fp64 --out "$RES/a1_mjx_fp64.json"
run "$RES/h1_mjx_fp32.json"  benchmark/mjx-comparison/bench_mjx_jacobian.py \
    --robot h1 --dtype fp32 --out "$RES/h1_mjx_fp32.json"
run "$RES/h1_mjx_fp64.json"  benchmark/mjx-comparison/bench_mjx_jacobian.py \
    --robot h1 --dtype fp64 --out "$RES/h1_mjx_fp64.json"

# ---- LP / sequential CusADi ABA Jacobians ----
run "$RES/a1_lp_fp64.json"   benchmark/mjx-comparison/bench_lp_jacobian.py \
    --robot a1 --method level_parallel --out "$RES/a1_lp_fp64.json"
run "$RES/h1_lp_fp64.json"   benchmark/mjx-comparison/bench_lp_jacobian.py \
    --robot h1 --method level_parallel --out "$RES/h1_lp_fp64.json"
run "$RES/a1_seq_fp64.json"  benchmark/mjx-comparison/bench_lp_jacobian.py \
    --robot a1 --method sequential    --out "$RES/a1_seq_fp64.json"
run "$RES/h1_seq_fp64.json"  benchmark/mjx-comparison/bench_lp_jacobian.py \
    --robot h1 --method sequential    --out "$RES/h1_seq_fp64.json"

# ---- Forward dynamics only ----
run "$RES/a1_fwd_lp.json"        benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend level_parallel --out "$RES/a1_fwd_lp.json"
run "$RES/a1_fwd_seq.json"       benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend sequential    --out "$RES/a1_fwd_seq.json"
run "$RES/h1_fwd_lp.json"        benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend level_parallel --out "$RES/h1_fwd_lp.json"
run "$RES/h1_fwd_seq.json"       benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend sequential    --out "$RES/h1_fwd_seq.json"
run "$RES/a1_fwd_mjx_fp32.json"  benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend mjx --dtype fp32 --out "$RES/a1_fwd_mjx_fp32.json"
run "$RES/a1_fwd_mjx_fp64.json"  benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend mjx --dtype fp64 --out "$RES/a1_fwd_mjx_fp64.json"
run "$RES/h1_fwd_mjx_fp32.json"  benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend mjx --dtype fp32 --out "$RES/h1_fwd_mjx_fp32.json"
run "$RES/h1_fwd_mjx_fp64.json"  benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend mjx --dtype fp64 --out "$RES/h1_fwd_mjx_fp64.json"
run "$RES/a1_fwd_mjw.json"       benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot a1 --backend mujoco_warp --out "$RES/a1_fwd_mjw.json"
run "$RES/h1_fwd_mjw.json"       benchmark/mjx-comparison/bench_forward_dynamics.py \
    --robot h1 --backend mujoco_warp --out "$RES/h1_fwd_mjw.json"

echo "[done] all benchmarks run"
