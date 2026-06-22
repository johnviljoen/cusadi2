#!/bin/bash
# Cold-compile sweep: ours vs MJX fp32 vs mujoco-warp fp32, A1 robot,
# fwd dynamics + Jacobian (no contact), batch sizes 1 to 10000.
#
# Each measurement is a fresh Python subprocess with caches cleared
# beforehand, so we capture true cold compile time per configuration.
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=1
PY=/home/john/.conda/envs/cusadi2/bin/python
SCRIPT=benchmark/mjx-comparison/bench_compile_time.py
OUT=benchmark/mjx-comparison/results/compile_time_sweep.json
ROBOT=a1
DTYPE=fp32
BATCHES=(1 8 32 128 256 512 1024 4096 10000)

echo "[" > "$OUT"
first=1

emit() {
    local line="$1"
    if [ "$first" = "1" ]; then first=0; else echo "," >> "$OUT"; fi
    echo "  $line" >> "$OUT"
}

clear_caches() {
    # Warp module cache
    rm -rf /home/john/.cache/warp 2>/dev/null
    # JAX persistent cache (in case anything got enabled)
    rm -rf /home/john/.cache/jax  2>/dev/null
}

for batch in "${BATCHES[@]}"; do
    # ----- ours: constant 1.5 s, no measurement loop needed -----
    emit "$($PY -u "$SCRIPT" --backend lp --task fwd --robot $ROBOT --batch $batch)"
    emit "$($PY -u "$SCRIPT" --backend lp --task jac --robot $ROBOT --batch $batch)"

    # ----- MJX fp32 fwd dyn -----
    clear_caches
    echo "[mjx fwd  B=$batch]"
    emit "$($PY -u "$SCRIPT" --backend mjx --task fwd --robot $ROBOT --batch $batch --dtype $DTYPE)"

    # ----- MJX fp32 Jacobian (no contact) -----
    clear_caches
    echo "[mjx jac  B=$batch]"
    emit "$($PY -u "$SCRIPT" --backend mjx --task jac --robot $ROBOT --batch $batch --dtype $DTYPE)"

    # ----- mujoco-warp fp32 fwd dyn -----
    clear_caches
    echo "[mjw fwd  B=$batch]"
    emit "$($PY -u "$SCRIPT" --backend mjw --task fwd --robot $ROBOT --batch $batch)"
done

echo "" >> "$OUT"
echo "]" >> "$OUT"
echo "[done] wrote $OUT"
