#!/bin/bash
# Phase 1 of the MJPC-faithful example: gates, then the seed-1 closed loop with video, then seeds 2-5 without video.
cd "$(dirname "$0")"
PY=$HOME/miniconda3/envs/cusadi2/bin/python
$PY verify_ps.py --n-iter 4 > logs/verify_ps.log 2>&1 || { echo "GATES FAILED"; exit 1; }
$PY closed_loop.py --controller ps --seed 1 > logs/closed_loop_ps.log 2>&1
for s in 2 3 4 5; do
  $PY closed_loop.py --controller ps --seed $s --no-video --tag _seed$s > logs/closed_loop_ps_seed$s.log 2>&1
done
echo "phase 1 runs done"
