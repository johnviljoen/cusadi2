#!/bin/bash
# Detached tuning sweep: bash run_sweep.sh <grid> <seeds...>   (log: logs/sweep_grid<grid>.log; resumes over existing results)
cd "$(dirname "$0")"
GRID=$1; shift
nohup setsid "$HOME/miniconda3/envs/cusadi2/bin/python" sweep.py run --grid "$GRID" --seeds "$@" > "logs/sweep_grid$GRID.log" 2>&1 < /dev/null &
echo "launched grid $GRID seeds $* pid $!"
