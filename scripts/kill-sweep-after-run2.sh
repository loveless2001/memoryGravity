#!/usr/bin/env bash
# Watch for Run 2 results, then kill the sweep process
RESULTS="/home/lenovo/projects/memoryGravity/results/glyph_memory_arena/v2_optim_sweep/run2_lambda2.0.json"
PID=840866

while true; do
  if [ -f "$RESULTS" ]; then
    echo "Run 2 results detected. Killing sweep process $PID..."
    kill $PID 2>/dev/null
    echo "Done. Sweep paused after Run 2."
    exit 0
  fi
  # Check if process still alive
  if ! kill -0 $PID 2>/dev/null; then
    echo "Sweep process $PID already dead."
    exit 1
  fi
  sleep 30
done
