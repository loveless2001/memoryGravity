#!/bin/bash
# Poison Sweep Experiment (following docs/plan.md)
# Sweeps poison_N = [32, 64, 128, 256] (baseline and N=16 already complete)
# With unbuffered output for real-time logging

set -e

VENV_PYTHON="/home/lenovo/projects/memoryGravity/.venv/bin/python"
SCRIPT="train/tinystories_gpt.py"
MAX_STEPS=5000
BATCH_SIZE=8
LR=3e-4

echo "=== Memory Gravity Poison Sweep (Resumed) ==="
echo "Model: TinyStories GPT (~42M params)"
echo "Sweep: poison_N = [32, 64, 128, 256]"
echo "Steps: $MAX_STEPS per experiment"
echo "Started at: $(date)"
echo ""

# Create results directory
mkdir -p results

# Poison sweep - starting from N=32
for N in 256 128 64 32; do
    echo ""
    echo "=========================================="
    echo "[poison_N=$N] Starting experiment at $(date)"
    echo "=========================================="
    
    # Use python -u for unbuffered output + stdbuf for tee
    stdbuf -oL $VENV_PYTHON -u $SCRIPT \
        --poison_n $N \
        --max_steps $MAX_STEPS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        2>&1 | stdbuf -oL tee results/poison_n${N}.log
    
    echo ""
    echo "[poison_N=$N] Completed at $(date)"
    echo "=========================================="
done

echo ""
echo "=== Sweep Complete at $(date) ==="
echo "Results saved to results/"
echo "Checkpoints saved to checkpoints/"
