#!/usr/bin/env bash
# Phase A: Optimization Sweep for v2 Activation Reliability
# Fixed: mg_query_gated_local d32_h2, seq=512, window=64, bind=8, delay=128
# Variable: lambda_mass, curriculum, warmup, training length
# 3 seeds per group, screen on bind=8 delay=128

set -euo pipefail
cd "$(dirname "$0")/.."

# Activate project venv
source .venv/bin/activate

COMMON="--models mg_query_gated_local \
  --capacity-specs 32x2 \
  --seq-len 512 \
  --local-window 64 \
  --binding-zone-ratio 0.25 \
  --lr 1e-3 \
  --seeds 1,2,3 \
  --train-n-bindings 8 \
  --train-query-delays 128 \
  --train-distractor-rates 0.3 \
  --train-value-collisions false \
  --eval-n-bindings 8,16 \
  --eval-query-delays 128,256 \
  --eval-distractor-rates 0.3 \
  --eval-value-collisions false \
  --batch-size 64 \
  --log-interval 500 \
  --amp bf16"

OUTDIR="results/glyph_memory_arena/v2_optim_sweep"

echo "=========================================="
echo "Phase A: Optimization Sweep"
echo "=========================================="

# Run 2: lambda_mass=2.0
echo ""
echo "[Run 2] lambda_mass=2.0, 10k steps"
python -m train.run_glyph_memory_arena $COMMON \
  --lambda-mass 2.0 \
  --max-steps 10000 \
  --json-out "$OUTDIR/run2_lambda2.0.json" \
  --csv-out "$OUTDIR/run2_lambda2.0.csv"
echo "[Run 2] DONE"

# Run 3: lambda_mass=5.0
echo ""
echo "[Run 3] lambda_mass=5.0, 10k steps"
python -m train.run_glyph_memory_arena $COMMON \
  --lambda-mass 5.0 \
  --max-steps 10000 \
  --json-out "$OUTDIR/run3_lambda5.0.json" \
  --csv-out "$OUTDIR/run3_lambda5.0.csv"
echo "[Run 3] DONE"

# Run 4: Curriculum (delay 64 for first 5k steps, then 128)
echo ""
echo "[Run 4] Curriculum: delay 64→128 at step 5000, 10k steps"
python -m train.run_glyph_memory_arena $COMMON \
  --lambda-mass 1.0 \
  --max-steps 10000 \
  --curriculum-threshold 5000 \
  --curriculum-train-query-delays 64 \
  --json-out "$OUTDIR/run4_curriculum.json" \
  --csv-out "$OUTDIR/run4_curriculum.csv"
echo "[Run 4] DONE"

# Run 5: LR warmup=500
echo ""
echo "[Run 5] LR warmup=500, 10k steps"
python -m train.run_glyph_memory_arena $COMMON \
  --lambda-mass 1.0 \
  --max-steps 10000 \
  --warmup-steps 500 \
  --json-out "$OUTDIR/run5_warmup500.json" \
  --csv-out "$OUTDIR/run5_warmup500.csv"
echo "[Run 5] DONE"

# Run 6: Longer training, 20k steps
echo ""
echo "[Run 6] Longer training: 20k steps"
python -m train.run_glyph_memory_arena $COMMON \
  --lambda-mass 1.0 \
  --max-steps 20000 \
  --json-out "$OUTDIR/run6_20k.json" \
  --csv-out "$OUTDIR/run6_20k.csv"
echo "[Run 6] DONE"

echo ""
echo "=========================================="
echo "Phase A COMPLETE — all results in $OUTDIR/"
echo "=========================================="
