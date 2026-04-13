#!/bin/bash
# run_experiments.sh
# ------------------
# Runs all 4 model variants in sequence on the Azure VM.
# Start this script and let it run overnight — it will save checkpoints
# after each epoch so you won't lose progress if anything interrupts.
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# Estimated time on T4 GPU: ~24-30 hours for all 4 variants

set -e  # stop immediately if any command fails

echo "======================================"
echo " SBERT Experiment Runner"
echo " Started: $(date)"
echo "======================================"

# ── Variant 1: Baseline (mean pooling, sequential NLI training) ────────────
echo ""
echo "[1/4] Training baseline: mean pooling, sequential NLI..."
python training/train.py \
    --config configs/config.yaml \
    --pooling mean \
    --run_name baseline_mean

# ── Variant 2: Enhancement 1 (joint multi-task, best lambda) ──────────────
# Run a quick 1-epoch sweep to find the best lambda first
echo ""
echo "[2/4] Lambda sweep (1 epoch each)..."
for lam in 0.1 0.3 0.5 0.7 0.9; do
    echo "  Testing lambda=$lam ..."
    python training/train.py \
        --config configs/config.yaml \
        --pooling mean \
        --multitask \
        --lambda_weight $lam \
        --run_name "lambda_sweep_${lam}"
done

# After the sweep, check W&B to find the best lambda, then do a full run.
# Replace 0.5 below with your best lambda from the sweep.
BEST_LAMBDA=0.5
echo ""
echo "  Full multi-task run with lambda=$BEST_LAMBDA..."
python training/train.py \
    --config configs/config.yaml \
    --pooling mean \
    --multitask \
    --lambda_weight $BEST_LAMBDA \
    --run_name "multitask_lam${BEST_LAMBDA}"

# ── Variant 3: Enhancement 2 (weighted pooling, sequential) ───────────────
echo ""
echo "[3/4] Training with learned weighted pooling..."
python training/train.py \
    --config configs/config.yaml \
    --pooling weighted \
    --run_name weighted_pooling

# ── Variant 4: Both enhancements ──────────────────────────────────────────
echo ""
echo "[4/4] Training with BOTH enhancements..."
python training/train.py \
    --config configs/config.yaml \
    --pooling weighted \
    --multitask \
    --lambda_weight $BEST_LAMBDA \
    --run_name "both_enhancements"

# ── Evaluate all saved models ─────────────────────────────────────────────
echo ""
echo "======================================"
echo " Evaluating all models..."
echo "======================================"

python evaluation/evaluate.py \
    --model_path experiments/baseline_mean_best.pt \
    --pooling mean

python evaluation/evaluate.py \
    --model_path experiments/multitask_lam${BEST_LAMBDA}_best.pt \
    --pooling mean

python evaluation/evaluate.py \
    --model_path experiments/weighted_pooling_best.pt \
    --pooling weighted \
    --analyze_weights   # print token weights for the demo!

python evaluation/evaluate.py \
    --model_path experiments/both_enhancements_best.pt \
    --pooling weighted

echo ""
echo "======================================"
echo " All done! Finished: $(date)"
echo "======================================"
