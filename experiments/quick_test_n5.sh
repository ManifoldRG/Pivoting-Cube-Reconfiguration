#!/bin/bash

# ============================================================================
# MSSA Quick Test Suite: 5 Agents
# Minimal set of tests for rapid validation (~2-3 hours)
# ============================================================================

echo "============================================================================"
echo "MSSA Quick Test Suite: 5 Agents (Minimal)"
echo "This runs only the most important tests for rapid validation"
echo "============================================================================"

# Base configuration
BASE_ARGS="
  --num_agents 5
  --max_steps 500
  --episodes 1000
  --n_steps 2048
  --batch_size 128
  --epochs 10
  --lr 3e-4
  --gamma 0.99
  --lam 0.95
  --clip 0.2
  --hidden_dim 256
  --entropy_coef 0.01
"

# Create results directory
RESULTS_DIR="results/quick_n5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# ============================================================================
# Test 1: Soft Matching Reward (Doc.tex baseline)
# ============================================================================
echo "Test 1/5: Soft Matching Reward (doc.tex baseline)..."
python train/train_sb3.py $BASE_ARGS \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1_soft_matching"

# ============================================================================
# Test 2: Potential Reward (Alternative)
# ============================================================================
echo "Test 2/5: Potential Reward (alternative baseline)..."
python train/train_sb3.py $BASE_ARGS \
  --enable_potential_reward \
  --potential_scale 1.0 \
  --potential_normalize n2 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/2_potential"

# ============================================================================
# Test 3: Soft Matching + Four-Band Reduction
# ============================================================================
echo "Test 3/5: Soft Matching with Four-Band Reduction..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3_soft_four_band"

# ============================================================================
# Test 4: Soft Matching + Four-Band + Local (k=3)
# ============================================================================
echo "Test 4/5: Soft Matching with Four-Band + Local Neighborhood..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --use_local_neighborhood \
  --local_neighborhood_k 3 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/4_soft_four_band_local"

# ============================================================================
# Test 5: Hybrid Reward (Soft + Potential)
# ============================================================================
echo "Test 5/5: Hybrid Reward (Soft Matching + Potential)..."
python train/train_sb3.py $BASE_ARGS \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 50.0 \
  --enable_potential_reward \
  --potential_scale 0.5 \
  --potential_normalize n2 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/5_hybrid"

# ============================================================================
# Analysis
# ============================================================================
echo ""
echo "============================================================================"
echo "Quick Test Suite Complete!"
echo "============================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To analyze results:"
echo "  python experiments/analyze_results.py $RESULTS_DIR"
echo ""
echo "To view in TensorBoard:"
echo "  tensorboard --logdir=$RESULTS_DIR"
echo ""
echo "Quick comparison:"
grep -r "Success rate:" "$RESULTS_DIR"/*/training_sb3.log | sed 's|'$RESULTS_DIR'/||' | sed 's|/training_sb3.log:||'
echo ""
echo "============================================================================"
