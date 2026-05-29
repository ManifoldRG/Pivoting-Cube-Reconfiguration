#!/bin/bash

# ============================================================================
# MSSA Quick Test Suite: 7 Agents
# Focus on dimension reduction - critical for scaling!
# ============================================================================

echo "============================================================================"
echo "MSSA Quick Test Suite: 7 Agents (Dimension Reduction Focus)"
echo "Testing if dimension reduction enables scaling to larger n"
echo "============================================================================"

# Base configuration
# Note: Increased max_steps for 7 agents (more complex coordination)
BASE_ARGS="
  --num_agents 7
  --max_steps 3000
  --episodes 1500
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
RESULTS_DIR="results/quick_n7_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# ============================================================================
# Test 1: Full Matrix Baseline (56 dims)
# This will likely be the hardest - 56 dims is a lot!
# ============================================================================
echo "Test 1/6: Full Matrix Baseline (56 dims - O(n²))..."
echo "  ⚠️  This may struggle due to high dimensionality!"
python train/train_sb3.py $BASE_ARGS \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1_full_matrix_56d"

# ============================================================================
# Test 2: Four-Band Reduction (35 dims)
# 37.5% reduction - should help significantly!
# ============================================================================
echo "Test 2/6: Four-Band Reduction (35 dims - O(n))..."
echo "  ✅ 37.5% dimension reduction"
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/2_four_band_35d"

# ============================================================================
# Test 3: Four-Band + Local k=3 (19 dims)
# 66% reduction - most aggressive!
# ============================================================================
echo "Test 3/6: Four-Band + Local k=3 (19 dims - O(k))..."
echo "  ✅ 66% dimension reduction - most aggressive!"
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --use_local_neighborhood \
  --local_neighborhood_k 3 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3_four_band_local_k3_19d"

# ============================================================================
# Test 4: Four-Band + Local k=5 (27 dims)
# 52% reduction - balanced approach
# ============================================================================
echo "Test 4/6: Four-Band + Local k=5 (27 dims - O(k))..."
echo "  ✅ 52% dimension reduction - balanced"
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --use_local_neighborhood \
  --local_neighborhood_k 5 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/4_four_band_local_k5_27d"

# ============================================================================
# Test 5: Four-Band + Larger Network (compensate for reduction)
# Test if larger network helps with reduced observations
# ============================================================================
echo "Test 5/6: Four-Band + Larger Network (512 hidden)..."
echo "  💡 Testing if larger network compensates for dimension reduction"
python train/train_sb3.py \
  --num_agents 7 --max_steps 1000 --episodes 1500 \
  --use_four_band_reduction \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --lr 3e-4 --gamma 0.99 --lam 0.95 --clip 0.2 \
  --hidden_dim 512 \
  --entropy_coef 0.01 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/5_four_band_large_net"

# ============================================================================
# Test 6: Potential Reward + Four-Band (alternative baseline)
# Test if simpler reward works better at larger scale
# ============================================================================
echo "Test 6/6: Potential Reward + Four-Band..."
echo "  🔬 Testing alternative reward at scale"
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --enable_potential_reward \
  --potential_scale 1.0 \
  --potential_normalize n2 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/6_potential_four_band"

# ============================================================================
# Analysis
# ============================================================================
echo ""
echo "============================================================================"
echo "Quick Test Suite (n=7) Complete!"
echo "============================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "🔍 KEY QUESTION: Does dimension reduction enable scaling?"
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
echo "Expected outcomes:"
echo "  📊 Full matrix (56d):      May struggle (too many dims)"
echo "  📊 Four-band (35d):        Should improve significantly"
echo "  📊 Four-band + local (19d): Best scalability, possible perf trade-off"
echo ""
echo "============================================================================"
