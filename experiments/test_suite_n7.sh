#!/bin/bash

# ============================================================================
# MSSA Test Suite: 7 Agents
# Comprehensive testing for scaling with dimension reduction
# ============================================================================

# Base configuration
BASE_ARGS="
  --num_agents 7
  --max_steps 1000
  --episodes 2000
  --n_steps 2048
  --batch_size 128
  --epochs 10
  --lr 3e-4
  --gamma 0.99
  --lam 0.95
  --clip 0.2
  --hidden_dim 256
  --entropy_coef 0.01
  --value_coef 0.5
  --grad_clip 0.5
"

# Create results directory
RESULTS_DIR="results/n7_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "============================================================================"
echo "MSSA Test Suite: 7 Agents"
echo "Results will be saved to: $RESULTS_DIR"
echo "============================================================================"

# ============================================================================
# SECTION 1: DIMENSION REDUCTION SCALING TEST
# Critical for n=7: Does reduction enable learning?
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 1: Dimension Reduction Scaling"
echo "========================================="

# Test 1.1: Full Matrix (56 dims) - Baseline but may struggle
echo "Running Test 1.1: Full Matrix (56 dims)..."
python train/train_sb3.py $BASE_ARGS \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1.1_full_matrix"

# Test 1.2: Four-Band (35 dims) - Primary approach
echo "Running Test 1.2: Four-Band Reduction (35 dims)..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1.2_four_band"

# Test 1.3: Four-Band + Local k=3 (19 dims) - Aggressive
echo "Running Test 1.3: Four-Band + Local k=3 (19 dims)..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --use_local_neighborhood \
  --local_neighborhood_k 3 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1.3_four_band_local_k3"

# Test 1.4: Four-Band + Local k=5 (27 dims) - Balanced
echo "Running Test 1.4: Four-Band + Local k=5 (27 dims)..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --use_local_neighborhood \
  --local_neighborhood_k 5 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1.4_four_band_local_k5"

# Test 1.5: Four-Band + Local k=7 (35 dims) - Full coverage
echo "Running Test 1.5: Four-Band + Local k=7 (35 dims - full coverage)..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --use_local_neighborhood \
  --local_neighborhood_k 7 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1.5_four_band_local_k7"

# ============================================================================
# SECTION 2: NETWORK CAPACITY FOR REDUCED OBSERVATIONS
# Can larger networks compensate for dimension reduction?
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 2: Network Capacity Scaling"
echo "========================================="

# Test 2.1: Four-Band + Small Network (128)
echo "Running Test 2.1: Four-Band + Small Network (128)..."
python train/train_sb3.py \
  --num_agents 7 --max_steps 1000 --episodes 2000 \
  --use_four_band_reduction \
  --lr 3e-4 --gamma 0.99 --lam 0.95 --clip 0.2 \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --hidden_dim 128 \
  --entropy_coef 0.01 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/2.1_four_band_net128"

# Test 2.2: Four-Band + Medium Network (256) - Baseline
echo "Running Test 2.2: Four-Band + Medium Network (256)..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/2.2_four_band_net256"

# Test 2.3: Four-Band + Large Network (512)
echo "Running Test 2.3: Four-Band + Large Network (512)..."
python train/train_sb3.py \
  --num_agents 7 --max_steps 1000 --episodes 2000 \
  --use_four_band_reduction \
  --lr 3e-4 --gamma 0.99 --lam 0.95 --clip 0.2 \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --hidden_dim 512 \
  --entropy_coef 0.01 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/2.3_four_band_net512"

# ============================================================================
# SECTION 3: REWARD COMPARISON AT SCALE
# Do different rewards work better at n=7?
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 3: Reward Functions at Scale"
echo "========================================="

# Test 3.1: Soft Matching + Four-Band
echo "Running Test 3.1: Soft Matching + Four-Band..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3.1_soft_four_band"

# Test 3.2: Potential + Four-Band
echo "Running Test 3.2: Potential + Four-Band..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --enable_potential_reward \
  --potential_scale 1.0 \
  --potential_normalize n2 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3.2_potential_four_band"

# Test 3.3: Hybrid + Four-Band
echo "Running Test 3.3: Hybrid (Soft + Potential) + Four-Band..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 50.0 \
  --enable_potential_reward \
  --potential_scale 0.5 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3.3_hybrid_four_band"

# ============================================================================
# SECTION 4: EXTENDED TRAINING
# Validate convergence at larger scale
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 4: Extended Training"
echo "========================================="

# Test 4.1: Best Config Extended (4000 episodes)
echo "Running Test 4.1: Four-Band Extended Training (4000 episodes)..."
python train/train_sb3.py \
  --num_agents 7 --max_steps 1000 --episodes 4000 \
  --use_four_band_reduction \
  --lr 3e-4 --gamma 0.99 --lam 0.95 --clip 0.2 \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --hidden_dim 256 --entropy_coef 0.01 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/4.1_four_band_extended"

# Test 4.2: Four-Band + Local k=3 Extended
echo "Running Test 4.2: Four-Band + Local k=3 Extended (4000 episodes)..."
python train/train_sb3.py \
  --num_agents 7 --max_steps 1000 --episodes 4000 \
  --use_four_band_reduction \
  --use_local_neighborhood --local_neighborhood_k 3 \
  --lr 3e-4 --gamma 0.99 --lam 0.95 --clip 0.2 \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --hidden_dim 256 --entropy_coef 0.01 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/4.2_local_k3_extended"

# ============================================================================
# Generate Summary Report
# ============================================================================

echo ""
echo "============================================================================"
echo "Test Suite (n=7) Complete!"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To view results in TensorBoard:"
echo "  tensorboard --logdir=$RESULTS_DIR"
echo ""
echo "To analyze results:"
echo "  python experiments/analyze_results.py $RESULTS_DIR"
echo ""
echo "🎯 KEY INSIGHTS TO LOOK FOR:"
echo "  1. Does four-band reduction enable learning at n=7?"
echo "  2. What's the performance gap: full vs four-band vs local?"
echo "  3. Does network size help compensate for dimension reduction?"
echo "  4. Can we achieve >70% success rate with 66% dimension reduction?"
echo ""
echo "Expected dimension reduction benefits:"
echo "  Full (56d) → Four-band (35d):  37.5% reduction"
echo "  Four-band → Local k=3 (19d):   66% reduction from full"
echo "  Four-band → Local k=5 (27d):   52% reduction from full"
echo "============================================================================"
