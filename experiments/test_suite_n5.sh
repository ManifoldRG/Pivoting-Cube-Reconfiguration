#!/bin/bash

# ============================================================================
# MSSA Test Suite: 5 Agents
# Systematic comparison of rewards, dimension reduction, and hyperparameters
# ============================================================================

# Base configuration (shared across all tests)
BASE_ARGS="
  --num_agents 5
  --max_steps 500
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
RESULTS_DIR="results/n5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "============================================================================"
echo "MSSA Test Suite: 5 Agents"
echo "Results will be saved to: $RESULTS_DIR"
echo "============================================================================"

# ============================================================================
# SECTION 1: REWARD COMPARISON (No Dimension Reduction)
# Test different reward functions to find the best baseline
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 1: Reward Function Comparison"
echo "========================================="

# Test 1.1: Soft Matching Reward (Doc.tex approach)
echo "Running Test 1.1: Soft Matching Reward..."
python train/train_sb3.py $BASE_ARGS \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --log_dir "$RESULTS_DIR/1.1_soft_matching"

# Test 1.2: Potential Reward (Baseline)
echo "Running Test 1.2: Potential Reward..."
python train/train_sb3.py $BASE_ARGS \
  --enable_potential_reward \
  --potential_scale 1.0 \
  --potential_normalize n2 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1.2_potential"

# Test 1.3: Bounty Reward
echo "Running Test 1.3: Bounty Reward..."
python train/train_sb3.py $BASE_ARGS \
  --enable_bounty_reward \
  --bounty_gamma 0.999 \
  --bounty_eta 2.0 \
  --bounty_base_value 1.0 \
  --bounty_total_frac_of_success 0.3 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1.3_bounty"

# Test 1.4: Soft Matching + Potential (Hybrid)
echo "Running Test 1.4: Soft Matching + Potential..."
python train/train_sb3.py $BASE_ARGS \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 50.0 \
  --enable_potential_reward \
  --potential_scale 0.5 \
  --potential_normalize n2 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1.4_soft_plus_potential"

# Test 1.5: All Rewards Combined
echo "Running Test 1.5: All Rewards Combined..."
python train/train_sb3.py $BASE_ARGS \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 50.0 \
  --enable_potential_reward \
  --potential_scale 0.5 \
  --enable_bounty_reward \
  --bounty_gamma 0.999 \
  --bounty_eta 1.5 \
  --bounty_total_frac_of_success 0.2 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/1.5_all_rewards"

# ============================================================================
# SECTION 2: DIMENSION REDUCTION COMPARISON
# Compare observation representations (with best reward from Section 1)
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 2: Dimension Reduction Comparison"
echo "========================================="

# Test 2.1: Full Matrix (Baseline)
echo "Running Test 2.1: Full Matrix..."
python train/train_sb3.py $BASE_ARGS \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/2.1_full_matrix"

# Test 2.2: Four-Band Reduction
echo "Running Test 2.2: Four-Band Reduction..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/2.2_four_band"

# Test 2.3: Four-Band + Local Neighborhood (k=3)
echo "Running Test 2.3: Four-Band + Local Neighborhood (k=3)..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --use_local_neighborhood \
  --local_neighborhood_k 3 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/2.3_four_band_local_k3"

# Test 2.4: Four-Band + Local Neighborhood (k=5) - Full coverage for n=5
echo "Running Test 2.4: Four-Band + Local Neighborhood (k=5)..."
python train/train_sb3.py $BASE_ARGS \
  --use_four_band_reduction \
  --use_local_neighborhood \
  --local_neighborhood_k 5 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/2.4_four_band_local_k5"

# ============================================================================
# SECTION 3: HYPERPARAMETER ABLATION
# Test sensitivity to key hyperparameters (using best config from above)
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 3: Hyperparameter Ablation"
echo "========================================="

# Test 3.1: Higher Learning Rate
echo "Running Test 3.1: Higher Learning Rate (1e-3)..."
python train/train_sb3.py \
  --num_agents 5 --max_steps 500 --episodes 2000 \
  --lr 1e-3 \
  --gamma 0.99 --lam 0.95 --clip 0.2 \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --hidden_dim 256 --entropy_coef 0.01 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3.1_lr_high"

# Test 3.2: Lower Learning Rate
echo "Running Test 3.2: Lower Learning Rate (1e-4)..."
python train/train_sb3.py \
  --num_agents 5 --max_steps 500 --episodes 2000 \
  --lr 1e-4 \
  --gamma 0.99 --lam 0.95 --clip 0.2 \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --hidden_dim 256 --entropy_coef 0.01 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3.2_lr_low"

# Test 3.3: Larger Network
echo "Running Test 3.3: Larger Network (512 hidden)..."
python train/train_sb3.py \
  --num_agents 5 --max_steps 500 --episodes 2000 \
  --lr 3e-4 --gamma 0.99 --lam 0.95 --clip 0.2 \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --hidden_dim 512 \
  --entropy_coef 0.01 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3.3_network_large"

# Test 3.4: Higher Entropy Coefficient (more exploration)
echo "Running Test 3.4: Higher Entropy (0.05)..."
python train/train_sb3.py $BASE_ARGS \
  --entropy_coef 0.05 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3.4_entropy_high"

# Test 3.5: Different Soft Matching Decay
echo "Running Test 3.5: Slower Soft Matching Decay (0.9995)..."
python train/train_sb3.py $BASE_ARGS \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.9995 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/3.5_decay_slow"

# ============================================================================
# SECTION 4: EXTENDED TRAINING
# Longer training for best configurations
# ============================================================================

echo ""
echo "========================================="
echo "SECTION 4: Extended Training"
echo "========================================="

# Test 4.1: Extended Soft Matching
echo "Running Test 4.1: Extended Soft Matching (5000 episodes)..."
python train/train_sb3.py \
  --num_agents 5 --max_steps 500 --episodes 5000 \
  --lr 3e-4 --gamma 0.99 --lam 0.95 --clip 0.2 \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --hidden_dim 256 --entropy_coef 0.01 \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/4.1_extended_soft"

# Test 4.2: Extended with Four-Band
echo "Running Test 4.2: Extended Four-Band (5000 episodes)..."
python train/train_sb3.py \
  --num_agents 5 --max_steps 500 --episodes 5000 \
  --lr 3e-4 --gamma 0.99 --lam 0.95 --clip 0.2 \
  --n_steps 2048 --batch_size 128 --epochs 10 \
  --hidden_dim 256 --entropy_coef 0.01 \
  --use_four_band_reduction \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --log_dir "$RESULTS_DIR/4.2_extended_four_band"

# ============================================================================
# Generate Summary Report
# ============================================================================

echo ""
echo "============================================================================"
echo "Test Suite Complete!"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To view results in TensorBoard:"
echo "  tensorboard --logdir=$RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Compare success rates across reward functions (Section 1)"
echo "2. Compare convergence speed with dimension reduction (Section 2)"
echo "3. Identify best hyperparameters (Section 3)"
echo "4. Use extended training results to validate findings (Section 4)"
echo "============================================================================"
