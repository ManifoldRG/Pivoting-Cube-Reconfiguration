#!/bin/bash
# CPU Training Script for TPU VM
# Runs PyTorch/SB3 training with high CPU parallelism
# Usage: bash scripts/run_cpu_training.sh [stage]

set -e

# ============================================
# Configuration
# ============================================
STAGE="${1:-8}"  # Default to n=8

# Detect number of CPUs and set parallel envs accordingly
NUM_CPUS=$(nproc 2>/dev/null || echo 8)
NUM_ENVS=$((NUM_CPUS / 2))  # Use half of CPUs for envs
NUM_ENVS=$((NUM_ENVS > 64 ? 64 : NUM_ENVS))  # Cap at 64

echo "=========================================="
echo "  MSSA CPU Training - Stage n=$STAGE"
echo "=========================================="
echo "CPUs detected: $NUM_CPUS"
echo "Parallel environments: $NUM_ENVS"
echo ""

# ============================================
# Stage-specific hyperparameters
# Note: All stages use 85% rolling success rate as advancement threshold
# ============================================

# Rolling success window size (last N episodes)
SUCCESS_WINDOW_SIZE=100

# Target rolling success rate for advancement (85% for all stages)
TARGET_ROLLING_SUCCESS="0.85"

case $STAGE in
    4)
        MAX_STEPS=600
        EPISODES=2000
        LR="5e-4"
        ENTROPY="0.03"
        MIN_EPISODES=500
        ;;
    5)
        MAX_STEPS=750
        EPISODES=2500
        LR="4.5e-4"
        ENTROPY="0.03"
        MIN_EPISODES=500
        ;;
    6)
        MAX_STEPS=900
        EPISODES=2500
        LR="4e-4"
        ENTROPY="0.03"
        MIN_EPISODES=600
        ;;
    7)
        MAX_STEPS=1200
        EPISODES=3000
        LR="3.5e-4"
        ENTROPY="0.03"
        MIN_EPISODES=600
        ;;
    8)
        MAX_STEPS=2000
        EPISODES=5000
        LR="1e-4"
        ENTROPY="0.05"
        MIN_EPISODES=1500
        ;;
    10)
        MAX_STEPS=2500
        EPISODES=6000
        LR="8e-5"
        ENTROPY="0.04"
        MIN_EPISODES=2000
        ;;
    12)
        MAX_STEPS=3000
        EPISODES=7000
        LR="6e-5"
        ENTROPY="0.03"
        MIN_EPISODES=2500
        ;;
    15)
        MAX_STEPS=4000
        EPISODES=8000
        LR="5e-5"
        ENTROPY="0.025"
        MIN_EPISODES=3000
        ;;
    20)
        MAX_STEPS=5000
        EPISODES=10000
        LR="4e-5"
        ENTROPY="0.02"
        MIN_EPISODES=4000
        ;;
    30)
        MAX_STEPS=7500
        EPISODES=12000
        LR="3e-5"
        ENTROPY="0.015"
        MIN_EPISODES=5000
        ;;
    50)
        MAX_STEPS=12000
        EPISODES=15000
        LR="2e-5"
        ENTROPY="0.01"
        MIN_EPISODES=6000
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Supported stages: 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50"
        exit 1
        ;;
esac

echo "Target rolling success (last $SUCCESS_WINDOW_SIZE episodes): ${TARGET_ROLLING_SUCCESS}"

# ============================================
# Find previous model for curriculum
# ============================================
LOAD_MODEL=""
PREV_STAGE=""

# Map to previous stage
declare -A PREV_STAGES=(
    [5]=4 [6]=5 [7]=6 [8]=7 [10]=8 [12]=10 [15]=12 [20]=15 [30]=20 [50]=30
)

if [[ -v "PREV_STAGES[$STAGE]" ]]; then
    PREV_STAGE="${PREV_STAGES[$STAGE]}"
    POTENTIAL_MODELS=$(find runs -name "final_model.zip" -path "*n${PREV_STAGE}*" 2>/dev/null | sort -r | head -1)
    
    if [ -n "$POTENTIAL_MODELS" ]; then
        LOAD_MODEL="$POTENTIAL_MODELS"
        echo "Loading previous model from: $LOAD_MODEL"
    else
        echo "No previous model found for n=$PREV_STAGE. Training from scratch."
    fi
fi

# ============================================
# Build command
# ============================================
LOG_DIR="runs/gcp_curriculum/stage_n${STAGE}"

CMD="python train/train_sb3.py \
    --num_agents $STAGE \
    --max_steps $MAX_STEPS \
    --episodes $EPISODES \
    --lr $LR \
    --entropy_coef $ENTROPY \
    --use_unlabeled_mode \
    --enable_potential_reward \
    --use_four_band_reduction \
    --use_local_neighborhood \
    --local_neighborhood_k 7 \
    --use_vec_normalize \
    --use_separate_networks \
    --hidden_dim 512 \
    --target_success_rate $TARGET_ROLLING_SUCCESS \
    --min_episodes $MIN_EPISODES \
    --success_window_size $SUCCESS_WINDOW_SIZE \
    --checkpoint_interval 500 \
    --num_envs $NUM_ENVS \
    --log_dir $LOG_DIR"

if [ -n "$LOAD_MODEL" ]; then
    CMD="$CMD --load_model $LOAD_MODEL --reset_timesteps"
fi

echo ""
echo "Training command:"
echo "$CMD"
echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="

# Run training
eval $CMD

echo ""
echo "=========================================="
echo "  Training complete!"
echo "=========================================="
echo "Logs saved to: $LOG_DIR"
