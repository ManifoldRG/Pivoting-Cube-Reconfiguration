#!/bin/bash
# JAX CPU Smoke Test
# Quick local test of the JAX training pipeline (no TPU needed).
# Runs n=4 with only 4 parallel envs for ~1-2 minutes.
#
# Usage: bash scripts/run_jax_cpu_test.sh

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "  MSSA JAX CPU Smoke Test"
echo "=========================================="

# Activate environment
echo "--- Activating environment ---"
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate mssa
source env.sh

# Force CPU backend
export JAX_PLATFORMS=cpu

# Run unit tests first
echo ""
echo "--- Running JAX unit tests ---"
python -m pytest tests/test_jax_vs_numpy.py -v --tb=short 2>&1 | tail -30
echo ""

# Run a quick curriculum training (n=4 only, 4 envs)
echo "--- Running JAX curriculum training (n=4, 4 envs) ---"
python train_jax/train_curriculum.py \
    --target_n 4 \
    --n_envs 4 \
    --local_k 7

echo ""
echo "=========================================="
echo "  Smoke test complete!"
echo "=========================================="
