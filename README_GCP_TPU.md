# MSSA JAX/TPU Training Guide

This guide explains how to train the pivoting cube reconfiguration model on GCP TPU VMs using either CPU-accelerated PyTorch/SB3 or JAX for true TPU acceleration.

## Quick Start

### Option 1: CPU Training on TPU VM (Fastest Setup)

Use this to start training immediately with your existing PyTorch/SB3 code:

```bash
# 1. Create TPU VM (one-time)
export GCP_PROJECT_ID="your-project-id"
export GCP_ZONE="us-central1-a"

gcloud compute tpus tpu-vm create mssa-training \
  --zone=$GCP_ZONE \
  --project=$GCP_PROJECT_ID \
  --accelerator-type=v3-8 \
  --version=tpu-ubuntu2204-base

# 2. Setup environment
bash scripts/setup_gcp.sh --project $GCP_PROJECT_ID --repo https://github.com/YOUR_USER/MSSA.git

# 3. SSH and run training
gcloud compute tpus tpu-vm ssh mssa-training --zone=$GCP_ZONE
source ~/mssa-env/bin/activate
cd ~/MSSA
bash scripts/run_cpu_training.sh 8  # Train stage n=8
```

### Option 2: JAX Training on TPU (Maximum Performance)

Use this for 50-100x speedup with true TPU acceleration:

```bash
# 1. SSH into TPU VM
gcloud compute tpus tpu-vm ssh mssa-training --zone=us-central1-a

# 2. Install JAX dependencies
source ~/mssa-env/bin/activate
pip install -r requirements_jax.txt

# 3. Run JAX training
python train_jax/train_curriculum.py --target_n 50 --n_envs 512
```

## VM Specifications

### TPU v3-8 VM (Recommended)
- **96 vCPUs** - Excellent for parallel environments
- **335 GB RAM** - More than enough for large batches
- **TPU v3-8** - 8 TPU cores for JAX training
- **Free** with TPU quota!

```bash
gcloud compute tpus tpu-vm create mssa-training \
  --zone=us-central1-a \
  --accelerator-type=v3-8 \
  --version=tpu-ubuntu2204-base
```

### TPU v2-8 VM (Budget Option)
- **96 vCPUs**
- **335 GB RAM**
- **TPU v2-8** - Slightly slower but still fast

## Directory Structure

```
MSSA/
├── scripts/
│   ├── setup_gcp.sh        # GCP VM setup script
│   └── run_cpu_training.sh # CPU training launcher
├── jax_env/
│   ├── ogm_jax.py          # JAX environment implementation
│   └── __init__.py
├── train_jax/
│   ├── ppo_jax.py          # JAX PPO implementation
│   ├── train_curriculum.py # JAX curriculum training
│   └── __init__.py
├── requirements_cpu.txt    # CPU training dependencies
└── requirements_jax.txt    # JAX/TPU dependencies
```

## Training Modes

### CPU Training (PyTorch/SB3)

Uses your existing `train/train_sb3.py` with high parallelism:

```bash
# Single stage
bash scripts/run_cpu_training.sh 8

# Full curriculum (manual)
bash scripts/run_cpu_training.sh 4
bash scripts/run_cpu_training.sh 5
# ... continue with higher n values
```

**Expected speedup**: 5-10x over laptop (more CPU cores)

### JAX Training (TPU)

Uses the new JAX implementation for true TPU acceleration:

```bash
# Full curriculum to n=50
python train_jax/train_curriculum.py --target_n 50 --n_envs 512

# Resume from checkpoint
python train_jax/train_curriculum.py --target_n 50 --load_checkpoint runs/jax_curriculum/stage_n8/checkpoint.pkl --start_stage 4
```

**Expected speedup**: 50-100x over laptop

## Curriculum Stages

| Stage | n | Target Success | Max Steps | LR | Entropy |
|-------|---|----------------|-----------|----:|--------:|
| 1 | 4 | 90% | 600 | 5e-4 | 0.03 |
| 2 | 5 | 88% | 750 | 4.5e-4 | 0.03 |
| 3 | 6 | 85% | 900 | 4e-4 | 0.03 |
| 4 | 7 | 80% | 1200 | 3.5e-4 | 0.03 |
| 5 | 8 | 65% | 2000 | 1e-4 | 0.05 |
| 6 | 10 | 60% | 2500 | 8e-5 | 0.04 |
| 7 | 12 | 55% | 3000 | 6e-5 | 0.03 |
| 8 | 15 | 50% | 4000 | 5e-5 | 0.025 |
| 9 | 20 | 45% | 5000 | 4e-5 | 0.02 |
| 10 | 30 | 40% | 7500 | 3e-5 | 0.015 |
| 11 | 50 | 35% | 12000 | 2e-5 | 0.01 |

## Monitoring Training

### TensorBoard
```bash
# On your local machine
gcloud compute tpus tpu-vm ssh mssa-training --zone=us-central1-a -- -L 6006:localhost:6006

# On the VM
tensorboard --logdir runs/
```

### Logs
```bash
tail -f runs/gcp_curriculum/stage_n8/training_sb3.log
```

## Troubleshooting

### "RESOURCE_EXHAUSTED" Error
Reduce `--n_envs` or use a smaller batch size.

### Slow Training
1. Increase `--n_envs` (TPU VMs have lots of RAM)
2. Use JAX instead of PyTorch for true TPU acceleration
3. Check that you're using the TPU VM's CPUs, not SSH tunneling from your laptop

### JAX Not Using TPU
```python
import jax
print(jax.devices())  # Should show TPU devices
```

If showing CPU, reinstall JAX with TPU support:
```bash
pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Cleanup

To avoid charges, delete the TPU VM when done:

```bash
gcloud compute tpus tpu-vm delete mssa-training --zone=us-central1-a
```
