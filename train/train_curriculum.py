"""
Curriculum learning for scaling unlabeled cube reconfiguration to n=50.

This script implements an automated curriculum training pipeline that:
1. Trains progressively on larger n values (8 → 12 → 18 → 25 → 35 → 50)
2. Uses four-band + local neighborhood reduction for constant observation size
3. Transfers model weights between stages for efficient learning
4. Auto-advances when success rate exceeds threshold

Usage:
    # Train full curriculum to n=50
    python train/train_curriculum.py --target_n 50 --log_dir runs/curriculum

    # Start from a specific stage
    python train/train_curriculum.py --target_n 50 --start_stage 2

    # Resume from a saved model
    python train/train_curriculum.py --target_n 50 --load_model runs/curriculum/stage_n12/final_model.zip
"""

import argparse
import os
import subprocess
import glob
from datetime import datetime


# Curriculum stages: progressively increase n with appropriate hyperparameters
# Each stage fine-tunes from the previous one
#
# Design principles (v2 - improved based on training observations):
# 1. Smooth progression: n=4→5→6→7→8→10→12→15 (added intermediate stages)
# 2. More steps per agent: ~150-200 instead of ~120-125
# 3. Higher entropy for larger n: prevents getting stuck in local optima
# 4. Realistic target success rates: gradual decrease as n increases
#
CURRICULUM_STAGES = [
    # ========== Small-scale stages (n=4 to n=7) ==========
    # These build foundational skills with smooth progression
    {
        "n": 4,
        "min_episodes": 500,
        "max_episodes": 2000,
        "target_success": 0.90,
        "lr": 5e-4,
        "entropy": 0.03,
        "max_steps": 600,  # 150 steps per agent
    },
    {
        "n": 5,
        "min_episodes": 500,
        "max_episodes": 2500,
        "target_success": 0.88,
        "lr": 4.5e-4,
        "entropy": 0.03,
        "max_steps": 750,  # 150 steps per agent
    },
    {
        "n": 6,
        "min_episodes": 600,
        "max_episodes": 2500,
        "target_success": 0.85,
        "lr": 4e-4,
        "entropy": 0.03,
        "max_steps": 900,  # 150 steps per agent
    },
    {
        "n": 7,
        "min_episodes": 600,
        "max_episodes": 3000,
        "target_success": 0.80,
        "lr": 3.5e-4,
        "entropy": 0.03,
        "max_steps": 1050,  # 150 steps per agent
    },
    # ========== Medium-scale stages (n=8 to n=15) ==========
    # Higher entropy to encourage exploration in larger state space
    {
        "n": 8,
        "min_episodes": 750,
        "max_episodes": 4000,
        "target_success": 0.75,
        "lr": 3e-4,
        "entropy": 0.035,  # Increased entropy for better exploration
        "max_steps": 1400,  # 175 steps per agent
    },
    {
        "n": 10,
        "min_episodes": 800,
        "max_episodes": 4500,
        "target_success": 0.70,
        "lr": 2.5e-4,
        "entropy": 0.03,
        "max_steps": 1800,  # 180 steps per agent
    },
    {
        "n": 12,
        "min_episodes": 1000,
        "max_episodes": 5000,
        "target_success": 0.65,
        "lr": 2e-4,
        "entropy": 0.025,
        "max_steps": 2200,  # ~183 steps per agent
    },
    {
        "n": 15,
        "min_episodes": 1200,
        "max_episodes": 6000,
        "target_success": 0.60,
        "lr": 1.5e-4,
        "entropy": 0.02,
        "max_steps": 3000,  # 200 steps per agent
    },
    # ========== Large-scale stages (n=18 to n=50) ==========
    # For future scaling beyond n=15
    {
        "n": 18,
        "min_episodes": 1500,
        "max_episodes": 6000,
        "target_success": 0.55,
        "lr": 1.2e-4,
        "entropy": 0.02,
        "max_steps": 3600,  # 200 steps per agent
    },
    {
        "n": 25,
        "min_episodes": 2000,
        "max_episodes": 8000,
        "target_success": 0.50,
        "lr": 1e-4,
        "entropy": 0.015,
        "max_steps": 5000,  # 200 steps per agent
    },
    {
        "n": 35,
        "min_episodes": 2500,
        "max_episodes": 10000,
        "target_success": 0.45,
        "lr": 7e-5,
        "entropy": 0.012,
        "max_steps": 7000,  # 200 steps per agent
    },
    {
        "n": 50,
        "min_episodes": 3000,
        "max_episodes": 12000,
        "target_success": 0.40,
        "lr": 5e-5,
        "entropy": 0.01,
        "max_steps": 10000,  # 200 steps per agent
    },
]


def get_stages_up_to(target_n):
    """Return curriculum stages up to and including target_n."""
    return [s for s in CURRICULUM_STAGES if s["n"] <= target_n]


def find_latest_model(stage_dir):
    """Find the most recent final_model.zip in a stage directory."""
    # Look for timestamped subdirectories
    pattern = os.path.join(stage_dir, "n*_*/final_model.zip")
    models = glob.glob(pattern)

    if not models:
        # Fallback: check directly in stage_dir
        direct_model = os.path.join(stage_dir, "final_model.zip")
        if os.path.exists(direct_model):
            return direct_model
        return None

    # Return most recent
    return max(models, key=os.path.getmtime)


def run_training_stage(
    stage, model_path, log_dir, local_k=7, num_envs=1, use_unlabeled=True
):
    """
    Run a single curriculum stage.

    Args:
        stage: Dictionary with stage configuration
        model_path: Path to pretrained model (None for first stage)
        log_dir: Base log directory
        local_k: Local neighborhood size (default 7 for n up to 50)
        num_envs: Number of parallel environments (default 1)
        use_unlabeled: Use unlabeled (label-agnostic) mode (default True)

    Returns:
        Path to saved model
    """
    stage_dir = os.path.join(log_dir, f"stage_n{stage['n']}")

    # Build training command
    cmd = [
        "python",
        "train/train_sb3.py",
        "--num_agents",
        str(stage["n"]),
        "--max_steps",
        str(stage["max_steps"]),
        "--episodes",
        str(stage["max_episodes"]),
        "--lr",
        str(stage["lr"]),
        "--entropy_coef",
        str(stage["entropy"]),
        # Critical: observation reductions for scaling
        "--use_four_band_reduction",
        "--use_local_neighborhood",
        "--local_neighborhood_k",
        str(local_k),
        # Potential-based reward shaping
        "--enable_potential_reward",
        # Network architecture for larger problems
        "--use_separate_networks",
        "--hidden_dim",
        "512",
        # Normalization
        "--use_vec_normalize",
        # Logging
        "--log_dir",
        stage_dir,
        # Curriculum learning: early stopping when target success rate reached
        "--target_success_rate",
        str(stage["target_success"]),
        "--min_episodes",
        str(stage["min_episodes"]),
        "--success_window_size",
        "100",
        # Parallel environments for faster training
        "--num_envs",
        str(num_envs),
    ]

    # Add unlabeled mode flag if enabled
    if use_unlabeled:
        cmd.append("--use_unlabeled_mode")

    # Add model loading for fine-tuning (except first stage)
    if model_path and os.path.exists(model_path):
        cmd.extend(["--load_model", model_path, "--reset_timesteps"])
        print(f"  Fine-tuning from: {model_path}")
    else:
        print(f"  Training from scratch")

    mode_str = (
        "unlabeled (label-agnostic)" if use_unlabeled else "labeled (exact matching)"
    )
    print(f"\n{'=' * 70}")
    print(f"STAGE: n={stage['n']} [{mode_str}]")
    print(f"{'=' * 70}")
    print(f"Target success rate: {stage['target_success'] * 100:.0f}%")
    print(f"Episodes: {stage['min_episodes']} - {stage['max_episodes']}")
    print(f"Max steps per episode: {stage['max_steps']}")
    print(f"Learning rate: {stage['lr']:.2e}")
    print(f"Entropy coefficient: {stage['entropy']}")
    print(f"Local neighborhood k: {local_k}")
    print(f"Parallel environments: {num_envs}")
    print(f"{'=' * 70}\n")

    # Run training process
    print("Command:", " ".join(cmd))
    print("\n" + "=" * 70)
    print("TRAINING OUTPUT:")
    print("=" * 70 + "\n")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    # Stream output in real-time
    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        print(f"\n❌ Training failed with return code {process.returncode}")
        return None

    # Find saved model
    model_save_path = find_latest_model(stage_dir)

    if model_save_path and os.path.exists(model_save_path):
        print(f"\n✅ Stage completed successfully!")
        print(f"Model saved to: {model_save_path}")
        return model_save_path
    else:
        print(f"\n⚠️ Warning: Could not find saved model in {stage_dir}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum learning for cube reconfiguration scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target_n",
        type=int,
        default=50,
        help="Target number of agents to scale to (default: 50)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/curriculum",
        help="Base directory for all stage logs (default: runs/curriculum)",
    )
    parser.add_argument(
        "--start_stage",
        type=int,
        default=0,
        help="Stage index to start from, 0-indexed (default: 0)",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Path to model to start from (skips earlier stages)",
    )
    parser.add_argument(
        "--local_k",
        type=int,
        default=7,
        help="Local neighborhood size for observation reduction (default: 7)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments for faster training (default: 1, recommended: 4-8)",
    )
    parser.add_argument(
        "--labeled",
        action="store_true",
        help="Use labeled mode instead of unlabeled (agent i must reach position i)",
    )

    args = parser.parse_args()

    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"to_n{args.target_n}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Get stages for target
    stages = get_stages_up_to(args.target_n)

    if not stages:
        print(f"❌ Error: No curriculum stages defined for n={args.target_n}")
        print(f"Available stages: {[s['n'] for s in CURRICULUM_STAGES]}")
        return

    mode_str = "LABELED" if args.labeled else "UNLABELED"
    print("\n" + "=" * 70)
    print(f"CURRICULUM LEARNING PIPELINE ({mode_str})")
    print("=" * 70)
    print(f"Target: n={args.target_n} agents")
    print(
        f"Mode: {'Labeled (exact agent→position)' if args.labeled else 'Unlabeled (shape matching)'}"
    )
    print(f"Curriculum: {' → '.join([f'n={s["n"]}' for s in stages])}")
    print(f"Log directory: {log_dir}")
    print(f"Local neighborhood k: {args.local_k}")
    print(f"Parallel environments: {args.num_envs}")
    print("=" * 70 + "\n")

    # Track model path through stages
    current_model = args.load_model

    # Run each stage
    for i, stage in enumerate(stages):
        if i < args.start_stage:
            print(f"⏭️  Skipping stage {i} (n={stage['n']})")
            # Try to find model from skipped stage for next one
            stage_dir = os.path.join(log_dir, f"stage_n{stage['n']}")
            if os.path.exists(stage_dir):
                found_model = find_latest_model(stage_dir)
                if found_model:
                    current_model = found_model
            continue

        print(f"\n📚 Starting stage {i + 1}/{len(stages)}")

        model_path = run_training_stage(
            stage,
            current_model,
            log_dir,
            args.local_k,
            args.num_envs,
            use_unlabeled=not args.labeled,
        )

        if model_path:
            current_model = model_path
            print(f"\n✅ Stage {i + 1}/{len(stages)} complete: n={stage['n']}")
        else:
            print(f"\n❌ Stage {i + 1}/{len(stages)} failed: n={stage['n']}")
            print("Stopping curriculum training.")
            break

    print(f"\n{'=' * 70}")
    print("CURRICULUM COMPLETE!")
    print("=" * 70)
    if current_model:
        print(f"✅ Final model: {current_model}")
    print(f"📁 All logs saved in: {log_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
