from sb3_contrib import MaskablePPO
from ogm.ogm_gym_env import make_ogm_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import subprocess
import logging
import os

#export QT_QPA_PLATFORM=offscreen

# def run_testing_stage(
#     stage,
#     model_path,
#     log_dir,
#     local_k=7,
#     num_envs=1,
#     use_unlabeled=True,
#     plateau_min_delta=0.005,
# ):
#     """
#     Run a single curriculum stage.

#     Args:
#         stage: Dictionary with stage configuration
#         model_path: Path to pretrained model (None for first stage)
#         log_dir: Base log directory
#         local_k: Local neighborhood size (default 7 for n up to 50)
#         num_envs: Number of parallel environments (default 1)
#         use_unlabeled: Use unlabeled (label-agnostic) mode (default True)
#         plateau_min_delta: Minimum rolling-success improvement for plateau reset

#     Returns:
#         Path to saved model
#     """
#     stage_dir = os.path.join(log_dir, f"stage_n{stage['n']}")

#     # Build training command
#     cmd = [
#         "python",
#         "train/train_sb3.py",
#         "--num_agents",
#         str(stage["n"]),
#         "--max_steps",
#         str(stage["max_steps"]),
#         "--episodes",
#         str(stage["max_episodes"]),
#         "--lr",
#         str(stage["lr"]),
#         "--entropy_coef",
#         str(stage["entropy"]),
#         # Critical: observation reductions for scaling
#         "--use_four_band_reduction",
#         "--use_local_neighborhood",
#         "--local_neighborhood_k",
#         str(local_k),
#         # Potential-based reward shaping
#         "--enable_potential_reward",
#         # Network architecture for larger problems
#         "--use_separate_networks",
#         "--hidden_dim",
#         "512",
#         # Normalization
#         "--use_vec_normalize",
#         # Logging
#         "--log_dir",
#         stage_dir,
#         # Curriculum learning: early stopping when target success rate reached
#         "--target_success_rate",
#         str(stage["target_success"]),
#         "--min_episodes",
#         str(stage["min_episodes"]),
#         "--success_window_size",
#         "100",
#         "--plateau_patience",
#         str(stage.get("plateau_patience", 0)),
#         "--plateau_min_delta",
#         str(plateau_min_delta),
#         # Parallel environments for faster training
#         "--num_envs",
#         str(num_envs),
#     ]

#     # Add unlabeled mode flag if enabled
#     if use_unlabeled:
#         cmd.append("--use_unlabeled_mode")

#     # Add model loading for fine-tuning (except first stage)
#     if model_path and os.path.exists(model_path):
#         cmd.extend(["--load_model", model_path, "--reset_timesteps"])
#         print(f"  Fine-tuning from: {model_path}")
#     else:
#         print(f"  Training from scratch")

#     mode_str = (
#         "unlabeled (label-agnostic)" if use_unlabeled else "labeled (exact matching)"
#     )
#     print(f"\n{'=' * 70}")
#     print(f"STAGE: n={stage['n']} [{mode_str}]")
#     print(f"{'=' * 70}")
#     print(f"Target success rate: {stage['target_success'] * 100:.0f}%")
#     print(f"Episodes: {stage['min_episodes']} - {stage['max_episodes']}")
#     print(f"Max steps per episode: {stage['max_steps']}")
#     print(f"Learning rate: {stage['lr']:.2e}")
#     print(f"Entropy coefficient: {stage['entropy']}")
#     print(f"Local neighborhood k: {local_k}")
#     print(f"Parallel environments: {num_envs}")
#     print(f"Plateau patience: {stage.get('plateau_patience', 0)} episodes")
#     print(f"{'=' * 70}\n")

#     # Run training process
#     print("Command:", " ".join(cmd))
#     print("\n" + "=" * 70)
#     print("TRAINING OUTPUT:")
#     print("=" * 70 + "\n")

#     process = subprocess.Popen(
#         cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
#     )

#     # Stream output in real-time
#     for line in process.stdout:
#         print(line, end="", flush=True)

#     process.wait()

#     if process.returncode != 0:
#         print(f"\n❌ Training failed with return code {process.returncode}")
#         return None

#     # Find saved model
#     model_save_path = find_latest_model(stage_dir)

#     if model_save_path and os.path.exists(model_save_path):
#         print(f"\n✅ Stage completed successfully!")
#         print(f"Model saved to: {model_save_path}")
#         return model_save_path
#     else:
#         print(f"\n⚠️ Warning: Could not find saved model in {stage_dir}")
#         return None

# Curriculum stages: progressively increase n with appropriate hyperparameters
# Each stage fine-tunes from the previous one
#
# Design principles (v3 - tuned for reliable progression to n=15):
# 1. Smooth progression with small jumps: n=4→5→6→7→8→10→12→15
# 2. Realistic stage targets from observed runs (n=8 is typically ~55-60%)
# 3. Plateau stops to avoid spending thousands of episodes on stalled stages
# 4. More generous max_steps at larger n to reduce hard truncation bias
#

def setup_logging(log_dir):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "testing_sb3.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("Logging initialized. Log directory: %s", log_dir)

CURRICULUM_STAGES = [
    # ========== Small-scale stages (n=4 to n=7) ==========
    # These build foundational skills with smooth progression
    {
        "n": 4,
        "min_episodes": 400,
        "max_episodes": 1800,
        "target_success": 0.90,
        "lr": 5e-4,
        "entropy": 0.03,
        "max_steps": 600,  # 150 steps per agent
        "plateau_patience": 300,
    },
    {
        "n": 5,
        "min_episodes": 450,
        "max_episodes": 2200,
        "target_success": 0.86,
        "lr": 4.5e-4,
        "entropy": 0.03,
        "max_steps": 800,  # 160 steps per agent
        "plateau_patience": 350,
    },
    {
        "n": 6,
        "min_episodes": 550,
        "max_episodes": 2600,
        "target_success": 0.82,
        "lr": 4e-4,
        "entropy": 0.03,
        "max_steps": 950,  # ~158 steps per agent
        "plateau_patience": 400,
    },
    {
        "n": 7,
        "min_episodes": 600,
        "max_episodes": 3000,
        "target_success": 0.78,
        "lr": 3.5e-4,
        "entropy": 0.03,
        "max_steps": 1100,  # ~157 steps per agent
        "plateau_patience": 450,
    },
    # ========== Medium-scale stages (n=8 to n=15) ==========
    # Targets are intentionally lower than small-n stages to avoid
    # stalling the curriculum at n=8.
    {
        "n": 8,
        "min_episodes": 700,
        "max_episodes": 4200,
        "target_success": 0.58,
        "lr": 2.5e-4,
        "entropy": 0.03,
        "max_steps": 1600,  # 200 steps per agent
        "plateau_patience": 600,
    },
    {
        "n": 10,
        "min_episodes": 900,
        "max_episodes": 5000,
        "target_success": 0.50,
        "lr": 2e-4,
        "entropy": 0.028,
        "max_steps": 2200,  # 220 steps per agent
        "plateau_patience": 700,
    },
    {
        "n": 12,
        "min_episodes": 1100,
        "max_episodes": 6000,
        "target_success": 0.44,
        "lr": 1.5e-4,
        "entropy": 0.025,
        "max_steps": 2800,  # ~233 steps per agent
        "plateau_patience": 800,
    },
"""     {
        "n": 15,
        "min_episodes": 1400,
        "max_episodes": 7500,
        "target_success": 0.38,
        "lr": 1.2e-4,
        "entropy": 0.022,
        "max_steps": 3800,  # ~253 steps per agent
        "plateau_patience": 1000,
    },
    # ========== Large-scale stages (n=18 to n=50) ==========
    # For future scaling beyond n=15
    {
        "n": 18,
        "min_episodes": 1500,
        "max_episodes": 6000,
        "target_success": 0.34,
        "lr": 1e-4,
        "entropy": 0.02,
        "max_steps": 3600,  # 200 steps per agent
        "plateau_patience": 1200,
    },
    {
        "n": 25,
        "min_episodes": 2000,
        "max_episodes": 8000,
        "target_success": 0.30,
        "lr": 8e-5,
        "entropy": 0.015,
        "max_steps": 5000,  # 200 steps per agent
        "plateau_patience": 1500,
    },
    {
        "n": 35,
        "min_episodes": 2500,
        "max_episodes": 10000,
        "target_success": 0.26,
        "lr": 6e-5,
        "entropy": 0.012,
        "max_steps": 7000,  # 200 steps per agent
        "plateau_patience": 1800,
    },
    {
        "n": 50,
        "min_episodes": 3000,
        "max_episodes": 12000,
        "target_success": 0.22,
        "lr": 4e-5,
        "entropy": 0.01,
        "max_steps": 10000,  # 200 steps per agent
        "plateau_patience": 2200,
    }, """
]


######################## MY STUFF #####################################################
# for loop for stages
stage = CURRICULUM_STAGES[0]
asdf = 10
num_envs = 4
#model = MaskablePPO.load("runs/sb3/final_model")
# Parallel environments using multiprocessing
#env = SubprocVecEnv([make_env for _ in range(num_envs)])
model = MaskablePPO.load("/home/benjamin_faught/Pivoting-Cube-Reconfiguration/runs/curriculum_n12/to_n12_20260610_230927/stage_n4/n4_20260610_230932/final_model.zip")#,
        #     env=env,
        #     learning_rate=learning_rate,
        #     n_steps=args.n_steps,
        #     batch_size=args.batch_size,
        #     n_epochs=args.epochs,
        #     gamma=args.gamma,
        #     gae_lambda=args.lam,
        #     clip_range=args.clip,
        #     ent_coef=args.entropy_coef,
        #     vf_coef=args.value_coef,
        #     max_grad_norm=args.grad_clip,
        #     verbose=1,
        #     tensorboard_log=args.log_dir,
        # )
#env = make_ogm_env(num_agents=4, max_steps=500)
#env = make_ogm_env(num_agents=stage["n"], max_steps=stage["max_steps"], use_four_band_reduction=True, use_local_neighborhood=True, local_neighborhood_k=7, enable_potential_reward=True)
ogm_env = make_ogm_env(
            num_agents=stage["n"],
            max_steps=stage["max_steps"],
            step_cost=-0.01,
            enable_bounty_reward=False,
            bounty_gamma=0.999,
            bounty_eta=2.0,
            bounty_base_value=1.0,
            bounty_total_frac_of_success=0.2,
            bounty_cap_per_step=20.0,
            enable_potential_reward=True,
            potential_scale=1.0,
            potential_normalize="n2",
            success_bonus=100.0,
            step_cost_initial=-0.01,
            step_cost_min=-0.001,
            use_exponential_decay=True,
            # Soft matching reward parameters
            enable_soft_matching_reward=False,
            soft_matching_decay_beta=0.999,
            soft_matching_scale=100.0,
            # Dimension reduction parameters
            use_four_band_reduction=True,
            use_local_neighborhood=True,
            local_neighborhood_k=9,
            # Unlabeled mode
            use_unlabeled_mode=True,
            # Agent-specific local reward
            enable_local_reward=False,
            local_reward_scale=1.0,
        )

######################## NOT MY STUFF #####################################################

 # Run each stage
# for i, stage in enumerate(CURRICULUM_STAGES):
#     # if i < args.start_stage:
#     #     print(f"⏭️  Skipping stage {i} (n={stage['n']})")
#     #     # Try to find model from skipped stage for next one
#     #     stage_dir = os.path.join(log_dir, f"stage_n{stage['n']}")
#     #     if os.path.exists(stage_dir):
#     #         found_model = find_latest_model(stage_dir)
#     #         if found_model:
#     #             current_model = found_model
#     #     continue

#     print(f"\n📚 Starting stage {i + 1}/{len(stages)}")

#     model_path = run_training_stage(
#         stage,
#         current_model,
#         log_dir,
#         args.local_k,
#         resolved_num_envs,
#         use_unlabeled=not args.labeled,
#         plateau_min_delta=args.plateau_min_delta,
#     )

#     if model_path:
#         current_model = model_path
#         print(f"\n✅ Stage {i + 1}/{len(stages)} complete: n={stage['n']}")
#     else:
#         print(f"\n❌ Stage {i + 1}/{len(stages)} failed: n={stage['n']}")
#         print("Stopping curriculum training.")
#         break

######################## MY STUFF #####################################################

setup_logging("")
obs, info = ogm_env.reset()
done = False
while not done:
    action_mask = ogm_env.action_masks()
    action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
    obs, reward, terminated, truncated, info = ogm_env.step(action)
    curr_sqdist = ogm_env.env.ogm.compute_pairwise_sqdist(
                ogm_env.env.ogm.module_positions
            )
    phi = ogm_env.env.compute_unlabeled_soft_matching_score(curr_sqdist, ogm_env.env.final_sqdist)
    logging.info("reward: %s", reward)
    logging.info("phi: %s", phi)
    done = terminated or truncated