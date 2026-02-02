"""
Training script using Stable-Baselines3 with MaskablePPO.
Integrates custom rewards and action masking with SB3.
"""

# Suppress warnings before importing libraries
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import os
import logging
from datetime import datetime
from typing import Callable
import numpy as np

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ogm.ogm_gym_env import make_ogm_env
from visualizer.step_visualizer import StepVisualizer


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    Args:
        initial_value: Initial learning rate

    Returns:
        Function that computes current learning rate based on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress remaining goes from 1 (beginning) to 0 (end of training).
        """
        return progress_remaining * initial_value

    return func


def setup_logging(log_dir):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_sb3.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info("Logging initialized. Log directory: %s", log_dir)


class CustomCallback(BaseCallback):
    """
    Custom callback to track episode metrics and generate GIFs.
    Supports early stopping for curriculum learning when target success rate is reached.
    Supports periodic model checkpointing.
    """

    def __init__(
        self,
        gif_interval=0,
        log_dir=None,
        num_agents=3,
        verbose=0,
        # Curriculum learning parameters
        target_success_rate=None,  # Target success rate to trigger early stop (0.0-1.0)
        min_episodes=None,  # Minimum episodes before early stopping allowed
        window_size=100,  # Window size for computing rolling success rate
        # Checkpoint parameters
        checkpoint_interval=0,  # Save checkpoint every N episodes (0 disables)
        # Environment kwargs for GIF generation
        env_kwargs=None,
    ):
        super().__init__(verbose)
        self.gif_interval = gif_interval
        self.log_dir = log_dir
        self.num_agents = num_agents
        self.episode_count = 0
        self.success_count = 0
        self.episode_rewards = []
        self.episode_steps = []
        # Curriculum learning
        self.target_success_rate = target_success_rate
        self.min_episodes = min_episodes
        self.window_size = window_size
        self.recent_successes = []  # Rolling window of success/failure
        self.early_stop_triggered = False
        # Checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_episode = 0
        # GIF generation
        self.last_gif_episode = 0
        self.env_kwargs = env_kwargs or {}

    def _on_step(self) -> bool:
        """Called after each environment step. Handles multiple parallel environments."""
        # Get dones and infos for all environments
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0])

        # Process each environment that finished an episode
        for env_idx, (done, info, reward) in enumerate(zip(dones, infos, rewards)):
            if not done:
                continue

            self.episode_count += 1

            # Track metrics
            episode_step = info.get("episode_step", 0)
            self.episode_steps.append(episode_step)
            self.episode_rewards.append(reward)

            # Check success from info captured at termination time
            actual_success = bool(info.get("is_success", False))
            self.success_count += int(actual_success)

            # Track recent successes for rolling window
            self.recent_successes.append(1 if actual_success else 0)
            if len(self.recent_successes) > self.window_size:
                self.recent_successes.pop(0)

            # Extract metrics from info dict (stored at termination step)
            phi_current = info.get("phi_current", None)
            phi_max = info.get("phi_max", None)
            norm_diff = info.get("norm_diff", -1)
            max_diff = info.get("max_diff", -1)

            # Check if episode was truncated due to max_steps
            # Handle VecNormalize wrapper if present
            base_env = self.training_env
            if hasattr(base_env, "venv"):  # VecNormalize wraps venv
                base_env = base_env.venv
            # For SubprocVecEnv, we can't directly access envs, use info instead
            max_steps_from_info = info.get("max_steps", float("inf"))
            truncated_by_steps = episode_step >= max_steps_from_info

            # Log episode results (less verbose for parallel envs)
            if (
                self.episode_count % 10 == 0 or actual_success
            ):  # Log every 10th or successes
                if phi_current is not None:
                    logging.info(
                        "Episode %d [env%d] after %d steps -- reward: %.3f : success = %s | φ: %.4f/%.4f | truncated: %s",
                        self.episode_count,
                        env_idx,
                        episode_step,
                        reward,
                        actual_success,
                        phi_current,
                        phi_max,
                        truncated_by_steps,
                    )
                else:
                    logging.info(
                        "Episode %d [env%d] after %d steps -- reward: %.3f : success = %s",
                        self.episode_count,
                        env_idx,
                        episode_step,
                        reward,
                        actual_success,
                    )

            # Log to tensorboard
            if len(self.episode_rewards) > 0:
                self.logger.record(
                    "custom/success_rate", self.success_count / self.episode_count
                )
                self.logger.record(
                    "custom/avg_episode_length", np.mean(self.episode_steps[-100:])
                )
                # Log rolling success rate
                if len(self.recent_successes) >= self.window_size:
                    rolling_rate = sum(self.recent_successes) / len(
                        self.recent_successes
                    )
                    self.logger.record("custom/rolling_success_rate", rolling_rate)

            # Check for early stopping (curriculum advancement)
            if self._should_early_stop():
                logging.info(
                    "🎯 TARGET REACHED! Rolling success rate %.1f%% >= %.1f%% after %d episodes",
                    100.0 * sum(self.recent_successes) / len(self.recent_successes),
                    100.0 * self.target_success_rate,
                    self.episode_count,
                )
                self.early_stop_triggered = True
                return False  # Stop training

            # Save checkpoint periodically
            if self._should_save_checkpoint():
                self._save_checkpoint()

            # Generate GIF periodically
            if self._should_generate_gif():
                self._generate_gif()

        return True

    def _should_early_stop(self) -> bool:
        """Check if we should trigger early stopping based on success rate."""
        if self.target_success_rate is None:
            return False

        # Need minimum episodes before checking
        if self.min_episodes and self.episode_count < self.min_episodes:
            return False

        # Need full window of data
        if len(self.recent_successes) < self.window_size:
            return False

        # Check rolling success rate
        rolling_rate = sum(self.recent_successes) / len(self.recent_successes)
        return rolling_rate >= self.target_success_rate

    def _should_save_checkpoint(self) -> bool:
        """Check if we should save a checkpoint based on episode count."""
        if self.checkpoint_interval <= 0:
            return False
        if self.episode_count - self.last_checkpoint_episode >= self.checkpoint_interval:
            return True
        return False

    def _save_checkpoint(self):
        """Save a model checkpoint."""
        if self.log_dir is None:
            return

        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get rolling success rate for filename
        if len(self.recent_successes) >= self.window_size:
            rolling_rate = sum(self.recent_successes) / len(self.recent_successes)
            rate_str = f"_sr{rolling_rate:.2f}"
        else:
            rate_str = ""

        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_ep{self.episode_count}{rate_str}"
        )
        self.model.save(checkpoint_path)
        self.last_checkpoint_episode = self.episode_count
        logging.info(
            "💾 Checkpoint saved: %s (episode %d, success rate: %.1f%%)",
            checkpoint_path,
            self.episode_count,
            100.0 * self.success_count / self.episode_count if self.episode_count > 0 else 0,
        )

    def _should_generate_gif(self) -> bool:
        """Check if we should generate a GIF based on episode count."""
        if self.gif_interval <= 0:
            return False
        if self.episode_count - self.last_gif_episode >= self.gif_interval:
            return True
        return False

    def _generate_gif(self):
        """Generate a GIF by running an evaluation episode."""
        if self.log_dir is None:
            return

        eval_env = None
        try:
            from ogm.ogm_gym_env import OGMGymEnv
            from visualizer.step_visualizer import StepVisualizer
            from stable_baselines3.common.vec_env import VecNormalize

            gif_dir = os.path.join(self.log_dir, "gifs")
            os.makedirs(gif_dir, exist_ok=True)

            # Create a fresh evaluation environment
            eval_env = OGMGymEnv(num_agents=self.num_agents, **self.env_kwargs)
            
            # Check if training uses VecNormalize - we'll need to normalize obs manually
            training_env = self.training_env
            use_normalization = isinstance(training_env, VecNormalize)
            
            if use_normalization:
                # Get normalization parameters from training env
                obs_rms = training_env.obs_rms
                clip_obs = training_env.clip_obs
                epsilon = training_env.epsilon

            def normalize_obs(obs):
                """Normalize observation using training env statistics."""
                if not use_normalization:
                    return obs
                # Apply same normalization as VecNormalize
                obs = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon)
                obs = np.clip(obs, -clip_obs, clip_obs)
                return obs.astype(np.float32)

            # Reset and get initial config
            obs, _ = eval_env.reset()

            # Create visualizer with the OGM
            gif_path = os.path.join(gif_dir, f"episode_{self.episode_count}.gif")
            visualizer = StepVisualizer(eval_env.env.ogm, output_path=gif_path)
            visualizer.capture_state()

            # Run evaluation episode
            done = False
            truncated = False
            max_eval_steps = eval_env.max_steps
            step = 0

            while not (done or truncated) and step < max_eval_steps:
                # Get action mask
                action_masks = eval_env.action_masks()

                # Normalize obs and predict action using current model
                normalized_obs = normalize_obs(obs)
                action, _ = self.model.predict(normalized_obs, deterministic=True, action_masks=action_masks)

                # Take step
                obs, reward, done, truncated, info = eval_env.step(action)

                # Capture state after each full phase (all agents acted)
                if eval_env.current_agent_idx == 0:
                    visualizer.capture_state()
                    step += 1
                    
                    # Debug: log progress every 100 steps for GIF
                    if step % 100 == 0:
                        is_close = eval_env.env.ogm.check_final()
                        logging.debug(
                            "GIF eval step %d: done=%s, is_success=%s",
                            step, done, is_close
                        )

            # Generate the GIF
            visualizer.animate(pause_frames=10)

            success = eval_env.env.ogm.check_final()
            self.last_gif_episode = self.episode_count
            logging.info(
                "🎬 GIF saved: %s (episode %d, %d steps, success=%s)",
                gif_path,
                self.episode_count,
                step,
                success,
            )

        except Exception as e:
            logging.warning("Failed to generate GIF: %s", str(e))
            import traceback
            logging.warning("Traceback: %s", traceback.format_exc())
        finally:
            if eval_env is not None:
                try:
                    eval_env.close()
                except:
                    pass

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (after collecting experience)."""
        if self.episode_count > 0:
            success_rate = 100.0 * self.success_count / self.episode_count
            logging.info(
                "Progress: %d episodes completed | Success rate: %.2f%% (%d/%d)",
                self.episode_count,
                success_rate,
                self.success_count,
                self.episode_count,
            )


def train(args):
    """Main training loop using SB3."""
    setup_logging(args.log_dir)

    # Create environment with your custom rewards
    def make_env():
        return make_ogm_env(
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            step_cost=-0.01,
            enable_bounty_reward=args.enable_bounty_reward,
            bounty_gamma=args.bounty_gamma,
            bounty_eta=args.bounty_eta,
            bounty_base_value=args.bounty_base_value,
            bounty_total_frac_of_success=args.bounty_total_frac_of_success,
            bounty_cap_per_step=args.bounty_cap_per_step,
            enable_potential_reward=args.enable_potential_reward,
            potential_scale=args.potential_scale,
            potential_normalize=args.potential_normalize,
            success_bonus=args.success_bonus,
            step_cost_initial=args.step_cost_initial,
            step_cost_min=args.step_cost_min,
            use_exponential_decay=args.use_exponential_decay,
            # Soft matching reward parameters
            enable_soft_matching_reward=args.enable_soft_matching_reward,
            soft_matching_decay_beta=args.soft_matching_decay_beta,
            soft_matching_scale=args.soft_matching_scale,
            # Dimension reduction parameters
            use_four_band_reduction=args.use_four_band_reduction,
            use_local_neighborhood=args.use_local_neighborhood,
            local_neighborhood_k=args.local_neighborhood_k,
            # Unlabeled mode
            use_unlabeled_mode=args.use_unlabeled_mode,
            # Agent-specific local reward
            enable_local_reward=args.enable_local_reward,
            local_reward_scale=args.local_reward_scale,
        )

    # Create vectorized environment
    # SubprocVecEnv runs each env in a separate process (true parallelism)
    # DummyVecEnv runs sequentially (fallback for debugging)
    num_envs = args.num_envs
    if num_envs > 1:
        # Parallel environments using multiprocessing
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
        logging.info("Using %d parallel environments (SubprocVecEnv)", num_envs)
    else:
        # Single environment (original behavior)
        env = DummyVecEnv([make_env])
        logging.info("Using single environment (DummyVecEnv)")

    # Apply observation and reward normalization if enabled
    if args.use_vec_normalize:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=args.gamma,
        )
        logging.info("VecNormalize enabled: norm_obs=True, norm_reward=True")
        
        # If loading a pretrained model, try to load matching VecNormalize stats
        if args.load_model:
            # Look for vec_normalize.pkl in the same directory as the model
            model_dir = os.path.dirname(args.load_model)
            vec_normalize_path = os.path.join(model_dir, "vec_normalize.pkl")
            if os.path.exists(vec_normalize_path):
                import pickle
                with open(vec_normalize_path, "rb") as f:
                    old_env = pickle.load(f)
                # Copy the learned normalization statistics
                env.obs_rms = old_env.obs_rms
                env.ret_rms = old_env.ret_rms
                logging.info("Loaded VecNormalize stats from: %s", vec_normalize_path)
            else:
                logging.warning(
                    "No vec_normalize.pkl found at %s - using fresh normalization stats",
                    vec_normalize_path
                )

    logging.info(
        "Environment created: %d agents, max_steps=%d", args.num_agents, args.max_steps
    )
    logging.info(
        "Dimension reduction: four_band=%s, local_neighborhood=%s (k=%d)",
        args.use_four_band_reduction,
        args.use_local_neighborhood,
        args.local_neighborhood_k,
    )
    logging.info(
        "Soft matching reward: %s (scale=%.2f, beta=%.4f)",
        args.enable_soft_matching_reward,
        args.soft_matching_scale,
        args.soft_matching_decay_beta,
    )
    logging.info(
        "Local reward: %s (scale=%.2f)",
        args.enable_local_reward,
        args.local_reward_scale,
    )
    logging.info("Unlabeled mode: %s", args.use_unlabeled_mode)

    # Configure SB3 logger for tensorboard
    sb3_logger = configure(args.log_dir, ["stdout", "tensorboard"])

    # Learning rate: use schedule if enabled, otherwise constant
    if args.use_lr_schedule:
        learning_rate = linear_schedule(args.lr)
        logging.info("Using linear learning rate schedule: %.2e -> 0", args.lr)
    else:
        learning_rate = args.lr

    # Network architecture: separate policy and value networks for better learning
    if args.use_separate_networks:
        net_arch = dict(
            pi=[
                args.hidden_dim,
                args.hidden_dim,
                args.hidden_dim // 2,
            ],  # Policy: 3 layers
            vf=[
                args.hidden_dim,
                args.hidden_dim,
                args.hidden_dim // 2,
            ],  # Value: 3 layers
        )
        logging.info("Using separate pi/vf networks: %s", net_arch)
    else:
        net_arch = [args.hidden_dim, args.hidden_dim]
        logging.info("Using shared network: %s", net_arch)

    # Create or load MaskablePPO model
    if args.load_model:
        # Load pretrained model for fine-tuning
        logging.info("Loading pretrained model from: %s", args.load_model)
        model = MaskablePPO.load(
            args.load_model,
            env=env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            gae_lambda=args.lam,
            clip_range=args.clip,
            ent_coef=args.entropy_coef,
            vf_coef=args.value_coef,
            max_grad_norm=args.grad_clip,
            verbose=1,
            tensorboard_log=args.log_dir,
        )
        if args.reset_timesteps:
            model.num_timesteps = 0
            logging.info("Reset timestep counter to 0")
        logging.info("Successfully loaded model for fine-tuning")
    else:
        # Create new model from scratch
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,  # Steps per rollout
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            gae_lambda=args.lam,
            clip_range=args.clip,
            ent_coef=args.entropy_coef,
            vf_coef=args.value_coef,
            max_grad_norm=args.grad_clip,
            verbose=1,
            tensorboard_log=args.log_dir,
            policy_kwargs={"net_arch": net_arch},
        )

    model.set_logger(sb3_logger)
    logging.info("MaskablePPO model created with hyperparameters:")
    logging.info(
        "  lr=%.2e, gamma=%.3f, lambda=%.3f, clip=%.2f",
        args.lr,
        args.gamma,
        args.lam,
        args.clip,
    )
    logging.info(
        "  hidden_dim=%d, entropy_coef=%.3f, grad_clip=%.2f",
        args.hidden_dim,
        args.entropy_coef,
        args.grad_clip,
    )

    # Create callback with curriculum learning support
    # Environment kwargs needed for GIF generation (to recreate env)
    env_kwargs = {
        "max_steps": args.max_steps,
        "enable_potential_reward": args.enable_potential_reward,
        "use_four_band_reduction": args.use_four_band_reduction,
        "use_local_neighborhood": args.use_local_neighborhood,
        "local_neighborhood_k": args.local_neighborhood_k,
        "use_unlabeled_mode": args.use_unlabeled_mode,
    }

    callback = CustomCallback(
        gif_interval=args.gif_interval,
        log_dir=args.log_dir,
        num_agents=args.num_agents,
        verbose=1,
        # Curriculum learning parameters
        target_success_rate=args.target_success_rate,
        min_episodes=args.min_episodes,
        window_size=args.success_window_size,
        # Checkpointing
        checkpoint_interval=args.checkpoint_interval,
        # Environment kwargs for GIF generation
        env_kwargs=env_kwargs,
    )

    # Log early stopping configuration
    if args.target_success_rate is not None:
        logging.info(
            "Early stopping enabled: target_success_rate=%.1f%%, min_episodes=%s, window_size=%d",
            args.target_success_rate * 100,
            args.min_episodes,
            args.success_window_size,
        )
    else:
        logging.info("Early stopping disabled (no target_success_rate set)")

    if args.checkpoint_interval > 0:
        logging.info("Checkpointing enabled: saving every %d episodes", args.checkpoint_interval)
    else:
        logging.info("Checkpointing disabled")

    if args.gif_interval > 0:
        logging.info("GIF generation enabled: every %d episodes", args.gif_interval)
    else:
        logging.info("GIF generation disabled")

    # Calculate total timesteps based on episodes
    # Each episode is roughly (num_agents * steps_per_episode) environment steps
    total_timesteps = args.episodes * args.max_steps * args.num_agents
    logging.info(
        "Starting training for %d total timesteps (~%d episodes)",
        total_timesteps,
        args.episodes,
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # Save final model
    model_path = os.path.join(args.log_dir, "final_model")
    model.save(model_path)
    logging.info("Model saved to: %s", model_path)

    # Save VecNormalize statistics if used
    if args.use_vec_normalize:
        vec_normalize_path = os.path.join(args.log_dir, "vec_normalize.pkl")
        env.save(vec_normalize_path)
        logging.info("VecNormalize stats saved to: %s", vec_normalize_path)

    # Print final statistics
    if callback.episode_count > 0:
        success_rate = 100.0 * callback.success_count / callback.episode_count
        logging.info(
            "Training complete! Final success rate: %.2f%% (%d/%d)",
            success_rate,
            callback.success_count,
            callback.episode_count,
        )
        if len(callback.episode_steps) > 0:
            logging.info(
                "Average steps per episode: %.2f", np.mean(callback.episode_steps)
            )

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train OGM agents using SB3 MaskablePPO"
    )

    # Training parameters
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Approximate number of episodes to train",
    )
    parser.add_argument(
        "--num_agents", type=int, default=3, help="Number of agents in the environment"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Maximum steps per episode"
    )

    # SB3-specific parameters
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of steps per rollout (SB3 parameter)",
    )

    # PPO hyperparameters (matching your SimplePPOAgent)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of epochs per update"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--value_coef", type=float, default=0.5, help="Value loss coefficient"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=0.5, help="Gradient clipping threshold"
    )

    # Logging parameters
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/sb3",
        help="Directory for logs and TensorBoard",
    )
    parser.add_argument(
        "--gif_interval",
        type=int,
        default=0,
        help="Save an episode GIF every N episodes (0 disables)",
    )

    # Soft matching reward parameters
    parser.add_argument(
        "--enable_soft_matching_reward",
        action="store_true",
        help="Enable soft pairwise matching reward",
    )
    parser.add_argument(
        "--soft_matching_decay_beta",
        type=float,
        default=0.999,
        help="Time decay factor for soft matching reward",
    )
    parser.add_argument(
        "--soft_matching_scale",
        type=float,
        default=100.0,
        help="Scaling factor for soft matching reward",
    )

    # Bounty reward parameters
    parser.add_argument(
        "--enable_bounty_reward",
        action="store_true",
        help="Enable bounty reward system",
    )
    parser.add_argument(
        "--bounty_gamma", type=float, default=0.999, help="Bounty decay factor"
    )
    parser.add_argument(
        "--bounty_eta", type=float, default=2.0, help="Bounty multiplier coefficient"
    )
    parser.add_argument(
        "--bounty_base_value", type=float, default=1.0, help="Base bounty value"
    )
    parser.add_argument(
        "--bounty_total_frac_of_success",
        type=float,
        default=0.2,
        help="Total bounty as fraction of success bonus",
    )
    parser.add_argument(
        "--bounty_cap_per_step",
        type=float,
        default=20.0,
        help="Maximum bounty reward per step",
    )

    # Potential reward parameters
    parser.add_argument(
        "--enable_potential_reward",
        action="store_true",
        default=True,
        help="Enable potential-based shaping reward",
    )
    parser.add_argument(
        "--disable_potential_reward",
        action="store_false",
        dest="enable_potential_reward",
        help="Disable potential-based shaping reward",
    )
    parser.add_argument(
        "--potential_scale",
        type=float,
        default=1.0,
        help="Scaling factor for potential reward",
    )
    parser.add_argument(
        "--potential_normalize",
        type=str,
        default="n2",
        choices=["n2", "sqrt_n", "none"],
        help="Normalization for potential reward (n2 now uses n, not n²)",
    )
    parser.add_argument(
        "--success_bonus",
        type=float,
        default=100.0,
        help="Bonus reward for successful completion",
    )

    # Step cost parameters
    parser.add_argument(
        "--step_cost_initial", type=float, default=-0.01, help="Initial step cost"
    )
    parser.add_argument(
        "--step_cost_min", type=float, default=-0.001, help="Minimum step cost"
    )
    parser.add_argument(
        "--use_exponential_decay",
        action="store_true",
        default=True,
        help="Use exponential decay for step cost",
    )

    # Dimension reduction parameters
    parser.add_argument(
        "--use_four_band_reduction",
        action="store_true",
        help="Use four-band reduction (reduces O(n²) to O(n))",
    )
    parser.add_argument(
        "--use_local_neighborhood",
        action="store_true",
        help="Use local neighborhood reduction (further reduces to O(k))",
    )
    parser.add_argument(
        "--local_neighborhood_k",
        type=int,
        default=3,
        help="Number of rows for local neighborhood (default: 3)",
    )

    # Unlabeled mode (label-agnostic)
    parser.add_argument(
        "--use_unlabeled_mode",
        action="store_true",
        help="Enable unlabeled (label-agnostic) mode for rewards",
    )

    # Agent-specific local reward parameters
    parser.add_argument(
        "--enable_local_reward",
        action="store_true",
        help="Enable agent-specific local reward for better credit assignment",
    )
    parser.add_argument(
        "--local_reward_scale",
        type=float,
        default=1.0,
        help="Scaling factor for local reward",
    )

    # VecNormalize parameters
    parser.add_argument(
        "--use_vec_normalize",
        action="store_true",
        help="Enable observation and reward normalization via VecNormalize",
    )

    # Network architecture parameters
    parser.add_argument(
        "--use_separate_networks",
        action="store_true",
        help="Use separate policy and value networks (deeper architecture)",
    )

    # Learning rate schedule
    parser.add_argument(
        "--use_lr_schedule",
        action="store_true",
        help="Use linear learning rate schedule (decays to 0)",
    )

    # Model loading for curriculum/fine-tuning
    parser.add_argument(
        "--load_model",
        type=str,
        default=None,
        help="Path to pretrained model (.zip) to fine-tune from",
    )
    parser.add_argument(
        "--reset_timesteps",
        action="store_true",
        help="Reset training timestep counter when loading model",
    )

    # Curriculum learning / early stopping parameters
    parser.add_argument(
        "--target_success_rate",
        type=float,
        default=None,
        help="Target success rate (0.0-1.0) to trigger early stop for curriculum advancement",
    )
    parser.add_argument(
        "--min_episodes",
        type=int,
        default=None,
        help="Minimum episodes before early stopping is allowed",
    )
    parser.add_argument(
        "--success_window_size",
        type=int,
        default=100,
        help="Window size for computing rolling success rate (default: 100)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=0,
        help="Save model checkpoint every N episodes (default: 0 = disabled)",
    )

    # Parallel environments for faster training
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1, use 4-8 for speedup)",
    )

    args = parser.parse_args()

    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir = os.path.join(args.log_dir, f"n{args.num_agents}_{timestamp}")

    train(args)
