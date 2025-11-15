"""
Training script using Stable-Baselines3 with MaskablePPO.
Integrates custom rewards and action masking with SB3.
"""
import argparse
import os
import logging
from datetime import datetime
import numpy as np

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from ogm.ogm_gym_env import make_ogm_env
from visualizer.step_visualizer import StepVisualizer


def setup_logging(log_dir):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_sb3.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )
    logging.info("Logging initialized. Log directory: %s", log_dir)


class CustomCallback(BaseCallback):
    """
    Custom callback to track episode metrics and generate GIFs.
    """
    def __init__(self, gif_interval=0, log_dir=None, num_agents=3, verbose=0):
        super().__init__(verbose)
        self.gif_interval = gif_interval
        self.log_dir = log_dir
        self.num_agents = num_agents
        self.episode_count = 0
        self.success_count = 0
        self.episode_rewards = []
        self.episode_steps = []

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check if episode is done
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1

            # Get episode info
            info = self.locals.get('infos', [{}])[0]
            episode_reward = self.locals.get('rewards', [0])[0]

            # Track metrics
            episode_step = info.get('episode_step', 0)
            self.episode_steps.append(episode_step)
            self.episode_rewards.append(episode_reward)

            # Check success from info captured at termination time
            actual_success = bool(info.get('is_success', False))
            self.success_count += int(actual_success)

            # Extract metrics from info dict (stored at termination step)
            phi_current = info.get('phi_current', None)
            phi_max = info.get('phi_max', None)
            norm_diff = info.get('norm_diff', -1)
            max_diff = info.get('max_diff', -1)

            # Check if episode was truncated due to max_steps
            gym_env = self.training_env.envs[0].env  # OGMGymEnv
            truncated_by_steps = episode_step >= gym_env.max_steps

            # Log episode results with soft matching info
            if phi_current is not None:
                logging.info(
                    "Episode %d finished after %d steps -- reward: %.3f : success = %s | φ: %.4f/%.4f | norm_diff: %.6f, max_diff: %.6f | truncated: %s",
                    self.episode_count,
                    episode_step,
                    episode_reward,
                    actual_success,
                    phi_current,
                    phi_max,
                    norm_diff,
                    max_diff,
                    truncated_by_steps
                )
            else:
                logging.info(
                    "Episode %d finished after %d steps -- reward: %.3f : success = %s",
                    self.episode_count,
                    episode_step,
                    episode_reward,
                    actual_success
                )

            # Log to tensorboard (SB3 handles this automatically, but we can add custom metrics)
            if len(self.episode_rewards) > 0:
                self.logger.record("custom/success_rate", self.success_count / self.episode_count)
                self.logger.record("custom/avg_episode_length", np.mean(self.episode_steps[-100:]))

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (after collecting experience)."""
        if self.episode_count > 0:
            success_rate = 100.0 * self.success_count / self.episode_count
            logging.info(
                "Progress: %d episodes completed | Success rate: %.2f%% (%d/%d)",
                self.episode_count,
                success_rate,
                self.success_count,
                self.episode_count
            )


def train(args):
    """Main training loop using SB3."""
    setup_logging(args.log_dir)

    # Create environment with your custom rewards
    env = make_ogm_env(
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
    )

    logging.info("Environment created: %d agents, max_steps=%d", args.num_agents, args.max_steps)
    logging.info("Soft matching reward: %s (scale=%.2f, beta=%.4f)",
                 args.enable_soft_matching_reward,
                 args.soft_matching_scale,
                 args.soft_matching_decay_beta)

    # Configure SB3 logger for tensorboard
    sb3_logger = configure(args.log_dir, ["stdout", "tensorboard"])

    # Create MaskablePPO model
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
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
        policy_kwargs={
            "net_arch": [args.hidden_dim, args.hidden_dim]  # 2-layer network
        }
    )

    model.set_logger(sb3_logger)
    logging.info("MaskablePPO model created with hyperparameters:")
    logging.info("  lr=%.2e, gamma=%.3f, lambda=%.3f, clip=%.2f",
                 args.lr, args.gamma, args.lam, args.clip)
    logging.info("  hidden_dim=%d, entropy_coef=%.3f, grad_clip=%.2f",
                 args.hidden_dim, args.entropy_coef, args.grad_clip)

    # Create callback
    callback = CustomCallback(
        gif_interval=args.gif_interval,
        log_dir=args.log_dir,
        num_agents=args.num_agents,
        verbose=1
    )

    # Calculate total timesteps based on episodes
    # Each episode is roughly (num_agents * steps_per_episode) environment steps
    total_timesteps = args.episodes * args.max_steps * args.num_agents
    logging.info("Starting training for %d total timesteps (~%d episodes)",
                 total_timesteps, args.episodes)

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Save final model
    model_path = os.path.join(args.log_dir, "final_model")
    model.save(model_path)
    logging.info("Model saved to: %s", model_path)

    # Print final statistics
    if callback.episode_count > 0:
        success_rate = 100.0 * callback.success_count / callback.episode_count
        logging.info(
            "Training complete! Final success rate: %.2f%% (%d/%d)",
            success_rate,
            callback.success_count,
            callback.episode_count
        )
        if len(callback.episode_steps) > 0:
            logging.info("Average steps per episode: %.2f", np.mean(callback.episode_steps))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OGM agents using SB3 MaskablePPO')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=100,
                        help='Approximate number of episodes to train')
    parser.add_argument('--num_agents', type=int, default=3,
                        help='Number of agents in the environment')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')

    # SB3-specific parameters
    parser.add_argument('--n_steps', type=int, default=2048,
                        help='Number of steps per rollout (SB3 parameter)')

    # PPO hyperparameters (matching your SimplePPOAgent)
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--clip', type=float, default=0.2,
                        help='PPO clip parameter')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of epochs per update')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--value_coef', type=float, default=0.5,
                        help='Value loss coefficient')
    parser.add_argument('--grad_clip', type=float, default=0.5,
                        help='Gradient clipping threshold')

    # Logging parameters
    parser.add_argument('--log_dir', type=str, default='runs/sb3',
                        help='Directory for logs and TensorBoard')
    parser.add_argument('--gif_interval', type=int, default=0,
                        help='Save an episode GIF every N episodes (0 disables)')

    # Soft matching reward parameters
    parser.add_argument('--enable_soft_matching_reward', action='store_true',
                        help='Enable soft pairwise matching reward')
    parser.add_argument('--soft_matching_decay_beta', type=float, default=0.999,
                        help='Time decay factor for soft matching reward')
    parser.add_argument('--soft_matching_scale', type=float, default=100.0,
                        help='Scaling factor for soft matching reward')

    # Bounty reward parameters
    parser.add_argument('--enable_bounty_reward', action='store_true',
                        help='Enable bounty reward system')
    parser.add_argument('--bounty_gamma', type=float, default=0.999,
                        help='Bounty decay factor')
    parser.add_argument('--bounty_eta', type=float, default=2.0,
                        help='Bounty multiplier coefficient')
    parser.add_argument('--bounty_base_value', type=float, default=1.0,
                        help='Base bounty value')
    parser.add_argument('--bounty_total_frac_of_success', type=float, default=0.2,
                        help='Total bounty as fraction of success bonus')
    parser.add_argument('--bounty_cap_per_step', type=float, default=20.0,
                        help='Maximum bounty reward per step')

    # Potential reward parameters
    parser.add_argument('--enable_potential_reward', action='store_true', default=True,
                        help='Enable potential-based shaping reward')
    parser.add_argument('--disable_potential_reward', action='store_false',
                        dest='enable_potential_reward',
                        help='Disable potential-based shaping reward')
    parser.add_argument('--potential_scale', type=float, default=1.0,
                        help='Scaling factor for potential reward')
    parser.add_argument('--potential_normalize', type=str, default='n2',
                        choices=['n2', 'none'], help='Normalization for potential reward')
    parser.add_argument('--success_bonus', type=float, default=100.0,
                        help='Bonus reward for successful completion')

    # Step cost parameters
    parser.add_argument('--step_cost_initial', type=float, default=-0.01,
                        help='Initial step cost')
    parser.add_argument('--step_cost_min', type=float, default=-0.001,
                        help='Minimum step cost')
    parser.add_argument('--use_exponential_decay', action='store_true', default=True,
                        help='Use exponential decay for step cost')

    args = parser.parse_args()

    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir = os.path.join(args.log_dir, f"n{args.num_agents}_{timestamp}")

    train(args)
