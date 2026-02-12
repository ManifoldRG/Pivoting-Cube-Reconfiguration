"""
7-Agent Fixed Configuration Demo Training Script

Train MAPPO agents on a fixed 7-agent initial to final configuration.
Generates GIFs only on successful episodes, with parallel movement visualization.

Features:
- Fixed configurations from demo_configs_7.py
- Success-only GIF generation (exact position match, no rotation invariance)
- Parallel movement capture (one frame per timestep, not per agent)  
- 2x speed GIFs
- Ctrl+C graceful termination
- Continuous training until interrupted

Usage:
    python -m train.train_fixed_demo_7 --log_dir runs/demo7 --max_steps 1000
    
    Press Ctrl+C to stop and see summary.
"""

import argparse
import signal
import sys
import numpy as np
import logging
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from ogm.ogm_env import OGMEnv
from agent.mappo_agent import MAPPOAgent
from train.demo_configs_7 import get_configs, NUM_AGENTS
from visualizer.parallel_step_visualizer import ParallelStepVisualizer


# Global flag for graceful termination
interrupted = False
success_count = 0
total_episodes = 0


def check_exact_position_match(ogm):
    """Check if current positions exactly match final positions.
    
    Unlike check_final() which uses pairwise norms (rotation invariant),
    this checks that each module is at its exact target position.
    This ensures the blue cubes visually cover the red cubes.
    
    Args:
        ogm: OccupancyGridMap instance
        
    Returns:
        bool: True if all modules are at their exact target positions
    """
    for module_id in ogm.modules:
        current_pos = ogm.module_positions[module_id]
        target_pos = ogm.final_module_positions[module_id]
        if current_pos != target_pos:
            return False
    return True


def signal_handler(signum, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global interrupted
    print("\n\n[!] Ctrl+C received. Finishing current episode and exiting...")
    interrupted = True


def setup_logging(log_dir):
    """Initialize logging and TensorBoard."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )
    
    writer = SummaryWriter(log_dir=log_dir)
    logging.info("7-Agent Fixed Demo Training - Logging initialized")
    logging.info("TensorBoard directory: %s", log_dir)
    logging.info("Press Ctrl+C to stop training and generate summary")
    return writer


def train(args):
    """Main training loop with fixed configurations."""
    global interrupted, success_count, total_episodes
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    writer = setup_logging(args.log_dir)
    
    # Get fixed configurations
    init_conf, final_conf, grid_size = get_configs()
    num_agents = NUM_AGENTS
    
    logging.info("Configuration: %d agents", num_agents)
    logging.info("Initial: %s", init_conf)
    logging.info("Final: %s", final_conf)
    
    # Initialize environment
    env = OGMEnv(
        step_cost=-0.01,
        max_steps=args.max_steps,
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
        enable_soft_matching_reward=args.enable_soft_matching_reward,
        soft_matching_decay_beta=args.soft_matching_decay_beta,
        soft_matching_scale=args.soft_matching_scale,
    )
    
    # Initialize agent
    obs_dim = num_agents ** 2
    agent = MAPPOAgent(
        obs_dim, num_agents, action_dim=49, lr=args.lr,
        gamma=args.gamma, lam=args.lam, clip=args.clip,
        epochs=args.epochs, batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, entropy_coef=args.entropy_coef, 
        grad_clip=args.grad_clip,
        distance_temperature=args.distance_temperature, 
        aux_coef=args.aux_coef
    )
    
    steps_per_episode = []
    ep = 0
    
    logging.info("Starting training loop. Press Ctrl+C to stop.")
    
    # Main training loop - runs until interrupted
    while not interrupted:
        ep += 1
        total_episodes = ep
        
        # Reset with FIXED configurations (not random!)
        obs = env.reset(init_conf.copy(), final_conf.copy())
        done = False
        step = 0
        episode_reward = 0.0
        
        # Initialize visualizer for this episode
        visualizer = ParallelStepVisualizer(
            env.ogm, 
            output_path=os.path.join(args.log_dir, f"temp_episode_{ep}.gif"),
            fps=4  # 2x speed
        )
        visualizer.capture_parallel_step()  # Capture initial state
        
        while not done and step < args.max_steps and not interrupted:
            env.ogm.calc_pre_action_grid_map()
            phase_reward = 0.0
            
            # All agents take actions sequentially within one timestep
            for aid in range(num_agents):
                moves = env.ogm.calc_possible_actions()
                mask = moves[aid + 1]
                current_obs = obs
                
                # Compute candidate pairwise norms
                post_norms = env.ogm.calc_post_pairwise_norms()
                grid_size = env.ogm.curr_grid_map.shape[0]
                max_dist = max(np.sqrt(3) * (grid_size - 1), 1.0)
                
                candidates_flat = []
                for act_id in range(1, 50):
                    if (aid + 1) in post_norms and act_id in post_norms[aid + 1]:
                        mat = post_norms[aid + 1][act_id] / max_dist
                        candidates_flat.append(mat.flatten())
                    else:
                        candidates_flat.append(np.zeros((num_agents, num_agents)).flatten())
                candidates_flat = np.array(candidates_flat, dtype=np.float32)
                
                action, log_prob = agent.select_action(
                    current_obs, aid, candidates_flat=candidates_flat, mask=mask
                )
                
                obs, reward, done, _ = env.step((aid + 1, action + 1))
                agent.store(current_obs, aid, action, log_prob, reward, done, mask, candidates_flat)
                phase_reward = reward
                step += 1
                
                if done or step >= args.max_steps:
                    break
            
            # Capture state ONCE after all agents have moved (parallel visualization)
            visualizer.capture_parallel_step()
            
            episode_reward += phase_reward
            if done or step >= args.max_steps:
                break
        
        # Check for success using EXACT position match (not rotation invariant)
        # This ensures blue cubes exactly cover red cubes in the GIF
        actual_success = check_exact_position_match(env.ogm) if hasattr(env, 'ogm') and env.ogm else False
        
        # Only save GIF on success
        if actual_success:
            success_count += 1
            gif_path = os.path.join(args.log_dir, f"success_{success_count}.gif")
            visualizer.output_path = gif_path
            visualizer.animate(pause_frames=20, success=True)
            logging.info(f"[SUCCESS #{success_count}] Episode {ep} - GIF saved: {gif_path}")
        else:
            # Clean up temp visualizer without saving
            logging.info(
                f"Episode {ep} finished after {step} steps -- reward: {episode_reward:.3f} : success = False"
            )
        
        # Update agent
        metrics = agent.update()
        
        # Log metrics
        writer.add_scalar("reward/episode", episode_reward, ep)
        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"loss/{k}", v, ep)
        
        steps_per_episode.append(step)
        
        # Periodic logging
        if ep % 10 == 0:
            success_rate = 100.0 * success_count / ep
            logging.info(
                f"Progress: Episode {ep} | Success rate: {success_rate:.1f}% ({success_count}/{ep})"
            )
    
    # Final summary after Ctrl+C
    print("\n" + "=" * 60)
    print("7-AGENT TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {total_episodes}")
    print(f"Successful episodes: {success_count}")
    print(f"Success rate: {100.0 * success_count / max(total_episodes, 1):.2f}%")
    print(f"Average steps per episode: {np.mean(steps_per_episode):.2f}")
    print(f"GIFs saved: {success_count}")
    print(f"Output directory: {args.log_dir}")
    print("=" * 60)
    
    logging.info("Training completed. %d success GIFs generated.", success_count)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train MAPPO on fixed 7-agent configuration"
    )
    
    # Core parameters
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode (more for 7 agents)')
    parser.add_argument('--log_dir', type=str, default='runs/demo7',
                        help='Directory for logs, TensorBoard, and GIFs')
    
    # Learning parameters
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    
    # MAPPO-specific
    parser.add_argument('--distance_temperature', type=float, default=10.0)
    parser.add_argument('--aux_coef', type=float, default=0.1)
    
    # Reward configuration
    parser.add_argument('--enable_bounty_reward', action='store_true')
    parser.add_argument('--bounty_gamma', type=float, default=0.999)
    parser.add_argument('--bounty_eta', type=float, default=2.0)
    parser.add_argument('--bounty_base_value', type=float, default=1.0)
    parser.add_argument('--bounty_total_frac_of_success', type=float, default=0.2)
    parser.add_argument('--bounty_cap_per_step', type=float, default=20.0)
    
    parser.add_argument('--enable_potential_reward', action='store_true', default=True)
    parser.add_argument('--disable_potential_reward', action='store_false', 
                        dest='enable_potential_reward')
    parser.add_argument('--potential_scale', type=float, default=1.0)
    parser.add_argument('--potential_normalize', type=str, default='n2', 
                        choices=['n2', 'none'])
    parser.add_argument('--success_bonus', type=float, default=100.0)
    
    # Step cost decay
    parser.add_argument('--step_cost_initial', type=float, default=-0.01)
    parser.add_argument('--step_cost_min', type=float, default=-0.001)
    parser.add_argument('--use_exponential_decay', action='store_true', default=True)
    
    # Soft matching reward
    parser.add_argument('--enable_soft_matching_reward', action='store_true')
    parser.add_argument('--soft_matching_decay_beta', type=float, default=0.999)
    parser.add_argument('--soft_matching_scale', type=float, default=100.0)
    
    args = parser.parse_args()
    train(args)
