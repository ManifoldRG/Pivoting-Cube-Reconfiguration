import argparse 
import numpy as np
from ogm.ogm_env import OGMEnv
from agent.mappo_agent import MAPPOAgent
from ogm.random_configuration import random_configuration
import logging
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from visualizer.step_visualizer import StepVisualizer
from visualizer import visualize_position
from numpy.linalg import norm

def setup_logging(log_dir):
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
    logging.info("Logging initialized. TensorBoard directory: %s", log_dir)
    return writer


def train(args):
    writer = setup_logging(args.log_dir)

    init_conf, final_conf, grid_size = random_configuration(args.num_agents)
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
    )
    obs = env.reset(init_conf, final_conf)
    
    # Observation is two stacked (n x n) pairwise norms matrices; agent ctor multiplies by 2
    obs_dim = args.num_agents ** 2
    
    agent = MAPPOAgent(
        obs_dim, args.num_agents, action_dim=49, lr=args.lr,
        gamma=args.gamma, lam=args.lam, clip=args.clip,
        epochs=args.epochs, batch_size=args.batch_size,
        hidden_dim=args.hidden_dim, entropy_coef=args.entropy_coef, grad_clip=args.grad_clip,
        distance_temperature=args.distance_temperature, aux_coef=args.aux_coef
    )
    success_count = 0
    steps_per_episode = []
    
    for ep in range(args.episodes):
        init_conf, final_conf, _ = random_configuration(args.num_agents)
        obs = env.reset(init_conf, final_conf)
        done = False
        step = 0
        episode_reward = 0.0
        visualizer = None

        if args.gif_interval and (ep % args.gif_interval == 0):
            gif_name = os.path.join(args.log_dir, f"episode_{ep+1}.gif")
            visualizer = StepVisualizer(env.ogm, output_path=gif_name)
            # visualizer.set_final_state(final_conf)
            visualizer.capture_state()
            logging.info(f"Generating GIF for episode {ep+1}")


        # if visualizer:
        #     visualize_position.plot(init_conf, final_conf)


        while not done and step < args.max_steps:
            env.ogm.calc_pre_action_grid_map()
            phase_reward = 0.0

            # Sequential agent actions: 1..num_agents
            for aid in range(args.num_agents):
                moves = env.ogm.calc_possible_actions()
                mask = moves[aid + 1]
                current_obs = obs
                # Build candidate post-action pairwise norms (normalized) flattened
                post_norms = env.ogm.calc_post_pairwise_norms()
                grid_size = env.ogm.curr_grid_map.shape[0]
                max_dist = max(np.sqrt(3) * (grid_size - 1), 1.0)
                candidates_flat = []
                for act_id in range(1, 50):
                    if (aid + 1) in post_norms and act_id in post_norms[aid + 1]:
                        mat = post_norms[aid + 1][act_id] / max_dist
                        candidates_flat.append(mat.flatten())
                    else:
                        # placeholder; will be masked out
                        candidates_flat.append(np.zeros((args.num_agents, args.num_agents)).flatten())
                candidates_flat = np.array(candidates_flat, dtype=np.float32)

                action, log_prob = agent.select_action(current_obs, aid, candidates_flat, mask=mask)

                # Log Frobenius distance of chosen candidate to goal
                chosen_matrix_norm = candidates_flat[action].reshape(args.num_agents, args.num_agents)
                frob_to_goal = np.linalg.norm((env.ogm.final_pairwise_norms / max_dist) - chosen_matrix_norm, 'fro')
                writer.add_scalar("debug/frob_to_goal", frob_to_goal, step)

                obs, reward, done, _ = env.step((aid+1, action+1))
                agent.store(current_obs, aid, action, log_prob, reward, done, mask, candidates_flat)
                phase_reward = reward
                step += 1
                if visualizer:
                    visualizer.capture_state()
                if done or step >= args.max_steps:
                    break

            episode_reward += phase_reward
            if done or step >= args.max_steps:
                break

        # Capture final state after episode ends
        if visualizer:
            visualizer.capture_state()
            logging.info(f"Captured {len(visualizer.frames)} frames for episode {ep+1}")

        metrics = agent.update()
        writer.add_scalar("reward/episode", episode_reward, ep)

        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"loss/{k}", v, ep)
        
        if visualizer:
            visualizer.animate()
        
        actual_success = env.ogm.check_final() if hasattr(env, 'ogm') and env.ogm else False
        success_count += int(actual_success)
        steps_per_episode.append(step)
        
        if visualizer:
            if actual_success:
                logging.info(f"Episode {ep+1} succeeded - final state captured")
            else:
                logging.info(f"Episode {ep+1} failed - final state captured")

        logging.info(
            "Episode %d finished after %d steps -- reward: %.3f : success = %s", 
            ep + 1, 
            step, 
            episode_reward,
            actual_success
        )

    logging.info(
        "Success rate: %.2f%% (%d/%d)",
        100.0 * success_count / args.episodes,
        success_count,
        args.episodes,
    )
    logging.info("Average steps per episode: %.2f", np.mean(steps_per_episode))    

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--entropy_coef', type=float, default=0.02)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--distance_temperature', type=float, default=10.0)
    parser.add_argument('--aux_coef', type=float, default=0.1)
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for logs and TensorBoard')
    parser.add_argument('--gif_interval', type=int, default=0, help='Save an episode GIF every N episodes (0 disables)')
    # Reward config CLI switches
    parser.add_argument('--enable_bounty_reward', action='store_true')
    parser.add_argument('--bounty_gamma', type=float, default=0.999)
    parser.add_argument('--bounty_eta', type=float, default=2.0)
    parser.add_argument('--bounty_base_value', type=float, default=1.0)
    parser.add_argument('--bounty_total_frac_of_success', type=float, default=0.2)
    parser.add_argument('--bounty_cap_per_step', type=float, default=20.0)
    parser.add_argument('--enable_potential_reward', action='store_true', default=True)
    parser.add_argument('--potential_scale', type=float, default=1.0)
    parser.add_argument('--potential_normalize', type=str, default='n2', choices=['n2','none'])
    parser.add_argument('--success_bonus', type=float, default=100.0)
    args = parser.parse_args()
    train(args)
