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
    env = OGMEnv(step_cost=args.step_cost, max_steps=args.max_steps, 
                 enable_bounty_reward=args.enable_bounty_reward, 
                 bounty_gamma=args.bounty_gamma, bounty_eta=args.bounty_eta, bounty_base_value=args.bounty_base_value,
                 enable_potential_reward=args.enable_potential_reward,
                 potential_scale=args.potential_scale,
                 potential_normalize=args.potential_normalize,
                 success_bonus=args.success_bonus)
    obs = env.reset(init_conf, final_conf)
    
    # ---- NEW OBSERVATION DIM CALCULATION ----
    num_pairs = args.num_agents * (args.num_agents - 1) // 2
    obs_dim = num_pairs
    # -----------------------------------------
    
    agent = MAPPOAgent(obs_dim, args.num_agents, action_dim=49, lr=args.lr, 
                       gamma=args.gamma, lam=args.lam, clip=args.clip, 
                       epochs=args.epochs, batch_size=args.batch_size)
    success_count = 0
    steps_per_episode = []
    bounty_rewards_per_episode = []
    bounties_collected_per_episode = []
    
    for ep in range(args.episodes):
        init_conf, final_conf, _ = random_configuration(args.num_agents)
        obs = env.reset(init_conf, final_conf)
        done = False
        step = 0
        episode_reward = 0.0
        episode_bounty_reward = 0.0
        episode_bounties_collected = 0
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
            random_queue = env.ogm.calc_queue()
            env.ogm.calc_pre_action_grid_map()
            phase_reward = 0.0

            for aid in random_queue:
                aid = aid - 1
                mask = env.ogm.calc_possible_actions()[aid + 1]
                current_obs = obs
                action, log_prob = agent.select_action(current_obs, aid, mask=mask)
                obs, reward, done, info = env.step((aid+1, action+1))
                agent.store(current_obs, aid, action, log_prob, reward, done, mask)
                phase_reward = reward
                
                # Track bounty metrics
                episode_bounty_reward += info.get('bounty_reward', 0.0)
                episode_bounties_collected += info.get('bounties_collected', 0)
                if done or step >= args.max_steps: 
                    break

            episode_reward += phase_reward
            step += 1 
            if visualizer:
                visualizer.capture_state()
            if done or step >= args.max_steps:
                break

        # Capture final state after episode ends
        if visualizer:
            visualizer.capture_state()
            logging.info(f"Captured {len(visualizer.frames)} frames for episode {ep+1}")

        metrics = agent.update()
        writer.add_scalar("reward/episode", episode_reward, ep)
        writer.add_scalar("bounty/reward_per_episode", episode_bounty_reward, ep)
        writer.add_scalar("bounty/collected_per_episode", episode_bounties_collected, ep)

        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"loss/{k}", v, ep)
        
        if visualizer:
            visualizer.animate()
        
        actual_success = env.ogm.check_final() if hasattr(env, 'ogm') and env.ogm else False
        success_count += int(actual_success)
        steps_per_episode.append(step)
        bounty_rewards_per_episode.append(episode_bounty_reward)
        bounties_collected_per_episode.append(episode_bounties_collected)
        
        if visualizer:
            if actual_success:
                logging.info(f"Episode {ep+1} succeeded - final state captured")
            else:
                logging.info(f"Episode {ep+1} failed - final state captured")

        logging.info(
            "Episode %d finished after %d steps -- reward: %.3f (bounty: %.3f, collected: %d) : success = %s", 
            ep + 1, 
            step, 
            episode_reward,
            episode_bounty_reward,
            episode_bounties_collected,
            actual_success
        )

    logging.info(
        "Success rate: %.2f%% (%d/%d)",
        100.0 * success_count / args.episodes,
        success_count,
        args.episodes,
    )
    logging.info("Average steps per episode: %.2f", np.mean(steps_per_episode))
    logging.info("Average bounty reward per episode: %.3f", np.mean(bounty_rewards_per_episode))
    logging.info("Average bounties collected per episode: %.2f", np.mean(bounties_collected_per_episode))    

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=500)
    # PPO
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for logs and TensorBoard')
    parser.add_argument('--gif_interval', type=int, default=0, help='Save an episode GIF every N episodes (0 disables)')
    # Reward toggles / hyperparameters
    parser.add_argument('--enable_bounty_reward', action='store_true', default=False)
    parser.add_argument('--bounty_gamma', type=float, default=0.999)
    parser.add_argument('--bounty_eta', type=float, default=2.0)
    parser.add_argument('--bounty_base_value', type=float, default=1.0)
    parser.add_argument('--enable_potential_reward', action='store_true', default=True)
    parser.add_argument('--potential_scale', type=float, default=1.0)
    parser.add_argument('--potential_normalize', type=str, choices=['n2','max'], default='n2')
    parser.add_argument('--success_bonus', type=float, default=100.0)
    parser.add_argument('--step_cost', type=float, default=-0.01)
    args = parser.parse_args()
    train(args)
