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
    env = OGMEnv(step_cost=-0.01, max_steps=args.max_steps)
    obs = env.reset(init_conf, final_conf)
    obs_dim = grid_size ** 3
    agent = MAPPOAgent(obs_dim, args.num_agents, action_dim=48, lr=args.lr, 
                       gamma=args.gamma, lam=args.lam, clip=args.clip, 
                       epochs=args.epochs, batch_size=args.batch_size)
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


        # if visualizer:
        #     visualize_position.plot(init_conf, final_conf)


        while not done and step < args.max_steps:
            for aid in range(0, args.num_agents):
                mask = env.ogm.calc_possible_actions()[aid + 1]
                action, log_prob = agent.select_action(obs, aid, mask=mask)
                # action, log_prob = agent.select_action(obs, aid)
                obs, reward, done, _ = env.step((aid+1, action+1))
                agent.store(obs, aid, action, log_prob, reward, done, mask)
                episode_reward += reward
                step+=1
                if visualizer:
                    visualizer.capture_state()
                if done or step >= args.max_steps:
                    break

        metrics = agent.update()
        writer.add_scalar("reward/episode", episode_reward, ep)

        if metrics:
            for k, v in metrics.items():
                writer.add_scalar(f"loss/{k}", v, ep)

        if visualizer:
            visualizer.animate()
        
        success = done if env.steps_taken < env.max_steps else False
        success_count += int(success)
        steps_per_episode.append(step)
        logging.info(
            "Episode %d finished after %d steps -- reward: %.3f : success = %s", 
            ep + 1, 
            step, 
            episode_reward,
            success
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
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for logs and TensorBoard')
    parser.add_argument('--gif_interval', type=int, default=0, help='Save an episode GIF every N episodes (0 disables)')
    args = parser.parse_args()
    train(args)
