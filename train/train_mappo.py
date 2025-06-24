import argparse 
import numpy as np
from ogm.ogm_env import OGMEnv
from agent.mappo_agent import MAPPOAgent
from ogm.random_configuration import random_configuration

def train(args):
    init_conf, final_conf, grid_size = random_configuration(args.num_agents)
    env = OGMEnv(step_cost=-0.01, max_steps=args.max_steps)
    obs = env.reset(init_conf, final_conf)
    obs_dim = grid_size ** 3
    agent = MAPPOAgent(obs_dim, args.num_agents, action_dim=48, lr=args.lr, 
                       gamma=args.gamma, lam=args.lam, clip=args.clip, 
                       epochs=args.epochs, batch_size=args.batch_size)
    
    for ep in range(args.episodes):
        init_conf, final_conf, _ = random_configuration(args.num_agents)
        obs = env.reset(init_conf, final_conf)
        done = False
        step = 0
        while not done and step < args.max_steps:
            for aid in range(args.num_agents):
                action, log_prob = agent.select_action(obs, aid)
                obs, reward, done, _ = env.step((aid+1, action+1))
                agent.store(obs, aid, action, log_prob, reward, done)
                step+=1
                if done or step >= args.max_steps:
                    break

        agent.update()
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1} finished after {step} steps")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--num_agents', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    train(args)
