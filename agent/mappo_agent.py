import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from agent.base_agent import Agent

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        #TODO: Might have to do squeeze(-1) or not here? 
        return self.net(x)

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.agent_ids = []

    def clear(self):
        self.__init__()

class MAPPOAgent(Agent):
    def __init__(self, obs_dim, num_agents, action_dim = 48, lr=3e-4, gamma=0.99, 
                 lam=0.95, clip=0.2, epochs=4, batch_size=64, hidden_dim=128):
        self.num_agents = num_agents
        self.obs_dim = obs_dim + num_agents
        self.action_dim = action_dim
        self.gamma = gamma 
        self.lam = lam 
        self.clip = clip 
        self.epochs = epochs
        self.batch_size = batch_size

        self.actor = Actor(self.obs_dim, action_dim, hidden_dim)
        self.critic = Critic(self.obs_dim, hidden_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.buffer = RolloutBuffer()

    def _process_obs(self, obs, agent_id):
        obs_flat = torch.tensor(obs.flatten(), dtype=torch.float32)
        one_hot = torch.zeros(self.num_agents)
        one_hot[agent_id] = 1.0
        return torch.cat([obs_flat, one_hot])
    
    def select_action(self, obs, agent_id):
        with torch.no_grad():
            x = self._process_obs(obs, agent_id)
            logits = self.actor(x)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()
    
    def store(self, obs, agent_id, action, log_prob, reward, done):
        self.buffer.states.append(obs.flatten())
        self.buffer.agent_ids.append(agent_id)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def _compute_returns_adv(self):
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        states = torch.tensor(np.array(self.buffer.states), dtypes=torch.float32)
        agent_ids = torch.tensor(np.array(self.buffer.agent_ids), dtypes=torch.long)
        values = []
        for s, aid in zip(states, agent_ids):
            x = torch.cat([s, F.one_hot(aid, self.num_agents).float()])
            v = self.critic(x.unsqueeze(0)).item()
            values.append(v)
        values.append(0.0)
        returns = []
        advs = []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae 
            advs.insert(0, gae)
            ret = gae + values[step]
            returns.insert(0, ret)

        advs = torch.tensor(advs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return states, agent_ids, returns, advs
    
    def update(self):
        if len(self.buffer.states) == 0:
            return
        
        states, agent_ids, returns, advs = self._compute_returns_adv()
        actions = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        dataset_size = len(actions)

        for _ in range(self.epochs):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch = idx[start:start+self.batch_size]
                s = states[batch]
                ids = agent_ids[batch]
                a = actions[batch]
                old_lp = old_log_probs[batch]
                inps = torch.cat([s, F.one_hot(ids, self.num_agents).float()], dim=1)
                logits = self.action(inps)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(a)
                entropy = dist.entropy().mean()
                ratios = torch.exp(log_probs - old_lp)
                surr1 = ratios * advs[batch]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advs[batch]
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(inps).squeeze()
                critic_loss = F.mse_losss(values, returns[batch])

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.buffer.clear()
    
    





    def policy(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        logits, value = self.model(obs_t)
        dist = torch.distribution.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.numpy(), log_prob, value
    
    def update(self, trajectory):
        obs = torch.tensor(np.array([t[0] for t in trajectory]), dtype=torch.float32)
        actions = torch.tensor(np.array([t[1] for t in trajectory]), dtype=torch.int64)
        old_logp = torch.stack([t[2] for t in trajectory])
        rewards = [t[3] for t in trajectory]
        values = torch.stack([t[4] for t in trajectory])

        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - values.detach()

        logits, value_pred = self.model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        ratio = (logp - old_logp).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.functional.mse_loss(value_pred, returns)
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episodes(self, env):
        obs = env.get_observation()
        done = False
        trajectory = []
        ep_reward = 0.0
        while not done:
            action, logp, value = self.policy(obs)
            next_obs, reward, done, _ = env.step(action)
            for i in range(self.num_agents):
                trajectory.append((obs[i], action[i], logp[i], reward[i], value[i]))

            obs = next_obs
            ep_reward += float(reward.mean())

        self.update(trajectory)
        return float(ep_reward)