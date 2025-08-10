import logging
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
        logging.debug(
            "Actor initialized with input_dim=%d hidden_dim=%d action_dim=%d",
            input_dim,
            hidden_dim,
            action_dim
        )
        # print(input_dim, hidden_dim, action_dim)
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
        self.action_masks = []

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
    
    def select_action(self, obs, agent_id, mask=None):
        with torch.no_grad():
            x = self._process_obs(obs, agent_id)
            logits = self.actor(x)
            if mask is not None:
                m = torch.tensor(mask, dtype=torch.bool)
                logits = logits.masked_fill(~m, -1e9)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()
    
    def store(self, obs, agent_id, action, log_prob, reward, done, mask):
        self.buffer.states.append(obs.flatten())
        self.buffer.agent_ids.append(agent_id)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.action_masks.append(mask)
    
    def _compute_returns_adv(self):
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32)
        agent_ids = torch.tensor(np.array(self.buffer.agent_ids), dtype=torch.long)
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
        masks = self.buffer.action_masks
        return states, agent_ids, returns, advs, masks
    
    def update(self):
        if len(self.buffer.states) == 0:
            return {}
        
        states, agent_ids, returns, advs, masks = self._compute_returns_adv()
        actions = torch.tensor(self.buffer.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.bool)
        dataset_size = len(actions)


        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        batches = 0

        for _ in range(self.epochs):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch = idx[start:start+self.batch_size]
                s = states[batch]
                ids = agent_ids[batch]
                a = actions[batch]
                old_lp = old_log_probs[batch]
                inps = torch.cat([s, F.one_hot(ids, self.num_agents).float()], dim=1)
                logits = self.actor(inps)
                mask = masks[batch]
                logits = logits.masked_fill(~mask, -1e9)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(a)
                entropy = dist.entropy().mean()
                ratios = torch.exp(log_probs - old_lp)
                surr1 = ratios * advs[batch]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advs[batch]
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(inps).squeeze()
                critic_loss = F.mse_loss(values, returns[batch])

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                batches += 1

        self.buffer.clear()

        if batches == 0:
            return {}
        
        return {
            "actor_loss": total_actor_loss / batches,
            "critic_loss": total_critic_loss / batches, 
            "entropy": total_entropy / batches,
        }
    
    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])