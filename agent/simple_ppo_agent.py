import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from agent.base_agent import Agent

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        logging.debug(
            "Actor initialized with input_dim=%d hidden_dim=%d output_dim=%d",
            input_dim,
            hidden_dim,
            output_dim
        )
        # Using 3 layers for better capacity without being too deep
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer norm for stability
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
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

class SimplePPOAgent(Agent):
    def __init__(self, obs_dim, num_agents, action_dim=49, lr=3e-4, gamma=0.99, 
                 lam=0.95, clip=0.2, epochs=4, batch_size=64, hidden_dim=256,
                 entropy_coef=0.01, grad_clip=0.5, value_coef=0.5):
        """
        Simple PPO agent with direct discrete action output.
        
        Hyperparameter choices:
        - hidden_dim=256: Smaller than original (768) - start conservative
        - lr=3e-4: Standard PPO learning rate
        - entropy_coef=0.01: Lower than original (0.02) for more exploitation
        - grad_clip=0.5: Always use gradient clipping for stability
        - value_coef=0.5: Standard PPO value loss coefficient
        - batch_size=64: Keep same for fair comparison
        - epochs=4: Standard PPO epochs
        """
        self.num_agents = num_agents
        self.obs_dim = (obs_dim * 2) + num_agents  # Two pairwise matrices + one-hot
        self.action_dim = action_dim
        self.gamma = gamma 
        self.lam = lam 
        self.clip = clip 
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.value_coef = value_coef

        # Simple direct output: logits for 49 actions
        self.actor = Actor(self.obs_dim, self.action_dim, hidden_dim)
        self.critic = Critic(self.obs_dim, hidden_dim)
        
        # Separate optimizers sometimes help, but using single for simplicity
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=lr,
            eps=1e-5  # Small epsilon for numerical stability
        )
        
        self.buffer = RolloutBuffer()

    def _process_obs(self, obs, agent_id):
        """Convert observation and agent ID to network input."""
        obs_flat = torch.tensor(obs.ravel(), dtype=torch.float32)
        one_hot = torch.zeros(self.num_agents)
        one_hot[agent_id] = 1.0
        return torch.cat([obs_flat, one_hot])
    
    def select_action(self, obs, agent_id, mask=None):
        """
        Select action using masked categorical distribution.
        Much simpler than distance-based selection.
        """
        with torch.no_grad():
            x = self._process_obs(obs, agent_id)
            logits = self.actor(x)
            
            # Apply mask
            if mask is not None:
                m = torch.tensor(mask, dtype=torch.bool)
                logits = logits.masked_fill(~m, -1e9)
            
            # Standard categorical distribution
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()
    
    def store(self, obs, agent_id, action, log_prob, reward, done, mask):
        """Store transition in buffer."""
        self.buffer.states.append(obs.flatten())
        self.buffer.agent_ids.append(agent_id)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.action_masks.append(mask)
    
    def _compute_returns_adv(self):
        """Compute returns and advantages using GAE."""
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32)
        agent_ids = torch.tensor(np.array(self.buffer.agent_ids), dtype=torch.long)
        
        # Compute values for all states
        values = []
        with torch.no_grad():
            for s, aid in zip(states, agent_ids):
                x = torch.cat([s, F.one_hot(aid, self.num_agents).float()])
                v = self.critic(x.unsqueeze(0)).item()
                values.append(v)
        values.append(0.0)  # Bootstrap value for terminal state
        
        # GAE computation
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
        
        # Normalize advantages
        if len(advs) > 1:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        masks = self.buffer.action_masks
        return states, agent_ids, returns, advs, masks
    
    def update(self):
        """PPO update with multiple epochs over the buffer."""
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
        total_value_loss = 0.0
        batches = 0

        for _ in range(self.epochs):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.batch_size):
                batch = idx[start:start+self.batch_size]
                s = states[batch]
                ids = agent_ids[batch]
                a = actions[batch]
                old_lp = old_log_probs[batch]
                mask = masks[batch]
                
                # Build input
                inps = torch.cat([s, F.one_hot(ids, self.num_agents).float()], dim=1)
                
                # Actor forward pass
                logits = self.actor(inps)
                logits = logits.masked_fill(~mask, -1e9)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                log_probs = dist.log_prob(a)
                entropy = dist.entropy().mean()
                
                # PPO clipped loss
                ratios = torch.exp(log_probs - old_lp)
                surr1 = ratios * advs[batch]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advs[batch]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                values = self.critic(inps).squeeze(-1)
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                critic_loss = F.mse_loss(values, returns[batch])

                # Combined loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.grad_clip
                    )
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
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])