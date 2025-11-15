"""
Gymnasium wrapper for OGMEnv to work with Stable-Baselines3.
Supports action masking via sb3-contrib's MaskablePPO.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ogm.ogm_env import OGMEnv
from ogm.random_configuration import random_configuration


class OGMGymEnv(gym.Env):
    """
    Gym wrapper for the OGM environment with sequential multi-agent control.

    This wrapper handles the sequential agent execution internally, presenting
    a single-agent interface to SB3. Each step, one agent acts, and the
    observation includes agent identity via the obs structure.
    """
    metadata = {'render_modes': []}

    def __init__(self, num_agents=3, max_steps=500, **env_kwargs):
        super().__init__()

        self.num_agents = num_agents
        self.max_steps = max_steps

        # Create the underlying OGM environment with all our custom rewards
        # OGMEnv counts individual agent actions, so multiply by num_agents
        # to get the desired number of "phases" (all agents act once = 1 phase)
        ogm_max_steps = max_steps * num_agents if max_steps is not None else None
        self.env = OGMEnv(max_steps=ogm_max_steps, **env_kwargs)

        self.action_space = spaces.Discrete(49)

        # Observation space: flattened (2, n, n) pairwise norms + one-hot agent ID
        # our agent expects: obs.flatten() + one_hot(agent_id)
        obs_dim = 2 * num_agents * num_agents + num_agents
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state
        self.current_agent_idx = 0  # Which agent is acting (0-indexed)
        self.raw_obs = None  # The (2, n, n) observation from env
        self.episode_done = False
        self.steps_taken = 0
        self.init_conf = None
        self.final_conf = None

    def _get_obs(self):
        """Convert raw obs + agent ID into the format our model expects."""
        # Flatten the (2, n, n) observation
        obs_flat = self.raw_obs.flatten()

        # Create one-hot encoding for current agent
        agent_onehot = np.zeros(self.num_agents, dtype=np.float32)
        agent_onehot[self.current_agent_idx] = 1.0

        # Concatenate: [obs_flat, agent_onehot]
        return np.concatenate([obs_flat, agent_onehot])

    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation."""
        super().reset(seed=seed)

        # Generate random configuration
        self.init_conf, self.final_conf, _ = random_configuration(self.num_agents)

        # Reset the OGM environment
        self.raw_obs = self.env.reset(self.init_conf, self.final_conf)

        # Reset internal state
        self.current_agent_idx = 0
        self.episode_done = False
        self.steps_taken = 0

        # Prepare grid map for first agent
        self.env.ogm.calc_pre_action_grid_map()

        obs = self._get_obs()
        info = {}

        return obs, info

    def action_masks(self):
        """
        Return boolean mask of valid actions for the current agent.
        Required by MaskablePPO.
        """
        if self.env.ogm is None:
            return np.ones(49, dtype=bool)

        # Get possible moves for current agent (1-indexed in OGM)
        moves = self.env.ogm.calc_possible_actions()
        mask = moves[self.current_agent_idx + 1]  # Convert to 1-indexed

        return np.array(mask, dtype=bool)

    def step(self, action):
        """
        Execute one agent's action. When all agents have acted, increment step counter.
        """
        if self.episode_done:
            # Episode already finished, this shouldn't happen
            return self._get_obs(), 0.0, True, False, {}

        # Convert action from 0-indexed to 1-indexed for OGM
        action_id = int(action) + 1
        agent_id = self.current_agent_idx + 1  # 1-indexed for OGM

        # Take action in the environment (returns your custom rewards)
        self.raw_obs, reward, done, info = self.env.step((agent_id, action_id))

        # Move to next agent
        self.current_agent_idx += 1

        # Check if all agents have acted (one full "phase")
        if self.current_agent_idx >= self.num_agents:
            self.current_agent_idx = 0
            self.steps_taken += 1

            # Prepare grid map for next phase
            if not done:
                self.env.ogm.calc_pre_action_grid_map()

        # Check termination conditions
        terminated = done or (self.steps_taken >= self.max_steps)
        self.episode_done = terminated

        # Gymnasium API: (obs, reward, terminated, truncated, info)
        truncated = (self.steps_taken >= self.max_steps) and not self.env.ogm.check_final()

        obs = self._get_obs()
        info['episode_step'] = self.steps_taken
        info['current_agent'] = self.current_agent_idx

        return obs, float(reward), terminated, truncated, info


def make_ogm_env(num_agents=3, **kwargs):
    """
    Factory function to create OGM gym environment.
    """
    return OGMGymEnv(num_agents=num_agents, **kwargs)
