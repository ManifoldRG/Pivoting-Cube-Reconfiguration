"""
Gymnasium wrapper for OGMEnv to work with Stable-Baselines3.
Supports action masking via sb3-contrib's MaskablePPO.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ogm.ogm_env import OGMEnv
from ogm.random_configuration import random_configuration

# Fixed-size agent encoding dimension (constant regardless of n)
AGENT_ENCODING_DIM = 4


class OGMGymEnv(gym.Env):
    """
    Gym wrapper for the OGM environment with sequential multi-agent control.

    This wrapper handles the sequential agent execution internally, presenting
    a single-agent interface to SB3. Each step, one agent acts, and the
    observation includes agent identity via the obs structure.
    """

    metadata = {"render_modes": []}

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

        # Calculate observation dimension based on reduction settings
        use_four_band = env_kwargs.get("use_four_band_reduction", False)
        use_local_neighborhood = env_kwargs.get("use_local_neighborhood", False)
        local_k = env_kwargs.get("local_neighborhood_k", 3)

        # For curriculum learning: observation size must be CONSTANT across all n values
        # When both reductions are enabled, obs size = local_k × 4 (always)
        if use_four_band and use_local_neighborhood:
            # Fixed observation size: k rows × 4 bands
            # This enables transfer learning across different n values
            effective_rows = local_k
            matrix_cols = 4
        elif use_four_band:
            # Four-band always produces (n, 4) shape now
            effective_rows = num_agents
            matrix_cols = 4
        elif use_local_neighborhood:
            # Local neighborhood only
            effective_rows = min(local_k, num_agents)
            matrix_cols = num_agents
        else:
            # No reduction - full n×n matrix
            effective_rows = num_agents
            matrix_cols = num_agents

        obs_base_dim = effective_rows * matrix_cols

        # Add fixed-size agent encoding (constant 4 dims instead of n-dim one-hot)
        obs_dim = obs_base_dim + AGENT_ENCODING_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state
        self.current_agent_idx = 0  # Which agent is acting (0-indexed)
        self.raw_obs = (
            None  # Raw observation from env (shape depends on reduction settings)
        )
        self.episode_done = False
        self.steps_taken = 0
        self.init_conf = None
        self.final_conf = None
        self.use_four_band = use_four_band
        self.use_local_neighborhood = use_local_neighborhood
        self.local_k = local_k

    def _get_agent_encoding(self, agent_idx):
        """
        Fixed-size agent encoding (4 dims) - constant regardless of num_agents.
        Uses normalized position + sinusoidal encoding for smooth interpolation.
        """
        # Normalized position in [0, 1]
        pos = agent_idx / max(self.num_agents - 1, 1)

        # Sinusoidal encoding (like transformer positional encoding)
        encoding = np.array(
            [
                pos,  # Linear position
                np.sin(pos * np.pi),  # Sin encoding
                np.cos(pos * np.pi),  # Cos encoding
                np.sin(pos * 2 * np.pi),  # Higher frequency
            ],
            dtype=np.float32,
        )

        return encoding

    def _get_obs(self):
        """Convert raw obs + agent ID into the format our model expects."""
        obs = self.raw_obs

        # Apply local neighborhood reduction if enabled
        if self.use_local_neighborhood and self.env.ogm is not None:
            # Agent IDs are 1-indexed in OGM
            obs = self.env.ogm.calc_local_neighborhood_reduction(
                obs, self.current_agent_idx + 1, self.local_k
            )

        # Flatten the observation
        obs_flat = obs.flatten().astype(np.float32)

        # Pad to fixed size for curriculum learning (when n < local_k)
        # Expected size: local_k × 4 bands when both reductions enabled
        if self.use_four_band and self.use_local_neighborhood:
            expected_size = self.local_k * 4
            if len(obs_flat) < expected_size:
                # Pad with zeros for smaller n values
                padding = np.zeros(expected_size - len(obs_flat), dtype=np.float32)
                obs_flat = np.concatenate([obs_flat, padding])

        # Fixed-size agent encoding (4 dims instead of n-dim one-hot)
        agent_encoding = self._get_agent_encoding(self.current_agent_idx)

        # Concatenate: [obs_flat, agent_encoding]
        return np.concatenate([obs_flat, agent_encoding])

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
        truncated = (
            self.steps_taken >= self.max_steps
        ) and not self.env.ogm.check_final()

        obs = self._get_obs()
        info["episode_step"] = self.steps_taken
        info["current_agent"] = self.current_agent_idx
        info["max_steps"] = self.max_steps  # For callback truncation detection

        return obs, float(reward), terminated, truncated, info


def make_ogm_env(num_agents=3, **kwargs):
    """
    Factory function to create OGM gym environment.
    """
    return OGMGymEnv(num_agents=num_agents, **kwargs)
