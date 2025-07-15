import numpy as np
import pettingzoo
from pettingzoo.utils import parallel_to_aec, wrappers
from gymnasium import spaces

from ogm.occupancy_grid_map import OccupancyGridMap

class PivotingCubesEnv(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "pivoting_cubes_v0"}

    def __init__(self, initial_positions, final_positions, n_modules, empathy_lambda=0.0, max_episode_steps=200):
        """
        The constructor for the environment.
        """
        self.ogm = OccupancyGridMap(initial_positions, final_positions, n_modules)
        
        self.agents = [f"module_{i}" for i in range(1, n_modules + 1)]
        self.possible_agents = self.agents[:]
        self.n_modules = n_modules
        self.empathy_lambda = empathy_lambda
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0
        
        self._define_spaces()

    def _define_spaces(self):
        # Action space: 48 pivots + 1 NO-OP action
        self.action_spaces = {
            agent: spaces.Discrete(49) for agent in self.agents
        }

        # Observation space: A dictionary containing the agent's local grid
        # and a mask of legal actions.
        self.observation_spaces = {
            agent: spaces.Dict({
                # The 5x5x5 local map around the agent
                "observation": spaces.Box(low=0, high=self.n_modules, shape=(5, 5, 5), dtype=np.int8),
                # A binary mask for legal actions
                "action_mask": spaces.Box(low=0, high=1, shape=(49,), dtype=np.int8)
            }) for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        # Re-initialize the underlying OGM simulation
        self.ogm = OccupancyGridMap(
            self.ogm.original_module_positions,
            self.ogm.original_final_module_positions,
            self.n_modules
        )
        self.agents = [f"module_{i}" for i in range(1, self.n_modules + 1)]
        self.episode_step = 0

        # Get initial observations and infos
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        grid_map_t = self.ogm.curr_grid_map.copy()

        proposed_moves = {}
        target_positions = {}

        for agent_name, action in actions.items():
            if action == 0:  # NO-OP
                continue
            module_id = int(agent_name.split('_')[1])
            new_pos = self.ogm._compute_new_position(self.ogm.module_positions[module_id], action)
            if new_pos in target_positions:
                # Both moves fail. The first agent that claimed the spot also fails.
                conflicting_agent_id = target_positions[new_pos]
                if conflicting_agent_id in proposed_moves:
                    del proposed_moves[conflicting_agent_id]
            else:
                target_positions[new_pos] = module_id
                proposed_moves[module_id] = new_pos

        # validate connectivity
        if proposed_moves:
            future_positions = self.ogm.module_positions.copy()
            future_positions.update(proposed_moves)
            if not self.ogm.is_connected(future_positions):
                # the set of moves is invalid because it breaks the structure.
                # reject all moves for this timestep by clearing the dictionary.
                proposed_moves = {}

        # Execute valid, non-conflicting moves
        self.ogm.execute_moves(proposed_moves)

        # calc results
        terminations = {agent: self.ogm.check_final() for agent in self.agents}
        self.episode_step += 1
        truncations = {agent: False for agent in self.agents}
        if self.episode_step >= self.max_episode_steps:
            truncations = {agent: True for agent in self.agents}
            self.agents = []
        rewards = self._get_rewards(grid_map_t)
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        
        # if any agent terminates, the episode is over for all
        if any(terminations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self):
        # First, calculate all possible actions for the current state
        available_actions = self.ogm.calc_possible_actions()
        
        observations = {}
        for agent_name in self.agents:
            module_id = int(agent_name.split('_')[1])
            
            # Action Mask (always allow NO-OP)
            action_mask = np.zeros(49, dtype=np.int8)
            action_mask[0] = 1 
            legal_pivots = np.where(available_actions[module_id])[0]
            action_mask[legal_pivots + 1] = 1
            
            local_map = self.ogm.get_local_map(module_id, patch_size=5)

            observations[agent_name] = {
                "observation": local_map,
                "action_mask": action_mask
            }
        return observations

    def _get_rewards(self, grid_map_t):
        rewards = {}
        local_maps_t = {}
        local_maps_tp1 = {}
        final_local_maps = {}
        positions = {}
        for agent_name in self.agents:
            module_id = int(agent_name.split('_')[1])
            positions[agent_name] = self.ogm.module_positions[module_id]
            pos = positions[agent_name]
            half = 2
            x, y, z = pos
            x_min = max(x - half, 0)
            x_max = min(x + half + 1, grid_map_t.shape[0])
            y_min = max(y - half, 0)
            y_max = min(y + half + 1, grid_map_t.shape[1])
            z_min = max(z - half, 0)
            z_max = min(z + half + 1, grid_map_t.shape[2])
            local_map_t = np.zeros((5, 5, 5), dtype=np.int8)
            x_slice = slice(x_min, x_max)
            y_slice = slice(y_min, y_max)
            z_slice = slice(z_min, z_max)
            local_map_t[
                (x_min - (x - half)):(x_max - (x - half)),
                (y_min - (y - half)):(y_max - (y - half)),
                (z_min - (z - half)):(z_max - (z - half))
            ] = grid_map_t[x_slice, y_slice, z_slice]
            local_maps_t[agent_name] = local_map_t
            local_maps_tp1[agent_name] = self.ogm.get_local_map(module_id, patch_size=5)
            final_local_maps[agent_name] = self.ogm.get_final_local_map(module_id, patch_size=5)
        base_rewards = {}
        for agent_name in self.agents:
            obs_t = local_maps_t[agent_name]
            obs_tp1 = local_maps_tp1[agent_name]
            obs_f = final_local_maps[agent_name]
            # A: positions where obs_tp1 == obs_f
            A = set(zip(*np.where(obs_tp1 == obs_f)))
            # B: positions where obs_t == obs_f
            B = set(zip(*np.where(obs_t == obs_f)))
            base_rewards[agent_name] = len(A - B) - len(B - A)
        # Compute empathy term
        for agent_name in self.agents:
            pos = positions[agent_name]
            # Find all agents in the 5x5x5 box centered at pos
            neighbors = []
            for other_name in self.agents:
                if other_name == agent_name:
                    continue
                other_pos = positions[other_name]
                if all(abs(p - q) <= 2 for p, q in zip(pos, other_pos)):
                    neighbors.append(other_name)
            empathy_sum = sum(base_rewards[n] for n in neighbors)
            rewards[agent_name] = base_rewards[agent_name] + self.empathy_lambda * empathy_sum
        return rewards

    def render(self, mode="human"):
        print("Current Module Positions:", self.ogm.module_positions) 