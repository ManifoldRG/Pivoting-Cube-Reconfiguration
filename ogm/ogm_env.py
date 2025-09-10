# import gymnasium as gym
# from gymnasium import spaces
import numpy as np
from ogm.occupancy_grid_map import OccupancyGridMap

"""
Environment wrapper for OGM
"""
class OGMEnv:
    def __init__(self, step_cost = -0.01, max_steps = None):
        self.step_cost = step_cost
        self.max_steps = max_steps 
        self.steps_taken = 0
        self.ogm = None
        self.action_buffer = [] 
        self.action_count = 0 
        self.num_modules = None
        self.initial_norm_diff = None 

    def reset(self, initial_config, final_config):
        self.ogm = OccupancyGridMap(initial_config, final_config, len(initial_config))
        self.steps_taken = 0
        self.action_buffer = []
        self.action_count = 0
        self.num_modules = len(initial_config)
        self.initial_norm_diff = np.linalg.norm(
            self.ogm.final_pairwise_norms - self.ogm.curr_pairwise_norms, 'fro'
        )
        return self.get_observation()
    
    def step(self, action):
        if self.ogm is None:
            raise Exception("Environment not set. call reset function")

        self.action_buffer.append(action)
        self.action_count += 1

        reward = 0.0
        done = False
        observation = self.get_observation()
        info = {'step': self.steps_taken}

        # Return 0 reward if not all modules have acted
        if self.action_count < self.num_modules:
            return observation, reward, done, info

        self.steps_taken += 1
        invalid_move = False

        # Execute buffered actions
        for module, pivot in self.action_buffer:
            is_valid_action = self.ogm.calc_possible_actions()[module][pivot-1]
            if is_valid_action:
                self.ogm.take_action(module, pivot)
            else:
                invalid_move = True

        final_norm_diff = np.linalg.norm(
            self.ogm.final_pairwise_norms - self.ogm.curr_pairwise_norms, 'fro'
        )

        # Calculate reward
        # the Frobenius norm scales with n^2 (for n modules). This could result in large reward values for large n
        # so normalizing the norm difference by n^2 keeps rewards in a reasonable range
        potential_reward = (self.initial_norm_diff - final_norm_diff) / (self.num_modules ** 2)
        success_bonus = 100.0 if self.ogm.check_final() else 0.0
        invalid_move_penalty = -1.0 if invalid_move else 0.0
        reward = potential_reward + success_bonus + invalid_move_penalty + self.step_cost

        self.initial_norm_diff = final_norm_diff

        done = self.ogm.check_final() or (self.max_steps is not None and self.steps_taken >= self.max_steps)

        self.action_buffer = []
        self.action_count = 0

        observation = self.get_observation()
        info = {'step': self.steps_taken}
        return observation, reward, done, info
    
    def get_observation(self):
        """
        Stacks the current and final grid maps to create a 2-channel observation.
        This allows the agent to see both its current state and its goal state.
        """
        if self.ogm is None:
            raise Exception("Environment not set. call reset function")
        
        # Shape becomes (2, grid_size, grid_size, grid_size)
        return np.stack([self.ogm.curr_grid_map, self.ogm.final_grid_map], axis=0)