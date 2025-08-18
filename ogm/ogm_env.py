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
        self.steps_token = 0
        self.ogm = OccupancyGridMap
        self.distance_at_phase_start = 0

    def reset(self, initial_config, final_config):
        self.ogm = OccupancyGridMap(initial_config, final_config, len(initial_config))
        self.steps_taken = 0
        self.distance_at_phase_start = 0
        return self.get_observation()
    
    def step(self, action):
        if self.ogm is None:
            raise Exception("Environment not set. call reset function")

        module, pivot = action

        # 1. Calculate distance before the move
        initial_distance = np.sum(self.ogm.curr_grid_map != self.ogm.final_grid_map)

        # Execute the action regardless of which move it is
        is_valid_action = self.ogm.calc_possible_actions()[module][pivot-1]
        if is_valid_action:
            self.ogm.take_action(module, pivot)
        
        self.steps_taken += 1

        # 2. Check for success and calculate new distance
        done = self.ogm.check_final()
        final_distance = np.sum(self.ogm.curr_grid_map != self.ogm.final_grid_map)

            # Calculate reward based on the change over the entire phase
            potential_reward = self.distance_at_phase_start - final_distance 
            
            success_bonus = 100.0 if done else 0.0
            invalid_move_penalty = -1.0 if not is_valid_action else 0.0 # This penalty still applies to the last agent

            # The final reward for the whole phase
            reward = potential_reward + success_bonus + invalid_move_penalty + self.step_cost
        
        if self.max_steps is not None and self.steps_taken >= self.max_steps:
            done = True

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