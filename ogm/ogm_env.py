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

    def reset(self, initial_config, final_config):
        self.ogm = OccupancyGridMap(initial_config, final_config, len(initial_config))
        self.steps_taken = 0
        return self.get_observation()
    
    def step(self, action):
        if self.ogm is None:
            raise Exception("Environment not set. call reset function")

        module, pivot = action

        # --- New Reward Logic ---
        # 1. Calculate distance before the move
        # Using a simple count of differing cells as the distance metric.
        initial_distance = np.sum(self.ogm.curr_grid_map != self.ogm.final_grid_map)

        # Execute the action
        # NOTE: It's important to check if the action is valid BEFORE taking it.
        is_valid_action = self.ogm.calc_possible_actions()[module][pivot-1]
        
        if is_valid_action:
            self.ogm.take_action(module, pivot)
        
        self.steps_taken += 1

        # 2. Check for success and calculate new distance
        done = self.ogm.check_final()
        final_distance = np.sum(self.ogm.curr_grid_map != self.ogm.final_grid_map)

        # 3. Calculate the hybrid reward
        # Reward for making progress toward the goal
        potential_reward = initial_distance - final_distance 
        
        # Large bonus for completing the puzzle
        success_bonus = 100.0 if done else 0.0
        
        # Small penalty for invalid moves to discourage them
        invalid_move_penalty = -1.0 if not is_valid_action else 0.0

        # Combine the components into the final reward for this step
        reward = potential_reward + success_bonus + invalid_move_penalty + self.step_cost

        # Stop if max steps are reached
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