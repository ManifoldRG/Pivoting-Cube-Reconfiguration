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
    
    def get_rotation_invariant_distance(self):
        """
        Calculates the distance to the goal, accounting for all possible rotations.

        It computes the Hamming distance between the current grid map and every
        valid rotated version of the final grid map, returning the minimum
        distance found. This ensures the agent is rewarded for building the
        correct shape, regardless of its orientation.
        """
        if not hasattr(self.ogm, 'final_grid_maps') or len(self.ogm.final_grid_maps) == 0:
            return np.sum(self.ogm.curr_grid_map != self.ogm.final_grid_map)

        distances = [
            np.sum(self.ogm.curr_grid_map != final_map) 
            for final_map in self.ogm.final_grid_maps
        ]
        
        return min(distances)
    
    def step(self, action, is_first_move_in_phase=False, is_final_move_in_phase=False):
        if self.ogm is None:
            raise Exception("Environment not set. call reset function")

        module, pivot = action

        # If this is the first move, record the starting distance for the phase
        if is_first_move_in_phase:
            self.distance_at_phase_start = self.get_rotation_invariant_distance()

        # Execute the action regardless of which move it is
        is_valid_action = self.ogm.calc_possible_actions()[module][pivot-1]
        if is_valid_action:
            self.ogm.take_action(module, pivot)
        
        self.steps_taken += 1

        reward = self.step_cost  # Default reward is just the cost of taking a step
        done = False

        # Only calculate the full reward if this is the FINAL move in the phase
        if is_final_move_in_phase:
            done = self.ogm.check_final()
            final_distance = self.get_rotation_invariant_distance()

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