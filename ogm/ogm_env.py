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
        if self.ogm.calc_possible_actions()[module][pivot-1]:
            self.ogm.take_action(module, pivot)
            self.steps_taken += 1
            done = self.ogm.check_final()
            reward = 100.0 if done else self.step_cost
            if not done and self.ogm.module_positions[module] == self.ogm.final_module_positions[module]:
                reward=0.5

        else:
            self.steps_taken += 1
            done = False
            reward = -0.01 + self.step_cost
        ## This will allow ogm to stop at some point?? Not sure if will be useful
        if self.max_steps is not None and self.steps_taken >= self.max_steps:
            done = True

        observation = self.get_observation()
        info = {'step': self.steps_taken}
        return observation, reward, done, info
    
    def get_observation(self):

        if self.ogm is None:
            raise Exception("Environment not set. call reset function")

        return np.copy(self.ogm.curr_grid_map)