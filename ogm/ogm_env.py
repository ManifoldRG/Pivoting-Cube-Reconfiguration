# import gymnasium as gym
# from gymnasium import spaces
import math
import numpy as np
from ogm.occupancy_grid_map import OccupancyGridMap

"""
Environment wrapper for OGM
"""
class OGMEnv:
    def __init__(self, step_cost = -0.005, max_steps = None,
                 enable_bounty_reward = False, bounty_gamma = 0.999, bounty_eta = 2.0,
                 bounty_base_value = 1.0, bounty_total_frac_of_success = 0.2,
                 bounty_cap_per_step = 20.0, enable_potential_reward = True,
                 potential_scale = 1.0, potential_normalize = 'n2', success_bonus = 100.0,
                 step_cost_initial = -0.01, step_cost_min = -0.001, use_exponential_decay = True):
        # General
        self.step_cost = step_cost
        self.max_steps = max_steps 
        self.steps_taken = 0
        self.ogm = None
        self.action_buffer = [] 
        self.action_count = 0 
        self.num_modules = None
        self.initial_norm_diff = None
        
        # Exponential decay parameters
        self.step_cost_initial = step_cost_initial
        self.step_cost_min = step_cost_min
        self.use_exponential_decay = use_exponential_decay
        self.decay_rate = None 
        # Reward config
        self.enable_bounty_reward = enable_bounty_reward
        self.bounty_params = {
            'gamma': bounty_gamma,
            'eta': bounty_eta,
            'base_value': bounty_base_value,
        }
        self.bounty_total_frac_of_success = bounty_total_frac_of_success
        self.bounty_cap_per_step = bounty_cap_per_step
        self.enable_potential_reward = enable_potential_reward
        self.potential_scale = potential_scale
        self.potential_normalize = potential_normalize
        self.success_bonus = success_bonus

    def reset(self, initial_config, final_config):
        self.ogm = OccupancyGridMap(initial_config, final_config, len(initial_config))
        self.steps_taken = 0
        self.action_buffer = []
        self.action_count = 0
        self.num_modules = len(initial_config)
        self.initial_norm_diff = np.linalg.norm(
            self.ogm.curr_pairwise_norms, 'fro'
        )
        
        # Calculate decay rate for exponential step cost
        if self.use_exponential_decay and self.max_steps is not None:
            self.decay_rate = -math.log(self.step_cost_min / self.step_cost_initial) / self.max_steps
        else:
            self.decay_rate = None
        # Initialize bounty/potential state
        if self.enable_bounty_reward or self.enable_potential_reward:
            self.final_sqdist = self.ogm.compute_pairwise_sqdist(self.ogm.final_module_positions)
            self.curr_sqdist = self.ogm.compute_pairwise_sqdist(self.ogm.module_positions)
        if self.enable_bounty_reward:
            self.pairs, self.bounty_base_values = self.ogm.generate_pair_bounties(
                self.final_sqdist, base_value=self.bounty_params['base_value']
            )
            # scale to fraction of success bonus
            desired_total = float(self.bounty_total_frac_of_success) * float(self.success_bonus)
            current_sum = float(np.sum(self.bounty_base_values)) if len(self.bounty_base_values) else 0.0
            if current_sum > 0.0:
                scale = desired_total / current_sum
                self.bounty_base_values = np.array(self.bounty_base_values, dtype=float) * float(scale)
            self.bounty_available = np.ones(len(self.pairs), dtype=bool)
            for idx, (i, j) in enumerate(self.pairs):
                if self.curr_sqdist[i, j] == self.final_sqdist[i, j]:
                    self.bounty_available[idx] = False
            self.R0 = int(self.bounty_available.sum())
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

        # Execute action immediately (sequential execution)
        self.ogm.take_action(action[0], action[1])

        self.steps_taken += 1
        invalid_move = False

        # we've already "made" the move, so just update the map.
        self.ogm.calc_pre_action_grid_map()
        # if I still care about tracking invalid moves, then create a new method to compare against just the current grid map

        # Execute buffered actions
        """ for module, pivot in self.action_buffer:
            moves = self.ogm.calc_possible_actions()
            is_valid_action = moves[module][pivot-1]
            # we've already "made" the move, so just update the map.
            if is_valid_action:
                #self.ogm.take_action(module, pivot)
                self.ogm.calc_pre_action_grid_map()
            else:
                invalid_move = True """

        final_norm_diff = np.linalg.norm(
            self.ogm.curr_pairwise_norms, 'fro'
        )

        # Calculate reward
        # Potential-based term (optionally enabled)
        if self.enable_potential_reward:
            if self.potential_normalize == 'n2':
                norm_factor = (self.num_modules ** 2)
            else:
                norm_factor = 1.0
            potential_reward = ((self.initial_norm_diff - final_norm_diff) / norm_factor) * self.potential_scale
        else:
            potential_reward = 0.0

        success_bonus = self.success_bonus if self.ogm.check_final() else 0.0
        bounty_reward = 0.0
        if self.enable_bounty_reward:
            # Update sqdist and pay new matches
            self.curr_sqdist = self.ogm.compute_pairwise_sqdist(self.ogm.module_positions)
            newly_matched_idx = []
            for idx, (i, j) in enumerate(self.pairs):
                if self.bounty_available[idx] and self.curr_sqdist[i, j] == self.final_sqdist[i, j]:
                    newly_matched_idx.append(idx)
            R_before = int(self.bounty_available.sum()) if self.bounty_available is not None else 0
            multiplier = 1.0 + self.bounty_params['eta'] * (R_before / self.R0) if getattr(self, 'R0', 0) and R_before > 0 else 1.0
            bounty_reward_raw = 0.0
            for idx in newly_matched_idx:
                b0 = self.bounty_base_values[idx]
                decay = self.bounty_params['gamma'] ** self.steps_taken
                bounty_reward_raw += b0 * decay * multiplier
                self.bounty_available[idx] = False
            cap = getattr(self, 'bounty_cap_per_step', None)
            bounty_reward = float(min(bounty_reward_raw, cap)) if cap is not None else float(bounty_reward_raw)
        invalid_move_penalty = -1.0 if invalid_move else 0.0
        
        # Calculate step cost (exponential decay or flat)
        if self.use_exponential_decay and self.decay_rate is not None:
            current_step_cost = max(
                self.step_cost_initial * math.exp(-self.decay_rate * self.steps_taken),
                self.step_cost_min
            )
        else:
            current_step_cost = self.step_cost
            
        reward = potential_reward + success_bonus + invalid_move_penalty + bounty_reward + current_step_cost

        self.initial_norm_diff = final_norm_diff

        done = self.ogm.check_final() or (self.max_steps is not None and self.steps_taken >= self.max_steps)

        self.action_buffer = []
        self.action_count = 0

        observation = self.get_observation()
        info = {'step': self.steps_taken, 'bounty_reward': bounty_reward, 'potential_reward': potential_reward}
        return observation, reward, done, info
    
    def get_observation(self):
        """
        Returns a single pairwise norms matrix representing the difference
        between the current and final configurations.

        `OccupancyGridMap` maintains `curr_pairwise_norms` as
        (current_pairwise_norms - final_pairwise_norms), so a zero matrix
        means the goal configuration has been reached.
        """
        if self.ogm is None:
            raise Exception("Environment not set. call reset function")
        
        # Normalize by maximum possible grid distance for scale stability
        grid_size = self.ogm.curr_grid_map.shape[0]
        max_dist = np.sqrt(3) * (grid_size - 1)
        max_dist = max(max_dist, 1.0)

        # Single (num_modules x num_modules) matrix, already representing
        # distance to goal in pairwise-norm space.
        return self.ogm.curr_pairwise_norms / max_dist 