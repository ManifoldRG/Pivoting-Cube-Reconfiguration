# import gymnasium as gym
# from gymnasium import spaces
import numpy as np
from ogm.occupancy_grid_map import OccupancyGridMap

"""
Environment wrapper for OGM
"""
class OGMEnv:
    def __init__(self, step_cost = -0.005, max_steps = None, enable_bounty_reward = True, 
                 bounty_gamma = 0.999, bounty_eta = 2.0, bounty_base_value = 1.0,
                 enable_potential_reward = True, potential_scale = 1.0,
                 potential_normalize = 'n2', success_bonus = 100.0):
        self.step_cost = step_cost
        self.max_steps = max_steps 
        self.steps_taken = 0
        self.ogm = None
        self.action_buffer = [] 
        self.action_count = 0 
        self.num_modules = None
        self.initial_norm_diff = None
        
        # Bounty system parameters
        self.enable_bounty_reward = enable_bounty_reward
        self.bounty_params = {
            'gamma': bounty_gamma,  # decay factor
            'eta': bounty_eta,     # multiplier coefficient
            'base_value': bounty_base_value
        }
        
        # Bounty system state
        self.final_sqdist = None
        self.curr_sqdist = None
        self.pairs = None
        self.bounty_base_values = None
        self.bounty_multipliers = None
        self.R0 = None  # initial number of bounties 

        # Potential-based reward parameters/state
        self.enable_potential_reward = enable_potential_reward
        self.potential_scale = potential_scale
        self.potential_normalize = potential_normalize  # 'n2' or 'max' (only 'n2' used currently)
        self.success_bonus = success_bonus
        self.prev_frob_norm = None

    def reset(self, initial_config, final_config):
        self.ogm = OccupancyGridMap(initial_config, final_config, len(initial_config))
        self.steps_taken = 0
        self.action_buffer = []
        self.action_count = 0
        self.num_modules = len(initial_config)
        self.initial_norm_diff = np.linalg.norm(
            self.ogm.final_pairwise_norms - self.ogm.curr_pairwise_norms, 'fro'
        )
        
        # Initialize squared-distance matrices for potential/bounty systems
        if self.enable_bounty_reward or self.enable_potential_reward:
            self.final_sqdist = self.ogm.compute_pairwise_sqdist(self.ogm.final_module_positions)
            self.curr_sqdist = self.ogm.compute_pairwise_sqdist(self.ogm.module_positions)

        # Initialize bounty system
        if self.enable_bounty_reward:
            self.pairs, self.bounty_base_values = self.ogm.generate_pair_bounties(
                self.final_sqdist, base_value=self.bounty_params['base_value']
            )
            self.bounty_multipliers = np.ones(len(self.pairs), dtype=float)
            self.R0 = len(self.pairs)

        # Initialize potential state
        if self.enable_potential_reward:
            self.prev_frob_norm = float(np.linalg.norm(self.curr_sqdist - self.final_sqdist, 'fro'))
        else:
            self.prev_frob_norm = None
        
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
            # Action 49 (no-op) is always valid
            if pivot == 49:
                is_valid_action = True
            else:
                is_valid_action = self.ogm.calc_possible_actions()[module][pivot-1]
            
            if is_valid_action:
                self.ogm.take_action(module, pivot)
            else:
                invalid_move = True

        # Update current squared distances after actions (for potential and/or bounty)
        if self.enable_bounty_reward or self.enable_potential_reward:
            self.curr_sqdist = self.ogm.compute_pairwise_sqdist(self.ogm.module_positions)

        # Calculate potential-based reward using squared-distance Frobenius norm
        potential_reward = 0.0
        if self.enable_potential_reward:
            frob = float(np.linalg.norm(self.curr_sqdist - self.final_sqdist, 'fro'))
            # Improvement: previous - current (positive if closer)
            raw_improvement = (self.prev_frob_norm - frob)
            if self.potential_normalize == 'n2':
                norm_factor = (self.num_modules ** 2)
            else:
                norm_factor = 1.0
            potential_reward = (raw_improvement / norm_factor) * self.potential_scale
            # Update previous
            self.prev_frob_norm = frob

        success_bonus = self.success_bonus if self.ogm.check_final() else 0.0
        invalid_move_penalty = -1.0 if invalid_move else 0.0
        
        # Calculate bounty reward
        bounty_reward = 0.0
        bounties_collected = 0
        if self.enable_bounty_reward:
            bounty_reward, bounties_collected = self._compute_bounty_reward()
        
        reward = potential_reward + success_bonus + invalid_move_penalty + bounty_reward + self.step_cost

        done = self.ogm.check_final() or (self.max_steps is not None and self.steps_taken >= self.max_steps)

        self.action_buffer = []
        self.action_count = 0

        observation = self.get_observation()
        info = {
            'step': self.steps_taken,
            'potential_reward': potential_reward,
            'bounty_reward': bounty_reward,
            'bounties_collected': bounties_collected,
            'bounties_remaining': np.sum(self.bounty_multipliers > 1e-6) if self.enable_bounty_reward else 0,
        }
        return observation, reward, done, info
    
    def _compute_bounty_reward(self):
        """
        Compute bounty reward for currently matched pairs, with diminishing returns.
        Uses a "leaky" bounty system where bounties can be re-earned for reduced rewards.
        """
        currently_matched_idx = []
        
        # Find all pairs that currently match their target distances
        for idx, (i, j) in enumerate(self.pairs):
            if self.curr_sqdist[i, j] == self.final_sqdist[i, j]:
                # Only provide a reward if the multiplier is not zero
                if self.bounty_multipliers[idx] > 1e-6:
                    currently_matched_idx.append(idx)
        
        bounty_reward = 0.0
        
        # Use the number of CURRENTLY correct pairs for the multiplier
        num_correct_pairs = (self.curr_sqdist == self.final_sqdist).sum() // 2
        multiplier = 1.0 + self.bounty_params['eta'] * (num_correct_pairs / self.R0)
        
        for idx in currently_matched_idx:
            b0 = self.bounty_base_values[idx]
            decay = self.bounty_params['gamma'] ** self.steps_taken
            # The reward is scaled by the current multiplier for that bounty
            pay = b0 * decay * multiplier * self.bounty_multipliers[idx]
            bounty_reward += pay
            # Reduce the multiplier for the next time this bounty is earned
            self.bounty_multipliers[idx] /= 10.0  # Diminishing returns factor
        
        return bounty_reward, len(currently_matched_idx)
    
    def get_observation(self):
        """
        Returns the observation for the agent, which is the vector of
        differences between target and current pairwise distances.
        """
        if self.ogm is None:
            raise Exception("Environment not set. call reset function")
        
        return self.ogm.get_state_observation_vector()