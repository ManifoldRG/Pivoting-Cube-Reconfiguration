# import gymnasium as gym
# from gymnasium import spaces
import math
import numpy as np
from ogm.occupancy_grid_map import OccupancyGridMap

"""
Environment wrapper for OGM with soft matching reward
"""
class OGMEnv:
    def __init__(self, step_cost = -0.005, max_steps = None,
                 enable_bounty_reward = False, bounty_gamma = 0.999, bounty_eta = 2.0,
                 bounty_base_value = 1.0, bounty_total_frac_of_success = 0.2,
                 bounty_cap_per_step = 20.0, enable_potential_reward = True,
                 potential_scale = 1.0, potential_normalize = 'n2', success_bonus = 100.0,
                 step_cost_initial = -0.01, step_cost_min = -0.001, use_exponential_decay = True,
                 enable_soft_matching_reward = False, soft_matching_decay_beta = 0.999,
                 soft_matching_scale = 1.0):
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
        
        # Soft matching reward parameters
        self.enable_soft_matching_reward = enable_soft_matching_reward
        self.soft_matching_decay_beta = soft_matching_decay_beta
        self.soft_matching_scale = soft_matching_scale
        self.phi_max = 0.0  # Running best score

    def compute_soft_matching_score(self, curr_sqdist, final_sqdist):
        """
        Compute normalized soft pairwise matching score Φ(M).
        
        Φ(M) = (2 / (n(n-1))) * Σ_{u<v} 1 / (1 + (D_uv - D_uv_goal)²)
        
        Returns value in (0, 1], where 1 = perfect match.
        """
        n = curr_sqdist.shape[0]
        if n <= 1:
            return 1.0
        
        total_score = 0.0
        pair_count = 0
        
        # Sum over all pairs u < v
        for u in range(n):
            for v in range(u + 1, n):
                # Use squared distances (already computed)
                curr_d_sq = curr_sqdist[u, v]
                goal_d_sq = final_sqdist[u, v]
                
                # Difference in squared distances
                diff_sq = (curr_d_sq - goal_d_sq)
                
                # Soft matching score for this pair
                pair_score = 1.0 / (1.0 + diff_sq * diff_sq)
                total_score += pair_score
                pair_count += 1
        
        # Normalize by number of pairs: n(n-1)/2
        phi = (2.0 * total_score) / (n * (n - 1))
        return phi

    def reset(self, initial_config, final_config):
        self.ogm = OccupancyGridMap(initial_config, final_config, len(initial_config))
        self.steps_taken = 0
        self.action_buffer = []
        self.action_count = 0
        self.num_modules = len(initial_config)
        self.initial_norm_diff = np.linalg.norm(
            self.ogm.final_pairwise_norms - self.ogm.curr_pairwise_norms, 'fro'
        )
        
        # Calculate decay rate for exponential step cost
        if self.use_exponential_decay and self.max_steps is not None:
            self.decay_rate = -math.log(self.step_cost_min / self.step_cost_initial) / self.max_steps
        else:
            self.decay_rate = None
            
        # Initialize bounty/potential state
        if self.enable_bounty_reward or self.enable_potential_reward or self.enable_soft_matching_reward:
            self.final_sqdist = self.ogm.compute_pairwise_sqdist(self.ogm.final_module_positions)
            self.curr_sqdist = self.ogm.compute_pairwise_sqdist(self.ogm.module_positions)
            
        # Initialize soft matching reward state
        if self.enable_soft_matching_reward:
            self.phi_max = self.compute_soft_matching_score(self.curr_sqdist, self.final_sqdist)
            
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

        # Update the map
        self.ogm.calc_pre_action_grid_map()

        final_norm_diff = np.linalg.norm(
            self.ogm.final_pairwise_norms - self.ogm.curr_pairwise_norms, 'fro'
        )

        # Calculate reward components
        
        # 1. Soft matching reward (if enabled)
        soft_matching_reward = 0.0
        if self.enable_soft_matching_reward:
            # Update current squared distances
            self.curr_sqdist = self.ogm.compute_pairwise_sqdist(self.ogm.module_positions)
            
            # Compute current soft matching score
            phi_current = self.compute_soft_matching_score(self.curr_sqdist, self.final_sqdist)
            
            # Time decay weight: w(t) = beta^t
            w_t = self.soft_matching_decay_beta ** self.steps_taken
            
            # Reward = w(t) * max(0, Φ(t+1) - Φ_max(t))
            improvement = phi_current - self.phi_max
            if improvement > 0:
                soft_matching_reward = w_t * improvement * self.soft_matching_scale
                # Update running best
                self.phi_max = phi_current
            
            info['phi_current'] = phi_current
            info['phi_max'] = self.phi_max
        
        # 2. Potential-based term (optionally enabled)
        potential_reward = 0.0
        if self.enable_potential_reward:
            if self.potential_normalize == 'n2':
                norm_factor = (self.num_modules ** 2)
            else:
                norm_factor = 1.0
            potential_reward = ((self.initial_norm_diff - final_norm_diff) / norm_factor) * self.potential_scale

        # 3. Success bonus
        is_success = self.ogm.check_final()
        success_bonus = self.success_bonus if is_success else 0.0
        
        # 4. Bounty reward (if enabled)
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
        
        # 5. Invalid move penalty
        invalid_move_penalty = -1.0 if invalid_move else 0.0
        
        # 6. Step cost (exponential decay or flat)
        if self.use_exponential_decay and self.decay_rate is not None:
            current_step_cost = max(
                self.step_cost_initial * math.exp(-self.decay_rate * self.steps_taken),
                self.step_cost_min
            )
        else:
            current_step_cost = self.step_cost
            
        # Total reward
        reward = (soft_matching_reward + potential_reward + success_bonus + 
                 invalid_move_penalty + bounty_reward + current_step_cost)

        self.initial_norm_diff = final_norm_diff

        done = is_success or (self.max_steps is not None and self.steps_taken >= self.max_steps)

        self.action_buffer = []
        self.action_count = 0

        observation = self.get_observation()

        # Add distance metrics for debugging
        norm_diff = np.linalg.norm(
            self.ogm.final_pairwise_norms - self.ogm.curr_pairwise_norms, 'fro'
        )
        max_diff = np.max(np.abs(
            self.ogm.final_pairwise_norms - self.ogm.curr_pairwise_norms
        ))

        info.update({
            'bounty_reward': bounty_reward,
            'potential_reward': potential_reward,
            'soft_matching_reward': soft_matching_reward,
            'norm_diff': norm_diff,
            'max_diff': max_diff,
            'is_success': bool(is_success)
        })
        return observation, reward, done, info
    
    def get_observation(self):
        """
        Stacks the current and final pairwise norms to create a 2-channel observation.
        This allows the agent to see both its current shape and its goal shape.
        """
        if self.ogm is None:
            raise Exception("Environment not set. call reset function")
        
        # Normalize by maximum possible grid distance for scale stability
        grid_size = self.ogm.curr_grid_map.shape[0]
        max_dist = np.sqrt(3) * (grid_size - 1)
        max_dist = max(max_dist, 1.0)

        curr = self.ogm.curr_pairwise_norms / max_dist
        final = self.ogm.final_pairwise_norms / max_dist

        # Shape becomes (2, num_modules, num_modules)
        return np.stack([curr, final], axis=0)