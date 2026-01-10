import math
import numpy as np

from ogm.ogm_env import OGMEnv
from utils.random_configuration import random_configuration
from agent.random_search_agent import RandomSearchAgent


def initiate_test_ogm(n=6):
    # Generate initial module positions
    module_positions, final_module_positions,_ = random_configuration(n)


    # Create occupancy grid map with 3 modules
    ogm = OGMEnv(step_cost=0,step_cost_initial=0,step_cost_min=0,potential_normalize=1,potential_scale=50)
    ogm.reset(module_positions,final_module_positions)

    # Print grid size and module positions
    print(f"Grid size: {ogm.ogm.grid_map.shape}")
    print(f"Original module 1 position: {module_positions[1]}")
    print(f"Recentered module 1 position: {ogm.ogm.module_positions[1]}")

    return ogm

ogm = initiate_test_ogm(n=6)


agent = RandomSearchAgent(max_steps=1000)

success = agent.search(ogm)


