import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import unittest
import numpy as np
from ogm import occupancy_grid_map
from agent import module_agent
import networkx as nx

class TestModuleAgent(unittest.TestCase):

    def assert_module_local_map_match(self, expected_local_map, module, ogm):
        test_agent = module_agent.ModuleAgent(module, ogm)
        local_map = test_agent.ogm.get_local_map(module)
        self.assertTrue(np.all(local_map == expected_local_map),
                         msg=f"{expected_local_map} \nis the expected local map but we got \n{local_map}")


    def assert_module_reward_match(self, expected_set_length, expected_reward, curr_positions, final_positions):
        # Validate that modules identified as articulation points have no pivot actions
        ogm = occupancy_grid_map.OccupancyGridMap(curr_positions, final_positions, len(curr_positions))
        test_agent = module_agent.ModuleAgent(2, ogm)
        min_set = test_agent.calc_curr_v_final_local_map()
        
        # Check full articulation list matches expected
        self.assertEqual(expected_set_length, len(min_set),
                         msg=f"{expected_set_length} is expected length of the set of shared elements between the current local map and the final local map, but got {len(min_set)}")
        self.assertEqual(expected_reward, test_agent.calc_self_reward(), 
                         msg=f"{0} is expected reward but got {test_agent.calc_self_reward()}")

    def test_configuration_case_1(self):
        """
        Basic configuration:

        """
        ogm = occupancy_grid_map.OccupancyGridMap({1:(4,4,4), 2:(5,4,4), 3:(5,5,4)}, {1:(4,4,4), 2:(5,4,4), 3:(5,5,4)}, 3)
        expected_local_map = np.array([[[0,0,0,0,0],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
                                        [[0,0,0,0,0],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
                                        [[0,0,0,0,0],[0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0]], 
                                        [[0,0,0,0,0],[0,0,0,0,0], [0,0,2,0,0], [0,0,3,0,0], [0,0,0,0,0]], 
                                        [[0,0,0,0,0],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]], )
        self.assert_module_local_map_match(expected_local_map, 1, ogm)

    def test_configuration_case_2(self):
        self.assert_module_reward_match(3, 0, {1:(4,4,4), 2:(5,4,4), 3:(5,5,4)}, {1:(4,4,4), 2:(5,4,4), 3:(6,4,4)})
    
if __name__ == "__main__":
    unittest.main()
