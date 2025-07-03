import unittest
import numpy as np
from MSSA.ogm import occupancy_grid_map
from MSSA.agent import module_agent
import networkx as nx

class TestModuleAgent(unittest.TestCase):

    def assert_module_local_map_match(self, expected_local_map, module, ogm):
        test_agent = module_agent.ModuleAgent(module, ogm)
        self.assertEqual(test_agent.get_local_map(), expected_local_map)

    def assert_module_reward_match(self, expected_reward, curr_positions, final_positions):
        # Validate that modules identified as articulation points have no pivot actions
        ogm = occupancy_grid_map.OccupancyGridMap(curr_positions, final_positions, len(curr_positions))
        art_points = []

        for mod_id in ogm.modules:
            is_art = self.is_articulation_point(mod_id, ogm.module_positions)
            pivot_ids = list(np.where(pivots[mod_id])[0] + 1)

            if is_art:
                # Articulation points must have no available pivot actions
                self.assertEqual(pivot_ids, [], msg=f"Module {mod_id} is articulation but has pivots: {pivot_ids}")
                art_points.append(mod_id)
            else:
                # Non-articulation points must still return a list
                self.assertIsInstance(pivot_ids, list)

        # Check full articulation list matches expected
        self.assertEqual(art_points, expected_articulation_points,
                         msg=f"{expected_articulation_points} are expected articulation points but got {art_points}")

    def test_configuration_case_1(self):
        """
        Basic configuration:
        Module 1 — (4,4,4)
        Module 2 — (4,5,4) <- articulation point
        Module 3 — (5,5,4)

        - Module 2 connects the other two; its removal disconnects the graph.
        - Only modules 1 and 3 should have pivots.
        """
        ogm = occupancy_grid_map.OccupancyGridMap({1:(4,4,4), 2:(5,4,4), 3:(5,5,4)}, {1:(4,4,4), 2:(5,4,4), 3:(5,5,4)}, 3)
        expected_local_map = np.array([[[0,0,0,0,0],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], [[0,0,0,0,0],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],[[0,0,0,0,0],[0,0,0,0,0], [0,0,1,0,0], [0,0,2,3,0], [0,0,0,0,0]], [[0,0,0,0,0],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]], [[0,0,0,0,0],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]], )
        self.assert_module_local_map_match(expected_local_map, 1, ogm)
    
if __name__ == "__main__":
    unittest.main()
