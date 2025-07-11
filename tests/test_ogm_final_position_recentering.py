import unittest
import numpy as np
from ogm.occupancy_grid_map import OccupancyGridMap
from agent.random_search_agent import RandomSearchAgent
import networkx as nx

class TestOGMFinalPositionRecentering(unittest.TestCase):

    def assert_recentered_positions_match(self, module_positions, final_module_positions, expected_recentered_module_positions, expected_recentered_final_module_positions):
        # Create occupancy grid map with 3 modules
        ogm = OccupancyGridMap(module_positions, final_module_positions, 3)

        # Print grid size and module positions
        print(f"Grid size: {ogm.grid_map.shape}")
        print(f"Original module 1 position: {module_positions[1]}")
        print(f"Recentered module 1 position: {ogm.module_positions[1]}")
        print(f"All recentered module positions: {ogm.module_positions}")
        print(f"Original module 1 final position: {final_module_positions[1]}")
        print(f"Recentered module 1  final position: {ogm.final_module_positions[1]}")
        print(f"All recentered module final positions: {ogm.final_module_positions}")  
        
        # Compare with expected positions
        self.assertEqual(
            ogm.module_positions, expected_recentered_module_positions,
            msg=f"\nExpected recentered module positions: {expected_recentered_module_positions}\nActual recentered module positions:   {ogm.module_positions}"
        )

        self.assertEqual(
            ogm.final_module_positions, expected_recentered_final_module_positions,
            msg=f"\nExpected recentered final module positions: {expected_recentered_final_module_positions}\nActual recentered final module positions:   {ogm.final_module_positions}"
        )

    def assert_random_agent_success(self):
        matrix = np.zeros((9,9,9))
        matrix[4, 4, 4] = 1
        matrix[4, 5, 4] = 2
        matrix[5,5,4] = 3
        module_positions = {1: (4,4,4), 2: (4,5,4), 3: (5,5,4)}
        final_matrix = np.zeros((9,9,9))
        final_matrix[4,4,4] = 1
        final_matrix[3,5,4] = 2
        final_matrix[4,5,4] = 3
        final_module_positions = {1: (4, 4, 5), 2: (3, 5, 5), 3: (4, 5, 5)}
        ogm = OccupancyGridMap(module_positions, final_module_positions, 3)
        agent = RandomSearchAgent(max_steps=1000)

        success = agent.search(ogm)

        print(f"Search {'succeeded' if success else 'failed'} after {agent.steps_taken} steps")

        self.assertTrue(success, msg=f"\nSuccess: {success}")

    def test_configuration_case_1(self):
        """
        Basic configuration:
        Module 1 — (4,4,4)
        Module 2 — (4,5,4) <- articulation point
        Module 3 — (5,5,4)

        - Module 2 connects the other two; its removal disconnects the graph.
        - Only modules 1 and 3 should have pivots.
        """
        # Define initial module positions
        module_positions = {
            1: (19, 3, 5),  # This will be recentered to the grid center
            2: (19, 4, 5),
            3: (19, 3, 6)
        }

        # Define final module positions
        final_module_positions = {
            1: (19, 3, 6),
            2: (19, 3, 5),
            3: (20, 3, 6)
        }
        
        expected_recentered_module_positions = {
            1: (4, 4, 4), 
            2: (4, 5, 4), 
            3: (4, 4, 5)
        }

        expected_recentered_final_module_positions = {
            1: (4, 4, 4), 
            2: (4, 4, 3), 
            3: (5, 4, 4)
        }
        
        self.assert_recentered_positions_match(module_positions, final_module_positions, expected_recentered_module_positions, expected_recentered_final_module_positions)
        self.assert_random_agent_success()

if __name__ == "__main__":
    unittest.main()
