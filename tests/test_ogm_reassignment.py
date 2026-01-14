import unittest
import numpy as np
import math
from ogm.occupancy_grid_map import OccupancyGridMap

class TestOGMReassignment(unittest.TestCase):
    def assert_correct_reassignment(self, module_positions, final_module_positions):
        ogm = OccupancyGridMap(module_positions, final_module_positions, len(module_positions))
        pairwise_norms = ogm.calc_pairwise_norms(ogm.module_positions)
        rearranged_matrix = ogm.calc_current_v_reassigned_final(pairwise_norms, ogm.final_pairwise_norms)

        print(pairwise_norms)
              
        self.assertTrue(
            np.allclose(np.zeros(pairwise_norms.shape), pairwise_norms - rearranged_matrix, atol=1e-6),
            msg=f"\nExpected reassigned matrix: {pairwise_norms}\nActual reassigned matrix:   {rearranged_matrix}"
        )

    def test_configuration_case_1(self):

        # Define initial module positions
        module_positions = {
            1: (4, 4, 4), 
            2: (4, 5, 4), 
            3: (4, 4, 6),
            4: (4, 4, 5)
        }

        # Define final module positions
        final_module_positions = {
            1: (4, 4, 4), 
            2: (4, 4, 3), 
            3: (5, 4, 4),
            4: (4, 4, 2)
        }

        self.assert_correct_reassignment(module_positions,final_module_positions)

if __name__ == "__main__":
    unittest.main()