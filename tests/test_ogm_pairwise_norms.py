import unittest
import numpy as np
import math
from ogm.occupancy_grid_map import OccupancyGridMap

class TestOGMPairwiseNorms(unittest.TestCase):
    def assert_pairwise_norms_match(self, module_positions, expected_pairwise_norms):
        ogm = OccupancyGridMap(module_positions, module_positions, 3)
        pairwise_norms = ogm.calc_pairwise_norms(ogm.module_positions)

        print(pairwise_norms)
              
        self.assertTrue(
            ((expected_pairwise_norms - pairwise_norms) + 1).all(),
            msg=f"\nExpected pairwise norms: {expected_pairwise_norms}\nActual pairwise norms:   {pairwise_norms}"
        )

    def assert_check_final(self, module_positions, final_module_positions):
        ogm = OccupancyGridMap(module_positions, final_module_positions, 3)

        self.assertTrue(
            ogm.check_final(),
            msg=f"\Current pairwise norms: {ogm.curr_pairwise_norms}\Final pairwise norms:   {ogm.final_pairwise_norms}"
        )

    def test_configuration_case_1(self):

        # Define initial module positions
        module_positions = {
            1: (4, 4, 4), 
            2: (4, 5, 4), 
            3: (4, 4, 5)
        }

        # Define final module positions
        final_module_positions = {
            1: (4, 4, 4), 
            2: (4, 4, 3), 
            3: (5, 4, 4)
        }

        self.assert_pairwise_norms_match(module_positions, np.array([[0.0, 1.0, 1.0], [1.0, 0.0, math.sqrt(2)], [1.0, math.sqrt(2), 0.0]]))
        self.assert_check_final(module_positions, final_module_positions)

if __name__ == "__main__":
    unittest.main()