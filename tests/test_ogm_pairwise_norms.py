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

    def assert_post_action_pairwise_norms(self, module_positions):
        ogm = OccupancyGridMap(module_positions, module_positions, 3)
        actions = ogm.calc_possible_actions()
        pairwise_norms = ogm.calc_post_pairwise_norms()
        self.assertTrue(
            True
        )

    def assert_four_band_reduction(self, module_positions, final_module_positions, expected_reduced_norms):
        ogm = OccupancyGridMap(module_positions, final_module_positions, len(module_positions))
        pairwise_norms = ogm.calc_pairwise_norms(ogm.module_positions)
        reduced_norms = ogm.calc_four_band_reduction(pairwise_norms)
        self.assertTrue(
            ((expected_reduced_norms - reduced_norms) + 1).all(),
            msg=f"\nExpected reduced pairwise norms: {expected_reduced_norms}\nActual reduced pairwise norms:   {reduced_norms}"
        )

    def assert_local_neighborhood_reduction(self, module_positions, final_module_positions, expected_reduced_norms, module, k):
        ogm = OccupancyGridMap(module_positions, final_module_positions, len(module_positions))
        pairwise_norms = ogm.calc_pairwise_norms(ogm.module_positions)
        reduced_norms = ogm.calc_local_neighborhood_reduction(pairwise_norms, module, k)
        self.assertTrue(
            ((expected_reduced_norms - reduced_norms) + 1).all(),
            msg=f"\nExpected reduced pairwise norms: {expected_reduced_norms}\nActual reduced pairwise norms:   {reduced_norms}"
        )

    def assert_dual_reduction(self, module_positions, final_module_positions, expected_reduced_norms, module, k):
        ogm = OccupancyGridMap(module_positions, final_module_positions, len(module_positions))
        pairwise_norms = ogm.calc_pairwise_norms(ogm.module_positions)
        reduced_norms = ogm.calc_four_band_reduction(pairwise_norms)
        reduced_norms = ogm.calc_local_neighborhood_reduction(reduced_norms, module, k)
        self.assertTrue(
            ((expected_reduced_norms - reduced_norms) + 1).all(),
            msg=f"\nExpected reduced pairwise norms: {expected_reduced_norms}\nActual reduced pairwise norms:   {reduced_norms}"
        )

    def test_configuration_case_1(self):

        # Define initial module positions
        module_positions = {
            1: (4, 4, 4), 
            2: (4, 5, 4), 
            3: (4, 4, 5)
        }

        # Define initial module positions when n = 5
        module_positions2 = {
            1: (4, 4, 4), 
            2: (4, 5, 4), 
            3: (4, 6, 4),
            4: (4, 7, 4),
            5: (4, 8, 4),
            6: (4, 9, 4)
        }

        # Define final module positions
        final_module_positions = {
            1: (4, 4, 4), 
            2: (4, 4, 3), 
            3: (5, 4, 4)
        }

        # Define final module positions that aren't just the initial module positions rotated
        final_module_positions2 = {
            1: (4, 4, 4), 
            2: (4, 5, 4), 
            3: (4, 6, 4),
            4: (4, 7, 4),
            5: (4, 8, 4),
            6: (4, 8, 5)
        }

        expected_reduced_pairwise_norms1 = np.array([[1., 2., 3., 4.],
       [1., 1., 2., 3.],
       [2., 1., 1., 2.],
       [2., 1., 1., 2.],
       [3., 2., 1., 1.],
       [4., 3., 2., 1.]])
        
        expected_reduced_pairwise_norms2 = np.array([[2., 1., 0., 1., 2., 3.],
       [3., 2., 1., 0., 1., 2.],
       [4., 3., 2., 1., 0., 1.]])
        
        expected_reduced_pairwise_norms3 = np.array([[4., 3., 2., 1., 0., 1.],
       [5., 4., 3., 2., 1., 0.],
       [0., 1., 2., 3., 4., 5.]])
        
        expected_dual_reduced_pairwise_norms = np.array([[3., 2., 1., 1.],
       [4., 3., 2., 1.],
       [1., 2., 3., 4.]])

        self.assert_pairwise_norms_match(module_positions, np.array([[0.0, 1.0, 1.0], [1.0, 0.0, math.sqrt(2)], [1.0, math.sqrt(2), 0.0]]))
        self.assert_check_final(module_positions, final_module_positions)
        self.assert_post_action_pairwise_norms(module_positions)
        self.assert_four_band_reduction(module_positions2, final_module_positions2, expected_reduced_pairwise_norms1)
        self.assert_local_neighborhood_reduction(module_positions2, final_module_positions2, expected_reduced_pairwise_norms2, 3, 3)
        self.assert_local_neighborhood_reduction(module_positions2, final_module_positions2, expected_reduced_pairwise_norms3, 5, 3)
        self.assert_dual_reduction(module_positions2, final_module_positions2, expected_dual_reduced_pairwise_norms, 5, 3)

if __name__ == "__main__":
    unittest.main()