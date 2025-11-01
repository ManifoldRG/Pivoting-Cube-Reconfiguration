import unittest
import numpy as np
from ogm import occupancy_grid_map
import networkx as nx

class TestOGMPossibleActions(unittest.TestCase):

    def assert_possible_actions_match(self, module_positions, expected_actions):
        # Verify that the list of valid pivot actions for each module matches what's expected.
        ogm = occupancy_grid_map.OccupancyGridMap(module_positions, module_positions, len(module_positions))
        actual_actions = ogm.calc_possible_actions()
        act = {}
        for m in ogm.modules:
            # Convert binary pivot array to list of action indices (1-indexed)
            act[m] = set(np.where(actual_actions[m])[0] + 1)

        expected_sets = {k : set(v) for k, v in expected_actions.items()}
        
        # Compare with expected action sets
        self.assertEqual(
            act, expected_sets,
            msg=f"\nExpected: {expected_sets}\nActual:   {act}"
        )

    def is_articulation_point(self, module_id, module_positions):
        # Build undirected graph of modules based on adjacency
        G = nx.Graph()
        for i, pos_i in module_positions.items():
            for j, pos_j in module_positions.items():
                if i < j and np.sum(np.abs(np.array(pos_i) - np.array(pos_j))) == 1:
                    G.add_edge(i, j)

        if module_id not in G:
            return False  # Not connected to anything

        G.remove_node(module_id)
        return not nx.is_connected(G)  # If graph breaks, it's articulation

    def assert_possible_articulation_point(self, module_positions, expected_articulation_points):
        # Validate that modules identified as articulation points have no pivot actions
        ogm = occupancy_grid_map.OccupancyGridMap(module_positions, module_positions, len(module_positions))
        pivots = ogm.calc_possible_actions()
        art_points = []

        for mod_id in ogm.modules:
            is_art = self.is_articulation_point(mod_id, ogm.module_positions)
            pivot_ids = list(np.where(pivots[mod_id])[0] + 1)

            if is_art:
                # Articulation points must have no available pivot actions
                self.assertEqual(pivot_ids, [49], msg=f"Module {mod_id} is articulation but has pivots: {pivot_ids}")
                art_points.append(mod_id)
            else:
                # Non-articulation points must still return a list
                self.assertIsInstance(pivot_ids, list)

        # Check full articulation list matches expected
        self.assertEqual(art_points, expected_articulation_points,
                         msg=f"{expected_articulation_points} are expected articulation points but got {art_points}")
        
    def assert_pivot_zones_match(self, module_positions, module_order_sequence, action_sequence, pivot_zones, possible_actions):
        ogm = occupancy_grid_map.OccupancyGridMap(module_positions, module_positions, len(module_positions))
        actions_list = []

        for i in range(len(module_order_sequence)):
            actions = ogm.calc_possible_actions()
            actions_list.append(actions)
            ogm.take_action(module_order_sequence[i], action_sequence[i])

        # Check pivot zones
        self.assertTrue(np.all(ogm.pivot_zone_grid_map == pivot_zones),
                         msg=f"{pivot_zones} are expected pivot zones but got {ogm.pivot_zone_grid_map}")
        
        for mod in actions.keys():
            self.assertTrue(np.all(actions[mod] == possible_actions[mod]),
                            msg=f"{possible_actions} are expected possible actions but got {actions}")
        ogm.calc_pre_action_grid_map()
        self.assertTrue(np.all(ogm.pivot_zone_grid_map == np.full((9, 9, 9), True)),
                         msg=f"{np.full((9, 9, 9), True)} are expected reset pivot zones but got {ogm.pivot_zone_grid_map}")



    def test_configuration_case_1(self):
        """
        Basic configuration:
        Module 1 — (4,4,4)
        Module 2 — (4,5,4) <- articulation point
        Module 3 — (5,5,4)

        - Module 2 connects the other two; its removal disconnects the graph.
        - Only modules 1 and 3 should have pivots.
        """

        # Initial and final grid configurations (hardcoded)
        matrix = np.zeros((9,9,9))
        matrix[4, 4, 4] = 1
        matrix[4, 5, 4] = 2
        matrix[5, 5, 4] = 3
        module_positions = {1: (4,4,4), 2: (4,5,4), 3: (5,5,4)}

        final_matrix = np.zeros((9,9,9))
        final_matrix[4, 4, 4] = 1
        final_matrix[3, 5, 4] = 2
        final_matrix[4, 5, 4] = 3
        final_module_positions = {1: (4, 4, 4), 2: (3, 5, 4), 3: (4, 5, 4)}

        num_modules = len(module_positions)

        # Instantiate OGM object
        ogm = occupancy_grid_map.OccupancyGridMap(module_positions, final_module_positions, num_modules)
        ogm.init_actions()

        # Expected pivot actions per module (from prior analysis)
        expected_actions = {
            1: [1, 10, 38, 40],
            2: [],
            3: [14, 15, 30, 32]
        }
        expected_actions = {1:[3, 12, 38, 40, 49], 2:[49], 3:[14, 15, 30, 32, 49]}

        # Expected articulation point
        expected_aticulation_points = [2]

        # Modules will act in this order:
        module_order_sequence = [1,2,3]

        # Modules will take these actions, respectively:
        action_sequence = [12, 49, 49]

        # At the end of the a sequence of actions, we expect this pivot zone array:
        pivot_zones = np.full((9, 9, 9), True)
        pivot_zones[4,4,4] = False
        pivot_zones[4,3,4] = False
        pivot_zones[3,5,4] = False
        pivot_zones[3,4,4] = False
        pivot_zones[3,3,4] = False
        pivot_zones[2,5,4] = False
        pivot_zones[2,4,4] = False
        pivot_zones[2,3,4] = False

        #check that mod3:pivot16 doesn't work
        possible_actions = {1: np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True]), 2: np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True]), 3: np.array([False, False, False, False, False, False, False, False, False,
       False, False, False, False,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False,  True, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True])}

        # Run articulation and pivot tests
        self.assert_possible_articulation_point(module_positions, expected_aticulation_points)
        self.assert_possible_actions_match(module_positions, expected_actions)
        #check that mod3:pivot16 doesn't work
        self.assert_pivot_zones_match(module_positions, module_order_sequence, action_sequence, pivot_zones, possible_actions)

    def test_configuration_case2(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(4,5,4), 3:(5,5,4)}, {1:[3, 12, 38, 40, 49], 2:[49], 3:[14, 15, 30, 32, 49]})
    def test_configuration_case3(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(4,3,4), 3:(5,3,4)}, {1:[1, 10, 46, 48, 49], 2:[49], 3:[13, 16, 30, 32, 49]})
    def test_configuration_case4(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(5, 4, 4), 3:(5, 3,4)}, {1:[7, 6, 22, 24, 49], 2:[49], 3: [11, 4, 38, 40, 49]})
    def test_configuration_case5(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(5,4,4), 3:(5,5,4)}, {1:[5, 8, 22, 24, 49], 2:[49], 3:[9, 2, 46, 48, 49]})
    def test_configuration_case6(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(4,5,4), 3:(4,5,3)}, {1:[39, 38, 4, 12, 49], 2:[49], 3:[43, 36, 20, 28, 49]})
    def test_configuration_case7(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(4,3,4), 3:(4,3,3)}, {1:[47, 46, 2, 10, 49], 2:[49], 3:[35, 44, 20, 28, 49]})
    def test_configuration_case8(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(4,5,4), 3:(4,5,5)}, {1:[37, 40, 4, 12, 49], 2:[49], 3:[41, 34, 18, 26, 49]})
    def test_configuration_case9(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(4,3,4), 3:(4,3,5)}, {1:[45, 48, 2, 10, 49], 2:[49], 3:[33, 42, 18, 26, 49]})
    def test_configuration_case10(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(3,4,4), 3:(3,4,5)}, {1:[29, 32, 14, 16, 49], 2:[49], 3:[17, 26, 34, 42, 49]})
    def test_configuration_case11(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(3,4,4), 3:(3,4,3)}, {1:[31, 30, 14, 16, 49], 2:[49], 3:[19, 28, 36, 44, 49]})
    def test_configuration_case12(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(5,4,4), 3:(5,4,5)}, {1:[21, 24, 6, 8, 49], 2:[49], 3:[25, 18, 34, 42, 49]})
    def test_configuration_case13(self):
        self.assert_possible_actions_match({1:(4,4,4), 2:(5,4,4), 3:(5,4,3)}, {1:[23, 22, 6, 8, 49], 2:[49], 3:[27, 20, 36, 44, 49]})

if __name__ == "__main__":
    unittest.main()
