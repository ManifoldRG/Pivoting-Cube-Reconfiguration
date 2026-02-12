"""
Validation tests: compare JAX environment outputs against the NumPy
reference implementation (occupancy_grid_map.py).

Run with:
    cd MSSA
    python -m pytest tests/test_jax_vs_numpy.py -v

Requires JAX to be installed (pip install jax jaxlib chex).
On CPU this is fine for validation; on TPU it will use the TPU backend
automatically.

Test structure:
    1. test_action_deltas        -- JAX deltas match NumPy take_action offsets
    2. test_pairwise_norms       -- JAX pairwise norms match NumPy calc_pairwise_norms
    3. test_four_band_reduction  -- JAX four-band matches NumPy calc_four_band_reduction
    4. test_articulation_points  -- JAX AP detection matches NumPy articulationPoints
    5. test_action_masks         -- JAX get_action_mask matches NumPy calc_possible_actions
                                    (superset check: JAX may allow a few extra moves
                                    because pivot_zone_grid_map is not ported)
    6. test_check_success        -- JAX success check matches NumPy check_final
    7. test_connected_config     -- JAX make_connected_configuration produces valid
                                    connected structures
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np

# --- lazy JAX import so tests can be discovered even without JAX ---
try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from ogm.occupancy_grid_map import OccupancyGridMap


def _skip_if_no_jax(cls):
    if not HAS_JAX:
        return unittest.skip("JAX not installed")(cls)
    return cls


# ============================================================
# Helper: build OGM and extract positions as (n,3) int array
# ============================================================


def _make_ogm(positions_dict, n):
    """Create OGM with positions_dict as both init and final (same shape)."""
    return OccupancyGridMap(positions_dict, positions_dict, n)


def _positions_to_array(pos_dict):
    """Dict {1:(x,y,z), ...} -> (n,3) int32 array (0-indexed rows)."""
    n = len(pos_dict)
    arr = np.zeros((n, 3), dtype=np.int32)
    for k in range(1, n + 1):
        arr[k - 1] = pos_dict[k]
    return arr


# ============================================================
# Test fixtures: a handful of hand-crafted configurations
# ============================================================

CONFIGS = [
    # (name, positions_dict, n)
    ("linear_4", {1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0), 4: (3, 0, 0)}, 4),
    ("L_shape_4", {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (1, 2, 0)}, 4),
    (
        "T_shape_5",
        {1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0), 4: (1, 1, 0), 5: (1, -1, 0)},
        5,
    ),
    (
        "3d_star_5",
        {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1), 5: (0, 0, -1)},
        5,
    ),
    (
        "L_3d_6",
        {
            1: (0, 0, 0),
            2: (1, 0, 0),
            3: (2, 0, 0),
            4: (2, 1, 0),
            5: (2, 1, 1),
            6: (2, 1, 2),
        },
        6,
    ),
]


# ============================================================
# 1. Action deltas
# ============================================================


@_skip_if_no_jax
class TestActionDeltas(unittest.TestCase):
    """Verify that _ACTION_DELTAS_NP in ogm_jax matches take_action."""

    def test_all_48_deltas(self):
        from jax_env.ogm_jax import _ACTION_DELTAS_NP

        # Ground truth extracted from take_action (1-indexed actions)
        EXPECTED = {
            1: (1, 0, 0),
            2: (1, -1, 0),
            3: (1, 0, 0),
            4: (1, 1, 0),
            5: (0, 1, 0),
            6: (1, 1, 0),
            7: (0, -1, 0),
            8: (1, -1, 0),
            9: (-1, 0, 0),
            10: (-1, -1, 0),
            11: (-1, 0, 0),
            12: (-1, 1, 0),
            13: (0, 1, 0),
            14: (-1, 1, 0),
            15: (0, -1, 0),
            16: (-1, -1, 0),
            17: (1, 0, 0),
            18: (1, 0, -1),
            19: (1, 0, 0),
            20: (1, 0, 1),
            21: (0, 0, 1),
            22: (1, 0, 1),
            23: (0, 0, -1),
            24: (1, 0, -1),
            25: (-1, 0, 0),
            26: (-1, 0, -1),
            27: (-1, 0, 0),
            28: (-1, 0, 1),
            29: (0, 0, 1),
            30: (-1, 0, 1),
            31: (0, 0, -1),
            32: (-1, 0, -1),
            33: (0, 1, 0),
            34: (0, 1, -1),
            35: (0, 1, 0),
            36: (0, 1, 1),
            37: (0, 0, 1),
            38: (0, 1, 1),
            39: (0, 0, -1),
            40: (0, 1, -1),
            41: (0, -1, 0),
            42: (0, -1, -1),
            43: (0, -1, 0),
            44: (0, -1, 1),
            45: (0, 0, 1),
            46: (0, -1, 1),
            47: (0, 0, -1),
            48: (0, -1, -1),
        }
        for action_1idx, expected_delta in EXPECTED.items():
            row = _ACTION_DELTAS_NP[action_1idx - 1]
            self.assertEqual(
                tuple(row),
                expected_delta,
                f"Action {action_1idx}: got {tuple(row)}, want {expected_delta}",
            )


# ============================================================
# 2. Pairwise norms
# ============================================================


@_skip_if_no_jax
class TestPairwiseNorms(unittest.TestCase):
    def test_against_numpy(self):
        from jax_env.ogm_jax import compute_pairwise_norms

        for name, pos_dict, n in CONFIGS:
            with self.subTest(name=name):
                ogm = _make_ogm(pos_dict, n)
                # NumPy reference (uses recentered positions)
                np_norms = ogm.calc_pairwise_norms(ogm.module_positions)

                # JAX version (use recentered positions too)
                jax_pos = _positions_to_array(ogm.module_positions)
                jax_norms = np.array(compute_pairwise_norms(jnp.array(jax_pos)))

                np.testing.assert_allclose(
                    jax_norms,
                    np_norms,
                    atol=1e-5,
                    err_msg=f"Pairwise norms mismatch for {name}",
                )


# ============================================================
# 3. Four-band reduction
# ============================================================


@_skip_if_no_jax
class TestFourBandReduction(unittest.TestCase):
    def test_against_numpy(self):
        from jax_env.ogm_jax import compute_pairwise_norms, calc_four_band_reduction

        for name, pos_dict, n in CONFIGS:
            with self.subTest(name=name):
                ogm = _make_ogm(pos_dict, n)
                np_norms = ogm.calc_pairwise_norms(ogm.module_positions)
                np_bands = ogm.calc_four_band_reduction(np_norms)

                jax_pos = _positions_to_array(ogm.module_positions)
                jax_norms = compute_pairwise_norms(jnp.array(jax_pos))
                jax_bands = np.array(calc_four_band_reduction(jax_norms))

                np.testing.assert_allclose(
                    jax_bands,
                    np_bands,
                    atol=1e-5,
                    err_msg=f"Four-band mismatch for {name}",
                )


# ============================================================
# 4. Articulation points
# ============================================================


@_skip_if_no_jax
class TestArticulationPoints(unittest.TestCase):
    def test_against_numpy(self):
        from jax_env.ogm_jax import is_articulation_point

        for name, pos_dict, n in CONFIGS:
            with self.subTest(name=name):
                ogm = _make_ogm(pos_dict, n)
                # NumPy AP list (1-indexed, [-1] if none)
                np_aps = ogm.articulationPoints(n, ogm.edges)
                np_ap_set = set(np_aps) if np_aps != [-1] else set()

                # JAX AP array (0-indexed bool)
                jax_pos = _positions_to_array(ogm.module_positions)
                jax_aps = np.array(is_articulation_point(jnp.array(jax_pos), n))
                jax_ap_set = set(i + 1 for i in range(n) if jax_aps[i])

                self.assertEqual(
                    jax_ap_set,
                    np_ap_set,
                    f"{name}: JAX APs {jax_ap_set} != NumPy APs {np_ap_set}",
                )


# ============================================================
# 5. Action masks
# ============================================================


@_skip_if_no_jax
class TestActionMasks(unittest.TestCase):
    """JAX action mask must be a SUPERSET of NumPy mask.

    The JAX version omits pivot_zone_grid_map checks, so it may
    allow a few extra moves that NumPy blocks.  Every move that
    NumPy allows must also be allowed by JAX.
    """

    def test_superset_property(self):
        from jax_env.ogm_jax import get_action_mask

        for name, pos_dict, n in CONFIGS:
            with self.subTest(name=name):
                ogm = _make_ogm(pos_dict, n)
                np_masks = ogm.calc_possible_actions()

                jax_pos = _positions_to_array(ogm.module_positions)
                grid_size = max(5, n * 2 + 3)

                for m in range(1, n + 1):
                    # JAX: rotate so module m is index 0
                    rolled = np.roll(jax_pos, -(m - 1), axis=0)
                    jax_mask = np.array(
                        get_action_mask(jnp.array(rolled), n, grid_size)
                    )

                    # NumPy mask: index 0..47 = actions 1..48, index 48 = no-op
                    np_mask = np_masks[m]

                    for a in range(49):
                        if np_mask[a]:
                            self.assertTrue(
                                jax_mask[a],
                                f"{name} module {m} action {a}: NumPy=True but JAX=False",
                            )


# ============================================================
# 6. Success check
# ============================================================


@_skip_if_no_jax
class TestCheckSuccess(unittest.TestCase):
    def test_identical_is_success(self):
        """Identical init and target -> success."""
        from jax_env.ogm_jax import check_success

        for name, pos_dict, n in CONFIGS:
            with self.subTest(name=name):
                ogm = _make_ogm(pos_dict, n)
                pos = _positions_to_array(ogm.module_positions)
                result = bool(check_success(jnp.array(pos), jnp.array(pos)))
                self.assertTrue(
                    result, f"{name}: identical positions should be success"
                )

    def test_different_is_not_success(self):
        """Different shapes -> not success (with high probability)."""
        from jax_env.ogm_jax import check_success

        # linear vs L-shape
        pos1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.int32)
        pos2 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0]], dtype=np.int32)
        result = bool(check_success(jnp.array(pos1), jnp.array(pos2)))
        self.assertFalse(result, "Linear vs L-shape should not be success")

    def test_rigid_rotation_is_success(self):
        """Same shape rotated 90 degrees -> success (unlabeled)."""
        from jax_env.ogm_jax import check_success

        # L-shape and its 90-degree rotation around z-axis
        pos1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.int32)
        # 90 deg rotation: (x,y) -> (-y, x)  =>  (0,0),(0,1),(-1,1)
        pos2 = np.array([[0, 0, 0], [0, 1, 0], [-1, 1, 0]], dtype=np.int32)
        result = bool(check_success(jnp.array(pos1), jnp.array(pos2)))
        self.assertTrue(result, "Rotated L-shape should match (unlabeled)")


# ============================================================
# 7. Connected configuration generation
# ============================================================


@_skip_if_no_jax
class TestConnectedConfig(unittest.TestCase):
    def _is_connected(self, positions):
        """Check connectivity via BFS."""
        n = len(positions)
        if n <= 1:
            return True
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if np.sum(np.abs(positions[i] - positions[j])) == 1:
                    adj[i].append(j)
                    adj[j].append(i)
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adj[node])
        return len(visited) == n

    def test_connectivity(self):
        from jax_env.ogm_jax import make_connected_configuration

        for n in [4, 5, 6, 8, 10, 15]:
            for seed in range(5):
                with self.subTest(n=n, seed=seed):
                    key = jax.random.PRNGKey(seed)
                    pos = np.array(make_connected_configuration(key, n))
                    self.assertTrue(
                        self._is_connected(pos),
                        f"n={n} seed={seed}: generated config is not connected",
                    )

    def test_no_duplicates(self):
        from jax_env.ogm_jax import make_connected_configuration

        for n in [4, 6, 8, 10]:
            key = jax.random.PRNGKey(42)
            pos = np.array(make_connected_configuration(key, n))
            # All positions must be unique
            unique = set(map(tuple, pos.tolist()))
            self.assertEqual(
                len(unique), n, f"n={n}: {n - len(unique)} duplicate positions"
            )


# ============================================================
# 8. Soft matching reward
# ============================================================


@_skip_if_no_jax
class TestSoftMatchingReward(unittest.TestCase):
    """Verify JAX soft matching scores match NumPy OGMEnv."""

    def test_labeled_soft_matching(self):
        """JAX compute_soft_matching_score matches OGMEnv.compute_soft_matching_score."""
        from jax_env.ogm_jax import compute_pairwise_sqdist, compute_soft_matching_score
        from ogm.ogm_env import OGMEnv

        env = OGMEnv()

        for name, pos_dict, n in CONFIGS:
            with self.subTest(name=name):
                pos1 = _positions_to_array(pos_dict)
                # Create a slightly different config for comparison
                pos2 = pos1.copy()
                pos2[0] = pos2[0] + np.array([1, 0, 0])  # shift one module

                # NumPy reference
                np_sqdist1 = np.zeros((n, n))
                np_sqdist2 = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        np_sqdist1[i, j] = np.sum((pos1[i] - pos1[j]) ** 2)
                        np_sqdist2[i, j] = np.sum((pos2[i] - pos2[j]) ** 2)
                np_score = env.compute_soft_matching_score(np_sqdist1, np_sqdist2)

                # JAX
                jax_sqdist1 = compute_pairwise_sqdist(jnp.array(pos1))
                jax_sqdist2 = compute_pairwise_sqdist(jnp.array(pos2))
                jax_score = float(compute_soft_matching_score(jax_sqdist1, jax_sqdist2))

                self.assertAlmostEqual(
                    jax_score, np_score, places=5,
                    msg=f"{name}: JAX={jax_score:.6f} != NumPy={np_score:.6f}",
                )

    def test_identical_is_one(self):
        """Soft matching of identical configurations should return 1.0."""
        from jax_env.ogm_jax import compute_pairwise_sqdist, compute_soft_matching_score

        for name, pos_dict, n in CONFIGS:
            with self.subTest(name=name):
                pos = _positions_to_array(pos_dict)
                sqdist = compute_pairwise_sqdist(jnp.array(pos))
                score = float(compute_soft_matching_score(sqdist, sqdist))
                self.assertAlmostEqual(
                    score, 1.0, places=5,
                    msg=f"{name}: identical configs should give score=1.0, got {score}",
                )

    def test_unlabeled_soft_matching(self):
        """JAX unlabeled soft matching uses assignment correctly."""
        from jax_env.ogm_jax import (
            compute_pairwise_sqdist,
            compute_pairwise_norms,
            compute_sorted_signatures,
            compute_signature_cost_matrix,
            greedy_assignment,
            compute_unlabeled_soft_matching_score,
        )

        for name, pos_dict, n in CONFIGS:
            with self.subTest(name=name):
                pos = _positions_to_array(pos_dict)
                pos_jax = jnp.array(pos)

                sqdist = compute_pairwise_sqdist(pos_jax)
                norms = compute_pairwise_norms(pos_jax)
                sigs = compute_sorted_signatures(norms)
                cost = compute_signature_cost_matrix(sigs, sigs)
                assignment = greedy_assignment(cost, n)

                # Identity assignment on same config should give score=1.0
                score = float(compute_unlabeled_soft_matching_score(
                    sqdist, sqdist, assignment
                ))
                self.assertAlmostEqual(
                    score, 1.0, places=4,
                    msg=f"{name}: self-match should give ~1.0, got {score}",
                )


# ============================================================
# 9. Reward computation (end-to-end)
# ============================================================


@_skip_if_no_jax
class TestRewardComputation(unittest.TestCase):
    """Verify compute_reward runs without errors and produces reasonable values."""

    def test_reward_runs(self):
        """compute_reward should execute without tracing errors."""
        from jax_env.ogm_jax import (
            OGMConfig,
            compute_reward,
            compute_pairwise_norms,
            compute_pairwise_sqdist,
            compute_sorted_signatures,
            compute_signature_cost_matrix,
            greedy_assignment,
        )

        for name, pos_dict, n in CONFIGS:
            with self.subTest(name=name):
                pos = jnp.array(_positions_to_array(pos_dict), dtype=jnp.int32)
                # Shift one module for "current" configuration
                curr_pos = pos.at[0].add(jnp.array([1, 0, 0]))

                norms_init = compute_pairwise_norms(pos)
                norms_final = compute_pairwise_norms(pos)
                final_sigs = compute_sorted_signatures(norms_final)
                init_sigs = compute_sorted_signatures(norms_init)
                cost = compute_signature_cost_matrix(init_sigs, final_sigs)
                assignment = greedy_assignment(cost, n)
                final_sqdist = compute_pairwise_sqdist(pos)

                from jax_env.ogm_jax import compute_reassigned_diff
                init_diff = compute_reassigned_diff(norms_init, norms_final, assignment)
                initial_norm_diff = float(jnp.linalg.norm(init_diff, "fro"))

                config = OGMConfig(
                    n=n, max_steps=1000, grid_size=max(5, n * 2 + 3),
                    use_unlabeled=True, local_k=min(7, n),
                    enable_soft_matching=True, enable_potential=True,
                )

                reward, m50, m75, m90, new_assgn, new_phi = compute_reward(
                    pos, curr_pos, pos,
                    initial_norm_diff,
                    False, False, False,
                    assignment, final_sigs,
                    0.5,  # phi_max
                    final_sqdist,
                    jnp.int32(0),  # step_count
                    config,
                )

                # Reward should be finite
                self.assertTrue(
                    np.isfinite(float(reward)),
                    f"{name}: reward is not finite: {float(reward)}",
                )
                # Assignment should be valid indices
                assgn_np = np.array(new_assgn)
                self.assertTrue(
                    np.all(assgn_np >= 0) and np.all(assgn_np < n),
                    f"{name}: invalid assignment indices",
                )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
