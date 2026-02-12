"""
JAX Environment Package for MSSA TPU Training.

This package provides JAX-native implementations of the
Occupancy Grid Map environment for TPU-accelerated RL training.

Imports are lazy so that the package can be *discovered* (e.g. by pytest)
even when JAX is not installed.  Actual symbols are only imported when
first accessed.
"""


def __getattr__(name):
    _EXPORTS = {
        "OGMState",
        "OGMConfig",
        "VectorizedOGMEnv",
        "reset",
        "step_with_action",
        "get_observation",
        "get_action_mask",
        "compute_pairwise_norms",
        "check_success",
        "compute_assignment",
        "compute_sorted_signatures",
        "compute_reassigned_diff",
        "greedy_assignment",
        "NUM_ACTIONS",
    }
    if name in _EXPORTS:
        from jax_env import ogm_jax

        return getattr(ogm_jax, name)
    raise AttributeError(f"module 'jax_env' has no attribute {name!r}")
