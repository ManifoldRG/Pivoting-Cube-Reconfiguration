"""
JAX Environment Package for MSSA TPU Training.

This package provides JAX-native implementations of the
Occupancy Grid Map environment for TPU-accelerated RL training.
"""

from jax_env.ogm_jax import (
    OGMState,
    OGMConfig,
    VectorizedOGMEnv,
    reset,
    step,
    get_observation,
    get_action_mask,
    compute_pairwise_norms,
    check_success,
    NUM_ACTIONS,
)

__all__ = [
    "OGMState",
    "OGMConfig", 
    "VectorizedOGMEnv",
    "reset",
    "step",
    "get_observation",
    "get_action_mask",
    "compute_pairwise_norms",
    "check_success",
    "NUM_ACTIONS",
]
