"""
JAX Training Package for MSSA TPU Training.
"""

from train_jax.ppo_jax import (
    ActorCritic,
    PPOConfig,
    Transition,
    create_train_state,
    make_ppo_update,
    make_rollout_fn,
    compute_gae,
    train,
)

from train_jax.train_curriculum import (
    CURRICULUM_STAGES,
    train_stage,
    run_curriculum,
)

__all__ = [
    "ActorCritic",
    "PPOConfig",
    "Transition",
    "create_train_state",
    "make_ppo_update",
    "make_rollout_fn",
    "compute_gae",
    "train",
    "CURRICULUM_STAGES",
    "train_stage",
    "run_curriculum",
]
