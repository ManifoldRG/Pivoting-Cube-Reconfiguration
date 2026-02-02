"""
Curriculum Training Script for JAX/TPU.

Implements curriculum learning for scaling to n=50 agents
using JAX-accelerated training on TPU.
"""

import jax
import jax.numpy as jnp
from jax import lax
import optax
from functools import partial
from typing import Dict, List, Optional, Tuple
import time
import os
import argparse
from datetime import datetime
import pickle

# Local imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_env.ogm_jax import OGMConfig, VectorizedOGMEnv, NUM_ACTIONS
from train_jax.ppo_jax import (
    ActorCritic,
    PPOConfig,
    create_train_state,
    make_ppo_update,
    make_rollout_fn,
    Transition,
)


# ============================================
# Curriculum Configuration
# ============================================

CURRICULUM_STAGES: List[Dict] = [
    {
        "n": 4,
        "max_steps": 600,
        "min_episodes": 500,
        "max_episodes": 2000,
        "target_success": 0.90,
        "lr": 5e-4,
        "entropy": 0.03,
    },
    {
        "n": 5,
        "max_steps": 750,
        "min_episodes": 500,
        "max_episodes": 2500,
        "target_success": 0.88,
        "lr": 4.5e-4,
        "entropy": 0.03,
    },
    {
        "n": 6,
        "max_steps": 900,
        "min_episodes": 600,
        "max_episodes": 2500,
        "target_success": 0.85,
        "lr": 4e-4,
        "entropy": 0.03,
    },
    {
        "n": 7,
        "max_steps": 1200,
        "min_episodes": 600,
        "max_episodes": 3000,
        "target_success": 0.80,
        "lr": 3.5e-4,
        "entropy": 0.03,
    },
    {
        "n": 8,
        "max_steps": 2000,
        "min_episodes": 1500,
        "max_episodes": 5000,
        "target_success": 0.65,
        "lr": 1e-4,
        "entropy": 0.05,
    },
    {
        "n": 10,
        "max_steps": 2500,
        "min_episodes": 2000,
        "max_episodes": 6000,
        "target_success": 0.60,
        "lr": 8e-5,
        "entropy": 0.04,
    },
    {
        "n": 12,
        "max_steps": 3000,
        "min_episodes": 2500,
        "max_episodes": 7000,
        "target_success": 0.55,
        "lr": 6e-5,
        "entropy": 0.03,
    },
    {
        "n": 15,
        "max_steps": 4000,
        "min_episodes": 3000,
        "max_episodes": 8000,
        "target_success": 0.50,
        "lr": 5e-5,
        "entropy": 0.025,
    },
    {
        "n": 20,
        "max_steps": 5000,
        "min_episodes": 4000,
        "max_episodes": 10000,
        "target_success": 0.45,
        "lr": 4e-5,
        "entropy": 0.02,
    },
    {
        "n": 30,
        "max_steps": 7500,
        "min_episodes": 5000,
        "max_episodes": 12000,
        "target_success": 0.40,
        "lr": 3e-5,
        "entropy": 0.015,
    },
    {
        "n": 50,
        "max_steps": 12000,
        "min_episodes": 6000,
        "max_episodes": 15000,
        "target_success": 0.35,
        "lr": 2e-5,
        "entropy": 0.01,
    },
]


# ============================================
# Training Functions
# ============================================

def make_env(stage: Dict, local_k: int = 7) -> VectorizedOGMEnv:
    """Create environment for a curriculum stage."""
    config = OGMConfig(
        n=stage["n"],
        max_steps=stage["max_steps"],
        grid_size=50,
        use_unlabeled=True,
        local_k=local_k,
    )
    return VectorizedOGMEnv(config)


def make_ppo_config(stage: Dict) -> PPOConfig:
    """Create PPO config for a curriculum stage."""
    return PPOConfig(
        learning_rate=stage["lr"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=stage["entropy"],
        max_grad_norm=0.5,
        n_steps=256,
        n_minibatches=8,
        n_epochs=4,
        anneal_lr=False,  # We handle LR in curriculum
    )


@partial(jax.jit, static_argnums=(0, 1))
def vectorized_reset(
    env: VectorizedOGMEnv,
    n_envs: int,
    keys: jax.Array
) -> Tuple[jax.Array, any]:
    """Reset multiple environments in parallel."""
    return jax.vmap(env.reset)(keys)


@partial(jax.jit, static_argnums=(0,))
def vectorized_step(
    env: VectorizedOGMEnv,
    states,
    actions: jax.Array,
    agent_indices: jax.Array
) -> Tuple[jax.Array, any, jax.Array, jax.Array, dict]:
    """Step multiple environments in parallel."""
    return jax.vmap(env.step)(states, actions, agent_indices)


def train_stage(
    stage: Dict,
    stage_idx: int,
    prev_params: Optional[dict] = None,
    log_dir: str = "runs/jax_curriculum",
    local_k: int = 7,
    n_envs: int = 256,  # Many parallel envs on TPU
    key: jax.Array = None,
    verbose: bool = True,
) -> Tuple[dict, float]:
    """
    Train a single curriculum stage.
    
    Args:
        stage: Stage configuration
        stage_idx: Index of this stage
        prev_params: Parameters from previous stage (for fine-tuning)
        log_dir: Directory for logs and checkpoints
        local_k: Local neighborhood size
        n_envs: Number of parallel environments
        key: Random key
        verbose: Whether to print progress
        
    Returns:
        Tuple of (trained_params, final_success_rate)
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    n = stage["n"]
    
    # Create environment and network
    env = make_env(stage, local_k)
    network = ActorCritic(action_dim=NUM_ACTIONS, hidden_dims=(512, 512, 256))
    ppo_config = make_ppo_config(stage)
    
    # Create stage log directory
    stage_dir = os.path.join(log_dir, f"stage_n{n}")
    os.makedirs(stage_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Stage {stage_idx + 1}: n={n}")
        print(f"  Target success: {stage['target_success']*100:.0f}%")
        print(f"  LR: {stage['lr']:.2e}, Entropy: {stage['entropy']:.3f}")
        print(f"  Max steps: {stage['max_steps']}, Episodes: {stage['max_episodes']}")
        print(f"  Parallel environments: {n_envs}")
        print(f"{'='*60}")
    
    # Initialize training state
    key, init_key = jax.random.split(key)
    train_state = create_train_state(init_key, network, ppo_config, env.obs_shape)
    
    # Load previous parameters if available
    if prev_params is not None:
        if verbose:
            print("Loading parameters from previous stage...")
        train_state = train_state.replace(params=prev_params)
    
    # Initialize environments
    key, *env_keys = jax.random.split(key, n_envs + 1)
    env_keys = jnp.stack(env_keys)
    obs_batch, env_states = vectorized_reset(env, n_envs, env_keys)
    agent_indices = jnp.zeros(n_envs, dtype=jnp.int32)
    
    # Create update functions
    ppo_update = make_ppo_update(network, ppo_config)
    
    # Tracking
    episode_count = 0
    success_count = 0
    recent_successes = []
    window_size = 100
    start_time = time.time()
    
    # Main training loop
    steps_per_update = ppo_config.n_steps * n_envs
    total_steps = stage["max_episodes"] * stage["max_steps"]
    n_updates = total_steps // steps_per_update
    
    for update in range(n_updates):
        key, rollout_key, update_key = jax.random.split(key, 3)
        
        # Collect rollouts from all environments
        # This is where the TPU parallelism shines
        transitions_list = []
        
        for step in range(ppo_config.n_steps):
            key, action_key = jax.random.split(key)
            
            # Get action masks for all envs
            action_masks = jax.vmap(env.get_action_mask)(env_states, agent_indices)
            
            # Forward pass for all envs
            pis, values = jax.vmap(
                lambda o, m: network.apply(train_state.params, o[None], m[None])
            )(obs_batch, action_masks)
            
            # Need to squeeze the batch dim added above
            values = values[:, 0]
            
            # Sample actions
            action_keys = jax.random.split(action_key, n_envs)
            actions = jax.vmap(lambda pi, k: pi.sample(seed=k))(pis, action_keys)
            actions = actions[:, 0]  # Remove extra dim
            log_probs = jax.vmap(lambda pi, a: pi.log_prob(a))(pis, actions)
            log_probs = log_probs[:, 0]
            
            # Step all environments
            next_obs, env_states, rewards, dones, infos = vectorized_step(
                env, env_states, actions, agent_indices
            )
            
            # Store transition
            transitions_list.append(Transition(
                obs=obs_batch,
                action=actions,
                reward=rewards,
                done=dones,
                value=values,
                log_prob=log_probs,
                action_mask=action_masks,
            ))
            
            # Update agent indices
            agent_indices = (agent_indices + 1) % n
            obs_batch = next_obs
            
            # Track episodes
            n_done = jnp.sum(dones).item()
            n_success = jnp.sum(jnp.array([info.get("success", False) for info in infos])).item() if isinstance(infos, list) else dones.sum().item()  # Simplified
            episode_count += n_done
            success_count += n_success
            
            for d in dones:
                if d:
                    recent_successes.append(1 if n_success > 0 else 0)
                    if len(recent_successes) > window_size:
                        recent_successes.pop(0)
            
            # Reset done environments
            done_mask = dones
            if jnp.any(done_mask):
                key, *reset_keys = jax.random.split(key, int(done_mask.sum()) + 1)
                reset_keys = jnp.stack(reset_keys)
                # TODO: Selective reset for done envs
        
        # Stack transitions
        transitions = jax.tree_map(lambda *xs: jnp.stack(xs), *transitions_list)
        
        # Reshape for PPO update: (n_steps, n_envs, ...) -> (n_steps * n_envs, ...)
        transitions = jax.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1),
            transitions
        )
        
        # Get bootstrap value
        action_masks = jax.vmap(env.get_action_mask)(env_states, agent_indices)
        _, next_values = jax.vmap(
            lambda o, m: network.apply(train_state.params, o[None], m[None])
        )(obs_batch, action_masks)
        next_value = next_values[:, 0].mean()  # Average across envs
        
        # PPO update
        train_state, metrics = ppo_update(
            train_state, transitions, next_value, update_key
        )
        
        # Logging
        if update % 10 == 0:
            rolling_success = sum(recent_successes) / max(len(recent_successes), 1)
            elapsed = time.time() - start_time
            steps_done = (update + 1) * steps_per_update
            
            if verbose:
                print(f"Update {update:4d} | Episodes: {episode_count:5d} | "
                      f"Success: {rolling_success*100:5.1f}% | "
                      f"Loss: {metrics['total_loss']:.4f} | "
                      f"Steps/s: {steps_done/elapsed:.0f}")
            
            # Check early stopping
            if (episode_count >= stage["min_episodes"] and 
                len(recent_successes) >= window_size and
                rolling_success >= stage["target_success"]):
                if verbose:
                    print(f"\n🎯 TARGET REACHED! Success rate: {rolling_success*100:.1f}%")
                break
        
        # Check max episodes
        if episode_count >= stage["max_episodes"]:
            if verbose:
                print(f"\nMax episodes reached ({stage['max_episodes']})")
            break
    
    # Final statistics
    final_success = success_count / max(episode_count, 1)
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nStage {stage_idx + 1} complete!")
        print(f"  Episodes: {episode_count}")
        print(f"  Final success rate: {final_success*100:.1f}%")
        print(f"  Time: {elapsed/60:.1f} minutes")
    
    # Save checkpoint
    checkpoint_path = os.path.join(stage_dir, "checkpoint.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump({
            "params": train_state.params,
            "stage": stage,
            "success_rate": final_success,
            "episodes": episode_count,
        }, f)
    
    if verbose:
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    return train_state.params, final_success


def run_curriculum(
    start_stage: int = 0,
    target_n: int = 50,
    log_dir: str = None,
    local_k: int = 7,
    n_envs: int = 256,
    load_checkpoint: str = None,
    verbose: bool = True,
):
    """
    Run full curriculum learning.
    
    Args:
        start_stage: Stage to start from (0-indexed)
        target_n: Target number of agents
        log_dir: Logging directory
        local_k: Local neighborhood size
        n_envs: Number of parallel environments
        load_checkpoint: Path to checkpoint to load
        verbose: Whether to print progress
    """
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/jax_curriculum/to_n{target_n}_{timestamp}"
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Filter stages up to target
    stages = [s for s in CURRICULUM_STAGES if s["n"] <= target_n]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  MSSA Curriculum Learning (JAX/TPU)")
        print(f"  Target: n={target_n}")
        print(f"  Stages: {len(stages)}")
        print(f"  Start stage: {start_stage + 1}")
        print(f"  Log directory: {log_dir}")
        print(f"  Parallel environments: {n_envs}")
        print(f"{'='*60}")
        
        # Print device info
        print(f"\nJAX devices: {jax.devices()}")
        print(f"TPU available: {any('TPU' in str(d) for d in jax.devices())}\n")
    
    # Load checkpoint if provided
    prev_params = None
    if load_checkpoint:
        with open(load_checkpoint, "rb") as f:
            checkpoint = pickle.load(f)
        prev_params = checkpoint["params"]
        if verbose:
            print(f"Loaded checkpoint from {load_checkpoint}")
    
    # Run stages
    key = jax.random.PRNGKey(42)
    
    for i, stage in enumerate(stages[start_stage:], start=start_stage):
        key, stage_key = jax.random.split(key)
        
        params, success_rate = train_stage(
            stage=stage,
            stage_idx=i,
            prev_params=prev_params,
            log_dir=log_dir,
            local_k=local_k,
            n_envs=n_envs,
            key=stage_key,
            verbose=verbose,
        )
        
        prev_params = params
        
        # Check if stage failed
        if success_rate < stage["target_success"] * 0.8:  # Allow 20% tolerance
            if verbose:
                print(f"\n⚠️  Stage {i+1} did not reach target. Consider adjusting hyperparameters.")
    
    if verbose:
        print(f"\n{'='*60}")
        print("  Curriculum training complete!")
        print(f"{'='*60}")
    
    return prev_params


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(description="JAX Curriculum Training for MSSA")
    parser.add_argument("--start_stage", type=int, default=0, help="Stage to start from (0-indexed)")
    parser.add_argument("--target_n", type=int, default=50, help="Target number of agents")
    parser.add_argument("--log_dir", type=str, default=None, help="Logging directory")
    parser.add_argument("--local_k", type=int, default=7, help="Local neighborhood size")
    parser.add_argument("--n_envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    run_curriculum(
        start_stage=args.start_stage,
        target_n=args.target_n,
        log_dir=args.log_dir,
        local_k=args.local_k,
        n_envs=args.n_envs,
        load_checkpoint=args.load_checkpoint,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
