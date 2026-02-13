"""
Curriculum training script for JAX / TPU.

Trains the MSSA pivoting-cube agent from n=4 up to n=50 using
curriculum learning.  Each stage:
  1. Creates a VectorizedOGMEnv for the current n.
  2. Initialises (or loads) an ActorCritic network.
  3. Runs vectorised rollout collection across n_envs parallel
     environments using vmap over the single-env rollout fn.
  4. Computes GAE PER-ENVIRONMENT (critical: avoids cross-env contamination).
  5. Concatenates all transitions and runs a batched PPO update.
  6. Tracks a rolling success rate (last ROLLING_WINDOW episodes).
  7. Advances to the next stage once the rolling success rate
     exceeds the stage target, plateaus, or max_episodes is reached.

Key fixes vs previous version:
  - Step budget: max_steps = phases * n (matches PyTorch semantics)
  - GAE computed per-environment, not on interleaved buffer
  - Per-env bootstrap values (not averaged)
  - Observation normalization (running mean/var, matches VecNormalize)
  - Host round-trips minimized (batch done/success extraction after rollout)
  - Network dims aligned with PyTorch (512, 512, 256)

Usage:
    python train_jax/train_curriculum.py --target_n 50 --n_envs 256

    # Resume from checkpoint
    python train_jax/train_curriculum.py --target_n 50 \
        --load_checkpoint runs/jax_curriculum/stage_n8/checkpoint.pkl \
        --start_stage 4
"""

import jax
import jax.numpy as jnp
from jax import lax
import optax
from functools import partial
from typing import Dict, List, Optional, Tuple
import time
import os
import sys
import argparse
from datetime import datetime
import pickle
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_env.ogm_jax import OGMConfig, VectorizedOGMEnv, NUM_ACTIONS, check_success
from train_jax.ppo_jax import (
    ActorCritic,
    PPOConfig,
    Transition,
    Categorical,
    NormalizerState,
    init_normalizer,
    update_normalizer,
    normalize_obs,
    create_train_state,
    make_ppo_update,
    compute_gae_batched,
)


# ============================================
# Curriculum stage definitions
# ============================================

ROLLING_WINDOW_SIZE = 100
DEFAULT_ADVANCE_THRESHOLD = 0.85
DEFAULT_PLATEAU_MIN_DELTA = 0.005

# Step budgets now use: phases_per_agent * n = total individual agent actions
# This matches PyTorch where max_steps = phases, and OGMEnv gets phases * n.
CURRICULUM_STAGES: List[Dict] = [
    # Stage 0: n=4
    {
        "n": 4,
        "phases_per_episode": 600,
        "min_episodes": 400,
        "max_episodes": 1800,
        "target_rolling_success": 0.90,
        "lr": 5e-4,
        "entropy": 0.03,
        "plateau_patience": 300,
    },
    # Stage 1: n=5
    {
        "n": 5,
        "phases_per_episode": 800,
        "min_episodes": 450,
        "max_episodes": 2200,
        "target_rolling_success": 0.86,
        "lr": 4.5e-4,
        "entropy": 0.03,
        "plateau_patience": 350,
    },
    # Stage 2: n=6
    {
        "n": 6,
        "phases_per_episode": 950,
        "min_episodes": 550,
        "max_episodes": 2600,
        "target_rolling_success": 0.82,
        "lr": 4e-4,
        "entropy": 0.03,
        "plateau_patience": 400,
    },
    # Stage 3: n=7
    {
        "n": 7,
        "phases_per_episode": 1100,
        "min_episodes": 600,
        "max_episodes": 3000,
        "target_rolling_success": 0.78,
        "lr": 3.5e-4,
        "entropy": 0.03,
        "plateau_patience": 450,
    },
    # Stage 4: n=8
    {
        "n": 8,
        "phases_per_episode": 1600,
        "min_episodes": 700,
        "max_episodes": 4200,
        "target_rolling_success": 0.58,
        "lr": 2.5e-4,
        "entropy": 0.03,
        "plateau_patience": 600,
    },
    # Stage 5: n=9
    {
        "n": 9,
        "phases_per_episode": 1900,
        "min_episodes": 800,
        "max_episodes": 4600,
        "target_rolling_success": 0.54,
        "lr": 2.2e-4,
        "entropy": 0.029,
        "plateau_patience": 650,
    },
    # Stage 6: n=10
    {
        "n": 10,
        "phases_per_episode": 2200,
        "min_episodes": 900,
        "max_episodes": 5000,
        "target_rolling_success": 0.50,
        "lr": 2e-4,
        "entropy": 0.028,
        "plateau_patience": 700,
    },
    # Stage 7: n=12
    {
        "n": 12,
        "phases_per_episode": 2800,
        "min_episodes": 1100,
        "max_episodes": 6000,
        "target_rolling_success": 0.44,
        "lr": 1.5e-4,
        "entropy": 0.025,
        "plateau_patience": 800,
    },
    # Stage 8: n=15
    {
        "n": 15,
        "phases_per_episode": 3800,
        "min_episodes": 1400,
        "max_episodes": 7500,
        "target_rolling_success": 0.38,
        "lr": 1.2e-4,
        "entropy": 0.022,
        "plateau_patience": 1000,
    },
    # Stage 9: n=20
    {
        "n": 20,
        "phases_per_episode": 4000,
        "min_episodes": 1800,
        "max_episodes": 9000,
        "target_rolling_success": 0.34,
        "lr": 1e-4,
        "entropy": 0.015,
        "plateau_patience": 1300,
    },
    # Stage 10: n=30
    {
        "n": 30,
        "phases_per_episode": 6000,
        "min_episodes": 2500,
        "max_episodes": 12000,
        "target_rolling_success": 0.28,
        "lr": 7e-5,
        "entropy": 0.01,
        "plateau_patience": 1600,
    },
    # Stage 11: n=50
    {
        "n": 50,
        "phases_per_episode": 10000,
        "min_episodes": 3000,
        "max_episodes": 15000,
        "target_rolling_success": 0.22,
        "lr": 5e-5,
        "entropy": 0.008,
        "plateau_patience": 2200,
    },
]


# ============================================
# Helpers
# ============================================


def _warmup_normalizer(
    env: VectorizedOGMEnv,
    norm_state: NormalizerState,
    n_envs: int,
    n_samples: int = 1000,
    key: jax.Array = None,
) -> NormalizerState:
    if key is None:
        key = jax.random.PRNGKey(42)

    @jax.jit
    def _collect_samples(carry, _):
        norm_state, env_states, obs_batch, key = carry
        action_masks = jax.vmap(env.get_action_mask)(env_states)
        logits = jnp.where(
            action_masks, jnp.zeros_like(action_masks, dtype=jnp.float32), -1e9
        )
        action_keys = jax.random.split(key, n_envs)
        actions = jax.vmap(lambda lg, k: jax.random.categorical(k, lg))(
            logits, action_keys
        )
        next_obs, next_states, _, dones = jax.vmap(env.step)(env_states, actions)

        reset_keys = jax.random.split(key, n_envs + 1)
        key = reset_keys[0]
        reset_obs, reset_states = jax.vmap(env.reset)(reset_keys[1:])
        next_obs = jnp.where(dones[:, None], reset_obs, next_obs)
        next_states = jax.tree.map(
            lambda r, n: jnp.where(dones.reshape((-1,) + (1,) * (r.ndim - 1)), r, n),
            reset_states,
            next_states,
        )

        norm_state = update_normalizer(norm_state, next_obs)
        return (norm_state, next_states, next_obs, key), None

    init_keys = jax.random.split(key, n_envs + 1)
    key = init_keys[0]
    env_keys = init_keys[1:]
    obs_batch, env_states = jax.vmap(env.reset)(env_keys)

    (norm_state, _, _, key), _ = lax.scan(
        _collect_samples,
        (norm_state, env_states, obs_batch, key),
        None,
        length=n_samples,
    )
    return norm_state


def make_env(
    stage: Dict, local_k: Optional[int] = None, reward_kwargs: Optional[Dict] = None
) -> VectorizedOGMEnv:
    """Create OGM environment for the given curriculum stage.

    Step budget fix (D12): max_steps = phases_per_episode * n
    This matches PyTorch where OGMEnv max_steps = phases * num_agents.
    """
    if local_k is None:
        local_k = stage["n"]

    assert isinstance(local_k, int), "local_k must be int after None check"

    n = stage["n"]
    # max_steps in the JAX env counts individual agent actions
    max_steps = stage["phases_per_episode"] * n

    # Build reward config kwargs
    rk = reward_kwargs or {}
    config = OGMConfig(
        n=n,
        max_steps=max_steps,
        grid_size=max(5, n * 2 + 3),  # match NumPy calculate_grid_size
        use_unlabeled=True,
        local_k=local_k,
        enable_soft_matching=rk.get("enable_soft_matching", False),
        soft_matching_scale=rk.get("soft_matching_scale", 100.0),
        soft_matching_decay_beta=rk.get("soft_matching_decay_beta", 0.999),
        enable_potential=rk.get("enable_potential", True),
        potential_scale=rk.get("potential_scale", 1.0),
        success_bonus=rk.get("success_bonus", 100.0),
        step_cost=rk.get("step_cost", -0.005),
        use_exponential_step_cost=rk.get("use_exponential_step_cost", False),
        step_cost_initial=rk.get("step_cost_initial", -0.01),
        step_cost_min=rk.get("step_cost_min", -0.001),
        enable_local_reward=rk.get("enable_local_reward", True),
        local_reward_scale=rk.get("local_reward_scale", 0.5),
    )
    return VectorizedOGMEnv(config)


def make_ppo_config(stage: Dict) -> PPOConfig:
    n_steps = max(256, min(1024, stage["phases_per_episode"] // 4))
    return PPOConfig(
        learning_rate=stage["lr"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=stage["entropy"],
        max_grad_norm=0.5,
        n_steps=n_steps,
        n_minibatches=8,
        n_epochs=4,
        anneal_lr=False,
    )


# ============================================
# Vectorised rollout + reset (the core TPU
# workload -- everything here is vmap-ed)
# ============================================


def _make_vectorised_step(env: VectorizedOGMEnv, network: ActorCritic):
    """Return a JIT-compiled function that does ONE step across n_envs.

    Observations are normalized using the running normalizer state.
    """

    @jax.jit
    def vec_step(train_state, env_states, obs_batch, norm_state, key):
        """Single environment-step for all envs in parallel.

        Args:
            train_state: Flax TrainState
            env_states: batched OGMState
            obs_batch: (n_envs, obs_dim) raw observations
            norm_state: NormalizerState for obs normalization
            key: PRNG key

        Returns:
            (env_states, obs_batch, key,
             transitions,          # Transition with leading (n_envs,)
             dones, successes)     # (n_envs,) bool arrays
        """
        n_envs = obs_batch.shape[0]
        key, action_key = jax.random.split(key)

        # --- normalize observations ---
        obs_normed = normalize_obs(norm_state, obs_batch)

        # --- action masks for all envs (vmap over env) ---
        action_masks = jax.vmap(env.get_action_mask)(env_states)  # (n_envs, 49)

        # --- forward pass (batched) with normalized obs ---
        logits, values = network.apply(train_state.params, obs_normed, action_masks)
        # logits: (n_envs, 49), values: (n_envs,)

        # --- sample actions ---
        action_keys = jax.random.split(action_key, n_envs)
        pi = Categorical(logits)  # batched
        actions = jax.vmap(lambda lg, k: jax.random.categorical(k, lg))(
            logits, action_keys
        )
        log_probs = pi.log_prob(actions)  # (n_envs,)

        # --- step all envs ---
        next_obs, next_states, rewards, dones = jax.vmap(env.step)(
            env_states, actions
        )  # all (n_envs, ...)

        # --- detect successes BEFORE auto-reset ---
        successes = jax.vmap(check_success)(
            next_states.module_positions,
            next_states.final_positions,
        )  # (n_envs,) bool

        # --- auto-reset done environments ---
        all_keys = jax.random.split(key, n_envs + 1)  # (n_envs+1, 2)
        key = all_keys[0]
        reset_keys = all_keys[1:]  # (n_envs, 2)
        reset_obs, reset_states = jax.vmap(env.reset)(reset_keys)  # (n_envs, ...)

        # where done, swap in the reset state/obs
        next_obs = jnp.where(dones[:, None], reset_obs, next_obs)
        next_states = jax.tree.map(
            lambda r, n: jnp.where(dones.reshape((-1,) + (1,) * (r.ndim - 1)), r, n),
            reset_states,
            next_states,
        )

        # Store NORMALIZED obs in transition (what the network actually saw)
        t = Transition(
            obs=obs_normed,
            action=actions,
            reward=rewards,
            done=dones,
            value=values,
            log_prob=log_probs,
            action_mask=action_masks,
        )

        return next_states, next_obs, key, t, dones, successes

    return vec_step


# ============================================
# Single-stage training loop
# ============================================


def train_stage(
    stage: Dict,
    stage_idx: int,
    prev_params: Optional[dict] = None,
    prev_norm_state: Optional[NormalizerState] = None,
    log_dir: str = "runs/jax_curriculum",
    local_k: Optional[int] = None,  # None = use stage["n"] (full visibility)
    n_envs: int = 256,
    key: Optional[jax.Array] = None,
    verbose: bool = True,
    rolling_window: int = ROLLING_WINDOW_SIZE,
    plateau_min_delta: float = DEFAULT_PLATEAU_MIN_DELTA,
    reward_kwargs: Optional[Dict] = None,
    writer: Optional[SummaryWriter] = None,
    global_step_offset: int = 0,
) -> Tuple[dict, float, NormalizerState, int]:
    """Train one curriculum stage.  Returns (final_params, rolling_success, norm_state, next_offset)."""

    if key is None:
        key = jax.random.PRNGKey(42)

    n = stage["n"]
    max_steps_total = stage["phases_per_episode"] * n

    # --- setup ---
    env = make_env(stage, local_k, reward_kwargs)
    # Network dims match PyTorch: pi=[512, 512, 256], vf=[512, 512, 256]
    network = ActorCritic(action_dim=NUM_ACTIONS, hidden_dims=(512, 512, 256))
    ppo_config = make_ppo_config(stage)

    stage_dir = os.path.join(log_dir, f"stage_n{n}")
    os.makedirs(stage_dir, exist_ok=True)

    target_rolling = stage.get("target_rolling_success", DEFAULT_ADVANCE_THRESHOLD)
    plateau_patience = int(stage.get("plateau_patience", 0))

    if verbose:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Stage {stage_idx + 1}: n={n}", flush=True)
        print(
            f"  Target rolling success (window={rolling_window}): {target_rolling * 100:.0f}%",
            flush=True,
        )
        if plateau_patience > 0:
            print(
                f"  Plateau stop: no rolling improvement >= {plateau_min_delta * 100:.2f}% "
                f"for {plateau_patience} episodes",
                flush=True,
            )
        print(f"  LR={stage['lr']:.2e}  ent={stage['entropy']}", flush=True)
        print(
            f"  max_steps={max_steps_total} ({stage['phases_per_episode']} phases x {n} agents)",
            flush=True,
        )
        print(f"  max_episodes={stage['max_episodes']}", flush=True)
        print(f"  Parallel envs: {n_envs}", flush=True)
        print(f"  JAX devices: {jax.devices()}", flush=True)
        print(f"{'=' * 60}", flush=True)

    # --- initialise network ---
    key, init_key = jax.random.split(key)
    train_state = create_train_state(init_key, network, ppo_config, env.obs_shape)

    if prev_params is not None:
        if verbose:
            print("  Loading params from previous stage...", flush=True)
        train_state = train_state.replace(params=prev_params)

    # --- initialise observation normalizer ---
    if prev_norm_state is not None:
        if verbose:
            print("  Warming up normalizer at new n...", flush=True)
        norm_state = _warmup_normalizer(
            env, prev_norm_state, n_envs, n_samples=1000, key=key
        )
        if verbose:
            print("  Normalizer warmed up at new n", flush=True)
    else:
        norm_state = init_normalizer(env.obs_dim)

    # --- initialise environments ---
    all_init_keys = jax.random.split(key, n_envs + 1)  # (n_envs+1, 2)
    key = all_init_keys[0]
    env_keys = all_init_keys[1:]  # (n_envs, 2)
    obs_batch, env_states = jax.vmap(env.reset)(env_keys)  # (n_envs, obs_dim), states

    # Warm up normalizer with initial observations
    norm_state = update_normalizer(norm_state, obs_batch)

    # --- compile step & update fns ---
    vec_step = _make_vectorised_step(env, network)
    ppo_update = make_ppo_update(network, ppo_config)

    # --- tracking ---
    episode_outcomes = []  # list of 0/1 for rolling window
    episode_count = 0
    ep_this_update = 0
    success_this_update = 0
    start_time = time.time()
    best_rolling = 0.0
    best_rolling_episode = 0
    stop_reason = "max_updates"

    steps_per_update = ppo_config.n_steps * n_envs
    total_budget = stage["max_episodes"] * max_steps_total
    n_updates = total_budget // steps_per_update
    global_step = global_step_offset

    # --- main loop ---
    for update_idx in range(n_updates):
        initial_ent = stage["entropy"]
        decay_progress = jnp.float32(update_idx) / jnp.float32(max(n_updates, 1))
        current_ent = initial_ent * (1.0 - 0.5 * decay_progress)
        current_ent = float(current_ent)
        # ---- collect n_steps of experience ----
        transitions_list = []
        all_dones_list = []
        all_successes_list = []
        ep_this_update = 0
        success_this_update = 0

        for _ in range(ppo_config.n_steps):
            env_states, obs_batch, key, t, dones, successes = vec_step(
                train_state, env_states, obs_batch, norm_state, key
            )

            transitions_list.append(t)
            all_dones_list.append(dones)
            all_successes_list.append(successes)

            # Update normalizer with new raw observations
            norm_state = update_normalizer(norm_state, obs_batch)

        # ---- extract episode outcomes (batch transfer to host) ----
        # Stack dones/successes: (n_steps, n_envs)
        all_dones = jnp.stack(all_dones_list, axis=0)  # (n_steps, n_envs)
        all_successes = jnp.stack(all_successes_list, axis=0)

        # Transfer to host once (not per-step)
        dones_np = all_dones.tolist()
        successes_np = all_successes.tolist()

        for step_idx in range(ppo_config.n_steps):
            for env_idx in range(n_envs):
                if dones_np[step_idx][env_idx]:
                    episode_count += 1
                    ep_this_update += 1
                    outcome = 1 if successes_np[step_idx][env_idx] else 0
                    success_this_update += outcome
                    episode_outcomes.append(outcome)
                    if len(episode_outcomes) > rolling_window:
                        episode_outcomes.pop(0)

        # ---- stack transitions: (n_steps, n_envs, ...) ----
        transitions_stacked = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0), *transitions_list
        )  # (n_steps, n_envs, ...)

        # ---- compute GAE PER-ENVIRONMENT (critical fix D6+D7) ----
        # Bootstrap values: per-env (not averaged!)
        obs_normed = normalize_obs(norm_state, obs_batch)
        action_masks = jax.vmap(env.get_action_mask)(env_states)  # (n_envs, 49)
        _, next_values = network.apply(
            train_state.params, obs_normed, action_masks
        )  # (n_envs,)

        # Extract (n_steps, n_envs) shaped arrays for per-env GAE
        rewards_2d = transitions_stacked.reward  # (n_steps, n_envs)
        values_2d = transitions_stacked.value  # (n_steps, n_envs)
        dones_2d = transitions_stacked.done.astype(jnp.float32)  # (n_steps, n_envs)

        advantages, returns = compute_gae_batched(
            rewards_2d,
            values_2d,
            dones_2d,
            next_values,
            ppo_config.gamma,
            ppo_config.gae_lambda,
        )  # Both (n_steps * n_envs,)

        # Flatten transitions to (n_steps*n_envs, ...)
        transitions_flat = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1),
            transitions_stacked,
        )

        # ---- PPO update ----
        key, update_key = jax.random.split(key)
        train_state, metrics = ppo_update(
            train_state, transitions_flat, advantages, returns, update_key, current_ent
        )

        # ---- compute rolling success ----
        window_len = len(episode_outcomes)
        win_success = sum(episode_outcomes)
        rolling_success = win_success / max(window_len, 1)
        if episode_count >= stage["min_episodes"] and window_len >= rolling_window:
            if best_rolling_episode == 0:
                best_rolling = rolling_success
                best_rolling_episode = episode_count
            elif rolling_success >= (best_rolling + plateau_min_delta):
                best_rolling = rolling_success
                best_rolling_episode = episode_count

        episodes_since_best = (
            episode_count - best_rolling_episode if best_rolling_episode > 0 else 0
        )

        # ---- logging ----
        if verbose and (update_idx % 5 == 0 or ep_this_update > 0):
            elapsed = time.time() - start_time
            bar_len = 20
            filled_success = int(rolling_success * bar_len)
            filled_fail = min(
                bar_len - filled_success,
                int(
                    (1.0 - rolling_success)
                    * bar_len
                    * (window_len / max(rolling_window, 1))
                ),
            )
            filled_empty = bar_len - filled_success - filled_fail
            bar = "X" * filled_success + "o" * filled_fail + "." * filled_empty

            print(
                f"  upd {update_idx:5d} | "
                f"ep {episode_count:6d} (+{ep_this_update:3d}, {success_this_update}/{ep_this_update} ok) | "
                f"win [{win_success:3d}/{window_len:3d}]=[{bar}] {rolling_success * 100:5.1f}% | "
                f"criterion: rolling(last_{rolling_window})>={target_rolling * 100:4.1f}% | "
                f"best={best_rolling * 100:5.1f}%@ep{best_rolling_episode:5d} "
                f"since_best={episodes_since_best:4d} | "
                f"pol={float(metrics.get('policy_loss', 0.0)):8.4f} "
                f"val={float(metrics.get('value_loss', 0.0)):8.4f} "
                f"ent={float(metrics.get('entropy', 0.0)):6.4f} | "
                f"{elapsed / 60:.1f}m",
                flush=True,
            )

        # ---- tensorboard logging ----
        if writer is not None:
            global_step += steps_per_update
            writer.add_scalar(
                f"stage_n{n}/rolling_success", rolling_success, global_step
            )
            writer.add_scalar(
                f"stage_n{n}/target_rolling_success", target_rolling, global_step
            )
            writer.add_scalar(
                f"stage_n{n}/best_rolling_success", best_rolling, global_step
            )
            writer.add_scalar(
                f"stage_n{n}/policy_loss",
                float(metrics.get("policy_loss", 0.0)),
                global_step,
            )
            writer.add_scalar(
                f"stage_n{n}/value_loss",
                float(metrics.get("value_loss", 0.0)),
                global_step,
            )
            writer.add_scalar(
                f"stage_n{n}/entropy", float(metrics.get("entropy", 0.0)), global_step
            )
            writer.add_scalar(
                f"stage_n{n}/reward", float(metrics.get("total_loss", 0.0)), global_step
            )  # actually total loss, but useful
            if ep_this_update > 0:
                writer.add_scalar(
                    f"stage_n{n}/success_rate_this_update",
                    success_this_update / ep_this_update,
                    global_step,
                )

        # ---- early stop on target ----
        if (
            episode_count >= stage["min_episodes"]
            and window_len >= rolling_window
            and rolling_success >= target_rolling
        ):
            if verbose:
                print(
                    f"\n  TARGET HIT: rolling {rolling_success * 100:.1f}% "
                    f"({win_success}/{window_len} in window) >= {target_rolling * 100:.0f}%",
                    flush=True,
                )
            stop_reason = "target"
            break

        # ---- plateau stop ----
        if (
            plateau_patience > 0
            and episode_count >= stage["min_episodes"]
            and window_len >= rolling_window
            and best_rolling_episode > 0
            and episodes_since_best >= plateau_patience
        ):
            if verbose:
                print(
                    f"\n  PLATEAU STOP: best rolling {best_rolling * 100:.1f}% "
                    f"at episode {best_rolling_episode}; no improvement >= "
                    f"{plateau_min_delta * 100:.2f}% for {plateau_patience} episodes",
                    flush=True,
                )
            stop_reason = "plateau"
            break

        # ---- max episodes ----
        if episode_count >= stage["max_episodes"]:
            if verbose:
                print(f"\n  Max episodes reached ({stage['max_episodes']})", flush=True)
            stop_reason = "max_episodes"
            break

    # --- final stats ---
    final_rolling = sum(episode_outcomes) / max(len(episode_outcomes), 1)
    final_win_success = sum(episode_outcomes)
    final_win_len = len(episode_outcomes)
    elapsed = time.time() - start_time

    if verbose:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Stage {stage_idx + 1} (n={n}) done", flush=True)
        print(f"  Total episodes completed : {episode_count}", flush=True)
        print(
            f"  Window (last {rolling_window}): {final_win_success}/{final_win_len} success "
            f"= {final_rolling * 100:.1f}%",
            flush=True,
        )
        print(f"  Best rolling success     : {best_rolling * 100:.1f}%", flush=True)
        print(f"  Target                   : {target_rolling * 100:.0f}%", flush=True)
        print(f"  Stop reason              : {stop_reason}", flush=True)
        print(f"  Wall time                : {elapsed / 60:.1f} min", flush=True)

    # --- checkpoint ---
    ckpt_path = os.path.join(stage_dir, "checkpoint.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {
                "params": train_state.params,
                "norm_state": norm_state,
                "stage": stage,
                "rolling_success_rate": final_rolling,
                "best_rolling_success": best_rolling,
                "episodes": episode_count,
                "rolling_window_size": rolling_window,
                "plateau_min_delta": plateau_min_delta,
                "plateau_patience": plateau_patience,
                "stop_reason": stop_reason,
            },
            f,
        )
    if verbose:
        print(f"  Checkpoint: {ckpt_path}", flush=True)
        print(f"{'=' * 60}", flush=True)

    return train_state.params, final_rolling, norm_state, global_step


# ============================================
# Full curriculum
# ============================================


def run_curriculum(
    start_stage: int = 0,
    target_n: int = 50,
    log_dir: Optional[str] = None,
    local_k: int = 7,
    n_envs: int = 256,
    load_checkpoint: Optional[str] = None,
    verbose: bool = True,
    reward_kwargs: Optional[Dict] = None,
    use_tensorboard: bool = True,
    rolling_window: int = ROLLING_WINDOW_SIZE,
    plateau_min_delta: float = DEFAULT_PLATEAU_MIN_DELTA,
):
    if log_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/jax_curriculum/to_n{target_n}_{ts}"
    os.makedirs(log_dir, exist_ok=True)

    stages = [s for s in CURRICULUM_STAGES if s["n"] <= target_n]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  MSSA JAX Curriculum  ->  n={target_n}")
        print(f"  Stages: {[s['n'] for s in stages]}")
        print(f"  Start:  stage {start_stage + 1}")
        print(f"  Envs:   {n_envs}")
        print(f"  Rolling window: {rolling_window}")
        print(f"  Plateau min delta: {plateau_min_delta * 100:.2f}%")
        print(f"  Log:    {log_dir}")
        print(f"  JAX devices: {jax.devices()}")
        tpu = any("tpu" in str(d).lower() for d in jax.devices())
        print(f"  TPU detected: {tpu}")
        print(f"{'=' * 60}\n")

    # --- optional checkpoint load ---
    prev_params = None
    prev_norm_state = None
    if load_checkpoint and os.path.exists(load_checkpoint):
        with open(load_checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        prev_params = ckpt["params"]
        prev_norm_state = ckpt.get("norm_state", None)
        if verbose:
            print(f"  Loaded checkpoint: {load_checkpoint}")

    key = jax.random.PRNGKey(42)
    writer = SummaryWriter(log_dir) if use_tensorboard else None
    global_step_offset = 0

    for i, stage in enumerate(stages[start_stage:], start=start_stage):
        key, stage_key = jax.random.split(key)

        params, success_rate, norm_state, global_step_offset = train_stage(
            stage=stage,
            stage_idx=i,
            prev_params=prev_params,
            prev_norm_state=prev_norm_state,
            log_dir=log_dir,
            local_k=local_k,
            n_envs=n_envs,
            key=stage_key,
            verbose=verbose,
            rolling_window=rolling_window,
            plateau_min_delta=plateau_min_delta,
            reward_kwargs=reward_kwargs,
            writer=writer,
            global_step_offset=global_step_offset,
        )

        prev_params = params
        prev_norm_state = norm_state

        # warn if far below target
        target = stage.get("target_rolling_success", DEFAULT_ADVANCE_THRESHOLD)
        if success_rate < target * 0.8:
            if verbose:
                print(
                    f"\n  WARNING: stage {i + 1} success {success_rate * 100:.1f}% "
                    f"< 80% of target {target * 100:.0f}%. "
                    f"Consider tuning hyperparameters or training longer."
                )

    if writer:
        writer.close()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Curriculum complete!")
        print(f"{'=' * 60}")

    return prev_params


# ============================================
# CLI
# ============================================


def main():
    parser = argparse.ArgumentParser(description="JAX Curriculum Training for MSSA")
    parser.add_argument("--start_stage", type=int, default=0)
    parser.add_argument("--target_n", type=int, default=50)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--local_k", type=int, default=7)
    parser.add_argument("--n_envs", type=int, default=256)
    parser.add_argument("--rolling_window", type=int, default=ROLLING_WINDOW_SIZE)
    parser.add_argument(
        "--plateau_min_delta",
        type=float,
        default=DEFAULT_PLATEAU_MIN_DELTA,
        help="Minimum rolling-success improvement to reset plateau patience",
    )
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--no_tensorboard", action="store_true", help="Disable TensorBoard logging"
    )
    # Reward configuration
    parser.add_argument(
        "--enable_soft_matching",
        action="store_true",
        help="Enable soft matching reward",
    )
    parser.add_argument("--soft_matching_scale", type=float, default=100.0)
    parser.add_argument("--soft_matching_decay_beta", type=float, default=0.999)
    parser.add_argument(
        "--enable_potential",
        action="store_true",
        default=True,
        help="Enable potential-based shaping (default: on)",
    )
    parser.add_argument(
        "--no_potential", action="store_true", help="Disable potential-based shaping"
    )
    parser.add_argument("--potential_scale", type=float, default=1.0)
    parser.add_argument("--success_bonus", type=float, default=100.0)
    parser.add_argument("--step_cost", type=float, default=-0.005)
    parser.add_argument("--use_exponential_step_cost", action="store_true")
    parser.add_argument("--step_cost_initial", type=float, default=-0.01)
    parser.add_argument("--step_cost_min", type=float, default=-0.001)
    parser.add_argument(
        "--no_local_reward", action="store_true", help="Disable local reward"
    )
    parser.add_argument("--local_reward_scale", type=float, default=0.5)

    args = parser.parse_args()

    # Build reward kwargs dict
    reward_kwargs = {
        "enable_soft_matching": args.enable_soft_matching,
        "soft_matching_scale": args.soft_matching_scale,
        "soft_matching_decay_beta": args.soft_matching_decay_beta,
        "enable_potential": not args.no_potential,
        "potential_scale": args.potential_scale,
        "success_bonus": args.success_bonus,
        "step_cost": args.step_cost,
        "use_exponential_step_cost": args.use_exponential_step_cost,
        "step_cost_initial": args.step_cost_initial,
        "step_cost_min": args.step_cost_min,
        "enable_local_reward": not args.no_local_reward,
        "local_reward_scale": args.local_reward_scale,
    }

    run_curriculum(
        start_stage=args.start_stage,
        target_n=args.target_n,
        log_dir=args.log_dir,
        local_k=args.local_k,
        n_envs=args.n_envs,
        load_checkpoint=args.load_checkpoint,
        verbose=not args.quiet,
        reward_kwargs=reward_kwargs,
        use_tensorboard=not args.no_tensorboard,
        rolling_window=args.rolling_window,
        plateau_min_delta=args.plateau_min_delta,
    )


if __name__ == "__main__":
    main()
