"""
Curriculum training script for JAX / TPU.

Trains the MSSA pivoting-cube agent from n=4 up to n=50 using
curriculum learning.  Each stage:
  1. Creates a VectorizedOGMEnv for the current n.
  2. Initialises (or loads) an ActorCritic network.
  3. Runs vectorised rollout collection across n_envs parallel
     environments using vmap over the single-env rollout fn.
  4. Concatenates all transitions and runs a batched PPO update.
  5. Tracks a rolling success rate (last ROLLING_WINDOW episodes).
  6. Advances to the next stage once the rolling success rate
     exceeds the stage target (or max_episodes is reached).

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_env.ogm_jax import OGMConfig, VectorizedOGMEnv, NUM_ACTIONS, check_success
from train_jax.ppo_jax import (
    ActorCritic,
    PPOConfig,
    Transition,
    Categorical,
    create_train_state,
    make_ppo_update,
    make_rollout_fn,
)


# ============================================
# Curriculum stage definitions
# ============================================

ROLLING_WINDOW_SIZE = 100
DEFAULT_ADVANCE_THRESHOLD = 0.85

CURRICULUM_STAGES: List[Dict] = [
    # Stage 0: n=4 (baseline)
    {
        "n": 4,
        "max_steps": 600,
        "min_episodes": 500,
        "max_episodes": 2000,
        "target_rolling_success": 0.90,  # PyTorch target
        "lr": 5e-4,
        "entropy": 0.03,
    },
    # Stage 1: n=5
    {
        "n": 5,
        "max_steps": 750,
        "min_episodes": 500,
        "max_episodes": 2500,
        "target_rolling_success": 0.88,  # PyTorch target
        "lr": 4.5e-4,
        "entropy": 0.03,
    },
    # Stage 2: n=6
    {
        "n": 6,
        "max_steps": 900,
        "min_episodes": 600,
        "max_episodes": 2500,
        "target_rolling_success": 0.85,  # PyTorch target
        "lr": 4e-4,
        "entropy": 0.03,
    },
    # Stage 3: n=7
    {
        "n": 7,
        "max_steps": 1050,  # PyTorch: 150 steps/agent
        "min_episodes": 600,
        "max_episodes": 3000,
        "target_rolling_success": 0.80,  # PyTorch target (was 0.85)
        "lr": 3.5e-4,
        "entropy": 0.03,
    },
    # Stage 4: n=8 (CRITICAL - match PyTorch + increase budget)
    {
        "n": 8,
        "max_steps": 1600,  # PyTorch: 1400, increased for better convergence
        "min_episodes": 1000,  # PyTorch: 750, increased
        "max_episodes": 6000,  # PyTorch: 4000, increased (2× budget)
        "target_rolling_success": 0.75,  # PyTorch target (was 0.85)
        "lr": 3e-4,  # PyTorch LR (was 1e-4 ❌)
        "entropy": 0.035,  # PyTorch entropy (was 0.05)
    },
    # Stage 5: n=10
    {
        "n": 10,
        "max_steps": 2000,  # PyTorch: 1800, increased
        "min_episodes": 1200,  # PyTorch: 800, increased
        "max_episodes": 6000,  # PyTorch: 4500, increased
        "target_rolling_success": 0.75,  # Target 75% (PyTorch: 0.70)
        "lr": 2.5e-4,  # PyTorch LR
        "entropy": 0.03,
    },
    # Stage 6: n=12 (NEW - target ≥75%)
    {
        "n": 12,
        "max_steps": 2400,  # PyTorch had continuation to n=15, adapting
        "min_episodes": 1500,  # PyTorch: 1000, increased
        "max_episodes": 7000,  # PyTorch: 5000, increased
        "target_rolling_success": 0.75,  # Target 75% (PyTorch: 0.65)
        "lr": 2e-4,  # PyTorch LR
        "entropy": 0.025,  # PyTorch entropy
    },
    # Stage 7: n=15 (for future extension)
    {
        "n": 15,
        "max_steps": 3000,
        "min_episodes": 2000,
        "max_episodes": 8000,
        "target_rolling_success": 0.70,
        "lr": 1.5e-4,
        "entropy": 0.02,
    },
    # Stage 8: n=20
    {
        "n": 20,
        "max_steps": 4000,
        "min_episodes": 3000,
        "max_episodes": 10000,
        "target_rolling_success": 0.65,
        "lr": 1e-4,
        "entropy": 0.015,
    },
    # Stage 9: n=30
    {
        "n": 30,
        "max_steps": 6000,
        "min_episodes": 4000,
        "max_episodes": 12000,
        "target_rolling_success": 0.60,
        "lr": 8e-5,
        "entropy": 0.01,
    },
    # Stage 10: n=50 (final goal)
    {
        "n": 50,
        "max_steps": 10000,
        "min_episodes": 5000,
        "max_episodes": 15000,
        "target_rolling_success": 0.50,
        "lr": 5e-5,
        "entropy": 0.008,
    },
]


# ============================================
# Helpers
# ============================================


def make_env(stage: Dict, local_k: Optional[int] = None) -> VectorizedOGMEnv:
    """Create OGM environment for the given curriculum stage.

    Args:
        stage: dict with 'n', 'max_steps', etc.
        local_k: number of rows in observation window. If None, defaults to stage['n']
                 (full visibility of all modules). Original default was 7.
    """
    if local_k is None:
        local_k = stage["n"]

    assert isinstance(local_k, int), "local_k must be int after None check"

    config = OGMConfig(
        n=stage["n"],
        max_steps=stage["max_steps"],
        grid_size=max(5, stage["n"] * 2 + 3),  # match NumPy calculate_grid_size
        use_unlabeled=True,
        local_k=local_k,
    )
    return VectorizedOGMEnv(config)


def make_ppo_config(stage: Dict) -> PPOConfig:
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
        anneal_lr=False,
    )


# ============================================
# Vectorised rollout + reset (the core TPU
# workload -- everything here is vmap-ed)
# ============================================


def _make_vectorised_step(env: VectorizedOGMEnv, network: ActorCritic):
    """Return a JIT-compiled function that does ONE step across n_envs."""

    @jax.jit
    def vec_step(train_state, env_states, obs_batch, key):
        """Single environment-step for all envs in parallel.

        Returns:
            (env_states, obs_batch, key,
             transitions,          # Transition with leading (n_envs,)
             dones, successes)     # (n_envs,) bool arrays
        """
        n_envs = obs_batch.shape[0]
        key, action_key = jax.random.split(key)

        # --- action masks for all envs (vmap over env) ---
        action_masks = jax.vmap(env.get_action_mask)(env_states)  # (n_envs, 49)

        # --- forward pass (batched) ---
        logits, values = network.apply(train_state.params, obs_batch, action_masks)
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
        # success = done AND not truncated.  We approximate: if done and
        # the potential is ~0 we call it a success.  A simpler proxy:
        # check_success on the NEW positions (before reset overwrites them).
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

        t = Transition(
            obs=obs_batch,
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
    log_dir: str = "runs/jax_curriculum",
    local_k: Optional[int] = None,  # None = use stage["n"] (full visibility)
    n_envs: int = 256,
    key: Optional[jax.Array] = None,
    verbose: bool = True,
    rolling_window: int = ROLLING_WINDOW_SIZE,
) -> Tuple[dict, float]:
    """Train one curriculum stage.  Returns (final_params, rolling_success)."""

    if key is None:
        key = jax.random.PRNGKey(42)

    n = stage["n"]

    # --- setup ---
    env = make_env(stage, local_k)
    network = ActorCritic(action_dim=NUM_ACTIONS, hidden_dims=(1024, 1024, 512))
    ppo_config = make_ppo_config(stage)

    stage_dir = os.path.join(log_dir, f"stage_n{n}")
    os.makedirs(stage_dir, exist_ok=True)

    target_rolling = stage.get("target_rolling_success", DEFAULT_ADVANCE_THRESHOLD)

    if verbose:
        print(f"\n{'=' * 60}", flush=True)
        print(f"  Stage {stage_idx + 1}: n={n}", flush=True)
        print(
            f"  Target rolling success (window={rolling_window}): {target_rolling * 100:.0f}%",
            flush=True,
        )
        print(f"  LR={stage['lr']:.2e}  ent={stage['entropy']}", flush=True)
        print(
            f"  max_steps={stage['max_steps']}  max_episodes={stage['max_episodes']}",
            flush=True,
        )
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

    # --- initialise environments ---
    all_init_keys = jax.random.split(key, n_envs + 1)  # (n_envs+1, 2)
    key = all_init_keys[0]
    env_keys = all_init_keys[1:]  # (n_envs, 2)
    obs_batch, env_states = jax.vmap(env.reset)(env_keys)  # (n_envs, obs_dim), states

    # --- compile step & update fns ---
    vec_step = _make_vectorised_step(env, network)
    ppo_update = make_ppo_update(network, ppo_config)

    # --- tracking ---
    episode_outcomes = []  # list of 0/1 for rolling window (last `rolling_window` completed episodes)
    episode_count = 0  # total episodes completed this stage (lifetime)
    ep_this_update = 0  # episodes completed in the current update iteration
    success_this_update = 0  # successes in the current update iteration
    start_time = time.time()
    best_rolling = 0.0

    steps_per_update = ppo_config.n_steps * n_envs
    total_budget = stage["max_episodes"] * stage["max_steps"]
    n_updates = total_budget // steps_per_update

    # --- main loop ---
    for update_idx in range(n_updates):
        # ---- collect n_steps of experience ----
        transitions_list = []
        ep_this_update = 0
        success_this_update = 0

        for _ in range(ppo_config.n_steps):
            env_states, obs_batch, key, t, dones, successes = vec_step(
                train_state, env_states, obs_batch, key
            )

            transitions_list.append(t)

            # track episodes that finished this step
            done_np = dones.tolist()  # materialise to host
            success_np = successes.tolist()
            for i in range(n_envs):
                if done_np[i]:
                    episode_count += 1
                    ep_this_update += 1
                    outcome = 1 if success_np[i] else 0
                    success_this_update += outcome
                    episode_outcomes.append(outcome)
                    # keep only the last `rolling_window` outcomes
                    if len(episode_outcomes) > rolling_window:
                        episode_outcomes.pop(0)

        # ---- stack & flatten transitions ----
        # Each element of transitions_list is a Transition with leading (n_envs,)
        # Stack -> (n_steps, n_envs, ...) then reshape -> (n_steps*n_envs, ...)
        transitions = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0), *transitions_list
        )  # (n_steps, n_envs, ...)
        transitions = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1),
            transitions,
        )  # (n_steps*n_envs, ...)

        # ---- bootstrap value ----
        action_masks = jax.vmap(env.get_action_mask)(env_states)  # (n_envs, 49)
        _, next_values = network.apply(
            train_state.params, obs_batch, action_masks
        )  # (n_envs,)
        next_value = next_values.mean()  # scalar

        # ---- PPO update ----
        key, update_key = jax.random.split(key)
        train_state, metrics = ppo_update(
            train_state, transitions, next_value, update_key
        )

        # ---- compute rolling success (every update, used for logging + early stop) ----
        window_len = len(episode_outcomes)
        win_success = sum(episode_outcomes)  # successes in the window
        rolling_success = win_success / max(window_len, 1)
        # best_rolling tracks once we have at least min_episodes worth of data
        if episode_count >= stage["min_episodes"]:
            best_rolling = max(best_rolling, rolling_success)

        # ---- logging ----
        if verbose:
            elapsed = time.time() - start_time

            # bar: X = success, o = fail, . = empty (window not full yet)
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
                f"pol={float(metrics.get('policy_loss', 0.0)):8.4f} "
                f"val={float(metrics.get('value_loss', 0.0)):8.4f} "
                f"ent={float(metrics.get('entropy', 0.0)):6.4f} | "
                f"{elapsed / 60:.1f}m",
                flush=True,
            )

        # ---- early stop on target (checked every update) ----
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
            break

        # ---- max episodes ----
        if episode_count >= stage["max_episodes"]:
            if verbose:
                print(f"\n  Max episodes reached ({stage['max_episodes']})", flush=True)
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
        print(f"  Wall time                : {elapsed / 60:.1f} min", flush=True)

    # --- checkpoint ---
    ckpt_path = os.path.join(stage_dir, "checkpoint.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(
            {
                "params": train_state.params,
                "stage": stage,
                "rolling_success_rate": final_rolling,
                "best_rolling_success": best_rolling,
                "episodes": episode_count,
                "rolling_window_size": rolling_window,
            },
            f,
        )
    if verbose:
        print(f"  Checkpoint: {ckpt_path}", flush=True)
        print(f"{'=' * 60}", flush=True)

    return train_state.params, final_rolling


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
        print(f"  Log:    {log_dir}")
        print(f"  JAX devices: {jax.devices()}")
        tpu = any("tpu" in str(d).lower() for d in jax.devices())
        print(f"  TPU detected: {tpu}")
        print(f"{'=' * 60}\n")

    # --- optional checkpoint load ---
    prev_params = None
    if load_checkpoint and os.path.exists(load_checkpoint):
        with open(load_checkpoint, "rb") as f:
            ckpt = pickle.load(f)
        prev_params = ckpt["params"]
        if verbose:
            print(f"  Loaded checkpoint: {load_checkpoint}")

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

        # warn if far below target
        target = stage.get("target_rolling_success", DEFAULT_ADVANCE_THRESHOLD)
        if success_rate < target * 0.8:
            if verbose:
                print(
                    f"\n  WARNING: stage {i + 1} success {success_rate * 100:.1f}% "
                    f"< 80% of target {target * 100:.0f}%. "
                    f"Consider tuning hyperparameters or training longer."
                )

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
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

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
