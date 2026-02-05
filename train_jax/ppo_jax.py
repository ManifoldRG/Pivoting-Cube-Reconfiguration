"""
PPO implementation in JAX for TPU training.

Pure-JAX PPO with action masking, designed to be vmap-ed over hundreds of
parallel environments.  No external RL library dependencies -- only JAX,
Flax, and Optax.

Key design decisions:
- Categorical distribution is implemented from scratch (no distrax) to
  avoid version-conflict headaches on TPU images.
- ActorCritic uses fully-separate policy / value networks (no shared
  trunk) because the observation is small (32 dims) and separate heads
  train more stably for this problem.
- Gradient clipping is done via optax.clip_by_global_norm (correct L2
  clipping) instead of per-element clipping.
- GAE is computed with lax.scan in reverse (standard efficient impl).
- The PPO update loop (epochs x minibatches) is written as nested
  lax.scan so the entire update is a single XLA computation -- no
  Python-level iteration after the first JIT compile.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import NamedTuple, Tuple, Callable, Optional
import chex

import flax.linen as nn
from flax.training.train_state import TrainState
import optax


# ============================================
# Pure-JAX Categorical distribution
# ============================================


class Categorical:
    """Minimal categorical distribution over logits.

    Supports batched logits (any leading batch dims).  All methods are
    traceable -- no Python-level branching on values.
    """

    def __init__(self, logits: chex.Array):
        self.logits = logits  # (..., num_classes)

    @property
    def probs(self) -> chex.Array:
        return jax.nn.softmax(self.logits, axis=-1)

    def sample(self, seed: chex.PRNGKey) -> chex.Array:
        """Sample one index per leading-batch element."""
        return jax.random.categorical(seed, self.logits, axis=-1)

    def log_prob(self, value: chex.Array) -> chex.Array:
        """Log-probability of `value` (integer index).

        value shape: (...)  -- same leading dims as logits minus last.
        """
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        # gather along last axis
        return jnp.take_along_axis(log_probs, value[..., None], axis=-1).squeeze(-1)

    def entropy(self) -> chex.Array:
        """Shannon entropy, shape (...)."""
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return -jnp.sum(self.probs * log_probs, axis=-1)


# ============================================
# Actor-Critic network
# ============================================


class ActorCritic(nn.Module):
    """Separate policy and value MLPs.

    action_dim : number of discrete actions (49 for MSSA)
    hidden_dims: tuple of hidden layer widths (default matches SB3 config)
    """

    action_dim: int
    hidden_dims: Tuple[int, ...] = (512, 512, 256)

    @nn.compact
    def __call__(
        self, x: chex.Array, action_mask: Optional[chex.Array] = None
    ) -> Tuple[chex.Array, chex.Array]:
        """Forward pass.

        Args:
            x:           observation  (..., obs_dim)
            action_mask: bool mask    (..., action_dim).  Invalid actions get
                         -1e9 logits so they have ~0 probability.

        Returns:
            logits: (..., action_dim)
            value:  (...,)
        """
        # --- policy head ---
        h = x
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.relu(h)
        logits = nn.Dense(self.action_dim)(h)  # (..., action_dim)

        if action_mask is not None:
            logits = jnp.where(action_mask, logits, -1e9)

        # --- value head (fully separate weights) ---
        v = x
        for dim in self.hidden_dims:
            v = nn.Dense(dim)(v)
            v = nn.relu(v)
        value = nn.Dense(1)(v).squeeze(-1)  # (...,)

        return logits, value


# ============================================
# PPO data structures
# ============================================


class Transition(NamedTuple):
    """One collected transition (can be batched along leading dims)."""

    obs: chex.Array  # (obs_dim,)
    action: chex.Array  # scalar int32
    reward: chex.Array  # scalar float32
    done: chex.Array  # scalar bool
    value: chex.Array  # scalar float32  (V(s) at collection time)
    log_prob: chex.Array  # scalar float32
    action_mask: chex.Array  # (action_dim,) bool


class PPOConfig(NamedTuple):
    """Hyperparameters -- all plain Python types (not traced)."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 256  # rollout horizon per env
    n_minibatches: int = 8
    n_epochs: int = 4
    anneal_lr: bool = False  # curriculum script handles LR


# ============================================
# GAE
# ============================================


@jax.jit
def compute_gae(
    rewards: chex.Array,  # (T,)
    values: chex.Array,  # (T,)
    dones: chex.Array,  # (T,)  bool
    next_value: chex.Array,  # scalar
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[chex.Array, chex.Array]:
    """Standard GAE with reverse scan.  Returns (advantages, returns)."""

    def _step(carry, t):
        gae, nxt_val = carry
        r, v, d = rewards[t], values[t], dones[t]
        delta = r + gamma * nxt_val * (1.0 - d) - v
        gae = delta + gamma * gae_lambda * (1.0 - d) * gae
        return (gae, v), gae

    T = rewards.shape[0]
    indices = jnp.arange(T - 1, -1, -1)  # reverse order

    (_, _), advantages_rev = lax.scan(_step, (jnp.float32(0.0), next_value), indices)
    advantages = advantages_rev[::-1]  # back to forward order
    returns = advantages + values
    return advantages, returns


# ============================================
# Training state helpers
# ============================================


def create_train_state(
    key: chex.PRNGKey,
    network: ActorCritic,
    config: PPOConfig,
    obs_shape: Tuple[int, ...],
) -> TrainState:
    """Initialise Flax model + optax optimiser, return a TrainState."""
    dummy_obs = jnp.zeros((1,) + obs_shape)
    dummy_mask = jnp.ones((1, network.action_dim), dtype=jnp.bool_)
    params = network.init(key, dummy_obs, dummy_mask)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )

    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


# ============================================
# PPO loss + update
# ============================================


def make_ppo_update(network: ActorCritic, config: PPOConfig) -> Callable:
    """Factory that returns a JIT-compiled full PPO update function.

    The returned function signature:
        ppo_update(train_state, transitions, next_value, key)
            -> (new_train_state, metrics_dict)

    `transitions` is a Transition with leading shape (T * n_envs,)
    (already flattened by the caller).
    """

    @jax.jit
    def _single_update(
        train_state: TrainState,
        batch: Transition,
        advantages: chex.Array,
        returns: chex.Array,
    ) -> Tuple[TrainState, dict]:
        """One gradient step on a minibatch."""

        def loss_fn(params):
            logits, value = network.apply(params, batch.obs, batch.action_mask)
            pi = Categorical(logits)

            log_prob_new = pi.log_prob(batch.action)
            ratio = jnp.exp(log_prob_new - batch.log_prob)

            # normalise advantages within minibatch
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # clipped surrogate
            pg1 = -adv_norm * ratio
            pg2 = -adv_norm * jnp.clip(
                ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps
            )
            policy_loss = jnp.maximum(pg1, pg2).mean()

            value_loss = 0.5 * jnp.mean((value - returns) ** 2)
            entropy = pi.entropy().mean()

            total_loss = (
                policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
            )

            return total_loss, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "total_loss": total_loss,
                "approx_kl": jnp.mean((ratio - 1.0) - jnp.log(ratio + 1e-8)),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.params
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, metrics

    @jax.jit
    def ppo_update(
        train_state: TrainState,
        transitions: Transition,  # (B,) flat batch
        next_value: chex.Array,  # scalar
        key: chex.PRNGKey,
    ) -> Tuple[TrainState, dict]:
        # --- GAE ---
        advantages, returns = compute_gae(
            transitions.reward,
            transitions.value,
            transitions.done.astype(jnp.float32),
            next_value,
            config.gamma,
            config.gae_lambda,
        )

        B = transitions.obs.shape[0]
        minibatch_size = B // config.n_minibatches

        def _epoch(carry, _):
            ts, key = carry
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, B)

            def _minibatch(ts, start):
                idx = lax.dynamic_slice(perm, (start,), (minibatch_size,))
                mb = jax.tree.map(lambda x: x[idx], transitions)
                mb_adv = advantages[idx]
                mb_ret = returns[idx]
                ts, metrics = _single_update(ts, mb, mb_adv, mb_ret)
                return ts, metrics

            starts = jnp.arange(0, B, minibatch_size)  # (n_minibatches,)
            ts, metrics = lax.scan(_minibatch, ts, starts)
            return (ts, key), metrics

        (train_state, _), all_metrics = lax.scan(
            _epoch, (train_state, key), None, length=config.n_epochs
        )

        # average metrics over epochs and minibatches
        metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
        return train_state, metrics

    return ppo_update


# ============================================
# Rollout collection
# ============================================


def make_rollout_fn(env, network: ActorCritic, config: PPOConfig) -> Callable:
    """Return a JIT-compiled function that collects n_steps of experience
    from a SINGLE environment.

    For vectorised (multi-env) collection the caller should vmap this
    function over (env_states, obs, keys).
    """
    n_steps = config.n_steps

    @jax.jit
    def collect_rollout(
        train_state: TrainState, env_state, obs: chex.Array, key: chex.PRNGKey
    ) -> Tuple[Transition, chex.Array, any, chex.PRNGKey]:
        """Collect n_steps transitions.

        Returns (transitions, final_obs, final_env_state, final_key).
        transitions has leading dim (n_steps,).
        """

        def _step(carry, _):
            env_state, obs, key = carry
            key, act_key = jax.random.split(key)

            # action mask
            action_mask = env.get_action_mask(env_state)  # (49,)

            # forward pass -- single env so add/remove batch dim
            logits, value = network.apply(
                train_state.params,
                obs[None],  # (1, obs_dim)
                action_mask[None],  # (1, 49)
            )
            logits = logits[0]  # (49,)
            value = value[0]  # scalar

            pi = Categorical(logits)
            action = pi.sample(seed=act_key)
            log_prob = pi.log_prob(action)

            # env step
            next_obs, next_state, reward, done = env.step(env_state, action)

            # if done, auto-reset (we still store the terminal transition)
            key, reset_key = jax.random.split(key)
            reset_obs, reset_state = env.reset(reset_key)
            next_obs = jnp.where(done, reset_obs, next_obs)
            next_state = jax.tree.map(
                lambda r, n: jnp.where(done, r, n), reset_state, next_state
            )

            t = Transition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                action_mask=action_mask,
            )
            return (next_state, next_obs, key), t

        (final_state, final_obs, final_key), transitions = lax.scan(
            _step,
            (env_state, obs, key),
            None,
            length=n_steps,
        )

        return transitions, final_obs, final_state, final_key

    return collect_rollout


# ============================================
# Full training loop (single-env, used by
# curriculum script which handles the
# multi-env vmap externally)
# ============================================


def train(
    env,
    network: ActorCritic,
    config: PPOConfig,
    n_total_steps: int,
    key: chex.PRNGKey,
    callback: Optional[Callable] = None,
):
    """Run PPO on a single environment instance.

    For multi-env training use the curriculum script which vmaps the
    rollout collection and batches the updates.
    """
    key, init_key, env_key = jax.random.split(key, 3)

    train_state = create_train_state(init_key, network, config, env.obs_shape)
    obs, env_state = env.reset(env_key)

    ppo_update = make_ppo_update(network, config)
    collect = make_rollout_fn(env, network, config)

    n_updates = n_total_steps // config.n_steps

    for update_idx in range(n_updates):
        key, rollout_key, update_key = jax.random.split(key, 3)

        transitions, obs, env_state, key = collect(
            train_state, env_state, obs, rollout_key
        )

        # bootstrap value
        mask = env.get_action_mask(env_state)
        _, next_value = network.apply(train_state.params, obs[None], mask[None])
        next_value = next_value[0]

        # flatten: (n_steps,) already flat for single env
        train_state, metrics = ppo_update(
            train_state, transitions, next_value, update_key
        )

        if callback is not None:
            callback(update_idx, metrics, transitions)

    return train_state
