"""
PPO Implementation in JAX for TPU Training.

This implements Proximal Policy Optimization with action masking,
compatible with the JAX OGM environment for TPU-accelerated training.

Based on PureJaxRL patterns for maximum performance.
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
# Pure JAX Categorical Distribution
# (Replaces distrax to avoid version conflicts)
# ============================================

class Categorical:
    """
    Pure JAX implementation of Categorical distribution.
    Compatible with newer JAX versions (0.5+).
    """
    
    def __init__(self, logits: chex.Array):
        self.logits = logits
        self._probs = None
    
    @property
    def probs(self) -> chex.Array:
        if self._probs is None:
            self._probs = jax.nn.softmax(self.logits, axis=-1)
        return self._probs
    
    def sample(self, seed: chex.PRNGKey) -> chex.Array:
        """Sample from the distribution."""
        return jax.random.categorical(seed, self.logits, axis=-1)
    
    def log_prob(self, value: chex.Array) -> chex.Array:
        """Compute log probability of value."""
        # Normalize logits for numerical stability
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        # Index into log_probs at the given value
        return jnp.take_along_axis(log_probs, value[..., None], axis=-1).squeeze(-1)
    
    def entropy(self) -> chex.Array:
        """Compute entropy of the distribution."""
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return -jnp.sum(self.probs * log_probs, axis=-1)


# ============================================
# Neural Network Architecture
# ============================================

class ActorCritic(nn.Module):
    """
    Actor-Critic network with separate policy and value heads.
    Supports action masking for MaskablePPO-style training.
    """
    action_dim: int
    hidden_dims: Tuple[int, ...] = (512, 512, 256)
    
    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        action_mask: Optional[chex.Array] = None
    ) -> Tuple[Categorical, chex.Array]:
        # Shared trunk (optional - set to empty for separate networks)
        
        # Actor network
        actor = x
        for dim in self.hidden_dims:
            actor = nn.Dense(dim)(actor)
            actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim)(actor)
        
        # Apply action mask (set invalid actions to -inf)
        if action_mask is not None:
            logits = jnp.where(action_mask, logits, -1e10)
        
        # Create distribution (using our pure JAX implementation)
        pi = Categorical(logits=logits)
        
        # Critic network (separate)
        critic = x
        for dim in self.hidden_dims:
            critic = nn.Dense(dim)(critic)
            critic = nn.relu(critic)
        value = nn.Dense(1)(critic)
        
        return pi, jnp.squeeze(value, axis=-1)


# ============================================
# PPO Data Structures
# ============================================

class Transition(NamedTuple):
    """Single transition for PPO training."""
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    action_mask: chex.Array


class PPOConfig(NamedTuple):
    """PPO hyperparameter configuration."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 128
    n_minibatches: int = 4
    n_epochs: int = 4
    anneal_lr: bool = True


# ============================================
# GAE Computation
# ============================================

@jax.jit
def compute_gae(
    rewards: chex.Array,
    values: chex.Array,
    dones: chex.Array,
    next_value: chex.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[chex.Array, chex.Array]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: (T,) reward sequence
        values: (T,) value estimates
        dones: (T,) done flags
        next_value: Value estimate for final state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    def gae_step(carry, transition):
        gae, next_value = carry
        reward, value, done = transition
        
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        
        return (gae, value), gae
    
    # Reverse scan through transitions
    transitions = (rewards, values, dones)
    _, advantages = lax.scan(
        gae_step,
        (jnp.zeros_like(next_value), next_value),
        transitions,
        reverse=True
    )
    
    returns = advantages + values
    
    return advantages, returns


# ============================================
# PPO Update
# ============================================

def make_ppo_update(
    network: ActorCritic,
    config: PPOConfig
) -> Callable:
    """
    Create PPO update function.
    
    Args:
        network: Actor-Critic network
        config: PPO configuration
        
    Returns:
        Update function
    """
    
    @jax.jit
    def update_step(
        train_state: TrainState,
        batch: Transition,
        advantages: chex.Array,
        returns: chex.Array
    ) -> Tuple[TrainState, dict]:
        """Single PPO update step."""
        
        def loss_fn(params):
            # Forward pass
            pi, value = network.apply(params, batch.obs, batch.action_mask)
            
            # Policy loss
            log_prob = pi.log_prob(batch.action)
            ratio = jnp.exp(log_prob - batch.log_prob)
            
            # Normalize advantages
            advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Clipped surrogate objective
            pg_loss1 = -advantages_norm * ratio
            pg_loss2 = -advantages_norm * jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
            
            # Value loss
            value_loss = 0.5 * ((value - returns) ** 2).mean()
            
            # Entropy bonus
            entropy = pi.entropy().mean()
            
            # Total loss
            total_loss = (
                pg_loss +
                config.vf_coef * value_loss -
                config.ent_coef * entropy
            )
            
            return total_loss, {
                "policy_loss": pg_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "total_loss": total_loss,
                "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
            }
        
        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        
        # Clip gradients
        grads = jax.tree_map(
            lambda g: jnp.clip(g, -config.max_grad_norm, config.max_grad_norm),
            grads
        )
        
        # Update parameters
        train_state = train_state.apply_gradients(grads=grads)
        
        return train_state, metrics
    
    def ppo_update(
        train_state: TrainState,
        transitions: Transition,
        next_value: chex.Array,
        key: chex.PRNGKey
    ) -> Tuple[TrainState, dict]:
        """Full PPO update with multiple epochs and minibatches."""
        
        # Compute GAE
        advantages, returns = compute_gae(
            transitions.reward,
            transitions.value,
            transitions.done,
            next_value,
            config.gamma,
            config.gae_lambda
        )
        
        # Flatten batch dimensions
        batch_size = transitions.obs.shape[0]
        minibatch_size = batch_size // config.n_minibatches
        
        def epoch_update(carry, _):
            train_state, key = carry
            key, subkey = jax.random.split(key)
            
            # Shuffle data
            perm = jax.random.permutation(subkey, batch_size)
            
            def minibatch_update(train_state, start_idx):
                idx = perm[start_idx:start_idx + minibatch_size]
                
                minibatch = jax.tree_map(lambda x: x[idx], transitions)
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                train_state, metrics = update_step(
                    train_state, minibatch, mb_advantages, mb_returns
                )
                
                return train_state, metrics
            
            train_state, metrics = lax.scan(
                minibatch_update,
                train_state,
                jnp.arange(0, batch_size, minibatch_size)
            )
            
            return (train_state, key), metrics
        
        # Run multiple epochs
        (train_state, _), all_metrics = lax.scan(
            epoch_update,
            (train_state, key),
            None,
            length=config.n_epochs
        )
        
        # Average metrics across epochs and minibatches
        metrics = jax.tree_map(lambda x: x.mean(), all_metrics)
        
        return train_state, metrics
    
    return ppo_update


# ============================================
# Training Loop
# ============================================

def create_train_state(
    key: chex.PRNGKey,
    network: ActorCritic,
    config: PPOConfig,
    obs_shape: Tuple[int, ...]
) -> TrainState:
    """Create initial training state."""
    dummy_obs = jnp.zeros((1,) + obs_shape)
    dummy_mask = jnp.ones((1, network.action_dim), dtype=jnp.bool_)
    
    params = network.init(key, dummy_obs, dummy_mask)
    
    if config.anneal_lr:
        # Linear learning rate annealing
        schedule = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0.0,
            transition_steps=1_000_000  # Adjust based on total steps
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=schedule)
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=config.learning_rate)
        )
    
    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optimizer
    )


def make_rollout_fn(
    env,
    network: ActorCritic,
    n_steps: int
) -> Callable:
    """
    Create rollout collection function.
    
    Args:
        env: Vectorized environment
        network: Actor-Critic network
        n_steps: Number of steps to collect
        
    Returns:
        Rollout function
    """
    
    @jax.jit
    def collect_rollout(
        train_state: TrainState,
        env_state,
        obs: chex.Array,
        agent_idx: int,
        key: chex.PRNGKey
    ) -> Tuple[Transition, chex.Array, any, int]:
        """Collect n_steps of experience."""
        
        def step_fn(carry, _):
            env_state, obs, agent_idx, key = carry
            key, action_key = jax.random.split(key)
            
            # Get action mask
            action_mask = env.get_action_mask(env_state, agent_idx)
            
            # Forward pass
            pi, value = network.apply(train_state.params, obs[None], action_mask[None])
            
            # Sample action
            action = pi.sample(seed=action_key)[0]
            log_prob = pi.log_prob(action)[0]
            
            # Environment step
            next_obs, env_state, reward, done, info = env.step(env_state, action, agent_idx)
            
            # Update agent index
            next_agent = (agent_idx + 1) % env_state.n
            
            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                value=value[0],
                log_prob=log_prob,
                action_mask=action_mask
            )
            
            return (env_state, next_obs, next_agent, key), transition
        
        (env_state, obs, agent_idx, key), transitions = lax.scan(
            step_fn,
            (env_state, obs, agent_idx, key),
            None,
            length=n_steps
        )
        
        return transitions, obs, env_state, agent_idx
    
    return collect_rollout


# ============================================
# Full Training Function
# ============================================

def train(
    env,
    network: ActorCritic,
    config: PPOConfig,
    n_total_steps: int,
    key: chex.PRNGKey,
    callback: Optional[Callable] = None
):
    """
    Full PPO training loop.
    
    Args:
        env: Vectorized environment
        network: Actor-Critic network
        config: PPO configuration
        n_total_steps: Total training steps
        key: Random key
        callback: Optional callback function
    """
    key, init_key, env_key = jax.random.split(key, 3)
    
    # Initialize
    train_state = create_train_state(init_key, network, config, env.obs_shape)
    obs, env_state = env.reset(env_key)
    agent_idx = 0
    
    # Create update functions
    ppo_update = make_ppo_update(network, config)
    collect_rollout = make_rollout_fn(env, network, config.n_steps)
    
    # Training loop
    n_updates = n_total_steps // config.n_steps
    
    for update in range(n_updates):
        key, rollout_key, update_key = jax.random.split(key, 3)
        
        # Collect rollout
        transitions, obs, env_state, agent_idx = collect_rollout(
            train_state, env_state, obs, agent_idx, rollout_key
        )
        
        # Get value for final state
        action_mask = env.get_action_mask(env_state, agent_idx)
        _, next_value = network.apply(train_state.params, obs[None], action_mask[None])
        
        # PPO update
        train_state, metrics = ppo_update(
            train_state, transitions, next_value[0], update_key
        )
        
        # Callback
        if callback is not None:
            callback(update, metrics, transitions)
    
    return train_state
