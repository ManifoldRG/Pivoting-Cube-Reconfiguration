"""
JAX-based Occupancy Grid Map for TPU-accelerated training.

This is a JAX reimplementation of the NumPy-based OccupancyGridMap for
running on TPUs with PureJaxRL-style vectorized training.

Key differences from NumPy version:
- All operations use jax.numpy instead of numpy
- State is immutable (functional updates)
- JIT-compilable for maximum performance
- Vectorizable via vmap for thousands of parallel environments
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import NamedTuple, Tuple, Optional
import chex


# ============================================
# Data Structures
# ============================================

class OGMState(NamedTuple):
    """Immutable state for the Occupancy Grid Map."""
    module_positions: chex.Array  # (n, 3) - xyz positions of each module
    final_positions: chex.Array   # (n, 3) - target positions
    n: int                        # Number of modules
    step_count: int               # Current step
    max_steps: int                # Maximum steps per episode


class OGMConfig(NamedTuple):
    """Configuration for the environment (static, not traced)."""
    n: int                        # Number of modules
    max_steps: int                # Maximum steps per episode
    grid_size: int                # Size of the grid (default 50x50)
    use_unlabeled: bool           # Whether to use unlabeled rewards
    local_k: int                  # Local neighborhood size


# ============================================
# Actions
# ============================================

# 48 pivot actions + 1 no-op = 49 total
# Each action is (axis, direction, rotation_direction)
# axis: 0=x, 1=y, 2=z
# direction: -1 or +1 (which side of cube)
# rotation: -1 or +1 (which way to rotate)

# Pre-computed action deltas for all 48 pivot moves
# Shape: (48, 3) - displacement in x, y, z for each action
# NOTE: Using numpy array at module level to avoid JAX initialization at import time
# This is converted to JAX array lazily when first used
import numpy as np

_ACTION_DELTAS_NP = np.array([
    # X-axis pivots (16 actions)
    [-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1],
    [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
    [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1],
    [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1],
    # Y-axis pivots (16 actions)
    [-1, -1, -1], [-1, -1, 1], [-1, 0, -1], [-1, 0, 1],
    [-1, 1, -1], [-1, 1, 1], [0, -1, -1], [0, -1, 1],
    [0, 1, -1], [0, 1, 1], [1, -1, -1], [1, -1, 1],
    [1, 0, -1], [1, 0, 1], [1, 1, -1], [1, 1, 1],
    # Z-axis pivots (16 actions)
    [-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1],
    [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
    [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1],
    [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1],
], dtype=np.int32)

NUM_ACTIONS = 49  # 48 pivot + 1 no-op

# Lazy initialization of JAX array
_ACTION_DELTAS_JAX = None

def get_action_deltas():
    """Get ACTION_DELTAS as a JAX array (lazy initialization)."""
    global _ACTION_DELTAS_JAX
    if _ACTION_DELTAS_JAX is None:
        _ACTION_DELTAS_JAX = jnp.array(_ACTION_DELTAS_NP)
    return _ACTION_DELTAS_JAX


# ============================================
# Core Functions
# ============================================

@jax.jit
def compute_pairwise_norms(positions: chex.Array) -> chex.Array:
    """
    Compute pairwise L2 distances between all module positions.
    
    Args:
        positions: (n, 3) array of module positions
        
    Returns:
        (n, n) array of pairwise distances
    """
    # Expand dimensions for broadcasting
    pos_i = positions[:, None, :]  # (n, 1, 3)
    pos_j = positions[None, :, :]  # (1, n, 3)
    
    # Compute squared differences
    diff = pos_i - pos_j  # (n, n, 3)
    sq_dist = jnp.sum(diff ** 2, axis=-1)  # (n, n)
    
    return jnp.sqrt(sq_dist)


@jax.jit
def compute_relative_norms(current: chex.Array, target: chex.Array) -> chex.Array:
    """
    Compute element-wise difference between current and target pairwise norms.
    
    Args:
        current: (n, n) current pairwise distances
        target: (n, n) target pairwise distances
        
    Returns:
        (n, n) relative distances (0 means at target)
    """
    return jnp.abs(current - target)


@jax.jit
def calc_four_band_reduction(pairwise_norms: chex.Array) -> chex.Array:
    """
    Reduce n×n pairwise norms to n×4 using diagonal bands.
    
    This maintains a constant observation size across different n values,
    which is critical for curriculum learning.
    
    Args:
        pairwise_norms: (n, n) pairwise distance matrix
        
    Returns:
        (n, 4) reduced observation
    """
    n = pairwise_norms.shape[0]
    
    def get_band_value(i: int, offset: int) -> float:
        j = (i + offset) % n
        return pairwise_norms[i, j]
    
    # Extract 4 bands: offsets -2, -1, +1, +2
    bands = []
    for offset in [-2, -1, 1, 2]:
        band = jax.vmap(lambda i: pairwise_norms[i, (i + offset) % n])(jnp.arange(n))
        bands.append(band)
    
    return jnp.stack(bands, axis=1)  # (n, 4)


@jax.jit
def calc_local_neighborhood(
    reduced_norms: chex.Array,
    agent_idx: int,
    k: int
) -> chex.Array:
    """
    Extract local neighborhood centered on the acting agent.
    
    Args:
        reduced_norms: (n, 4) four-band reduced observation
        agent_idx: Index of the acting agent (0-indexed)
        k: Size of local neighborhood
        
    Returns:
        (k, 4) local neighborhood observation
    """
    n = reduced_norms.shape[0]
    half_k = k // 2
    
    # Generate indices centered around agent
    offsets = jnp.arange(k) - half_k
    indices = (agent_idx + offsets) % n
    
    return reduced_norms[indices]


@jax.jit
def get_neighbors(positions: chex.Array, idx: int) -> chex.Array:
    """
    Get mask of which modules are neighbors (Manhattan distance = 1).
    
    Args:
        positions: (n, 3) module positions
        idx: Index of module to check neighbors for
        
    Returns:
        (n,) boolean mask of neighbors
    """
    pos = positions[idx]
    diffs = jnp.abs(positions - pos)
    manhattan = jnp.sum(diffs, axis=1)
    
    # Neighbor if Manhattan distance is exactly 1
    return (manhattan == 1)


@jax.jit
def check_connectivity(positions: chex.Array) -> bool:
    """
    Check if all modules form a single connected component.
    Uses iterative BFS that's JAX-compatible.
    
    Args:
        positions: (n, 3) module positions
        
    Returns:
        True if connected, False otherwise
    """
    n = positions.shape[0]
    
    # Build adjacency matrix
    def is_neighbor(i, j):
        diff = jnp.abs(positions[i] - positions[j])
        return jnp.sum(diff) == 1
    
    adj = jax.vmap(lambda i: jax.vmap(lambda j: is_neighbor(i, j))(jnp.arange(n)))(jnp.arange(n))
    
    # BFS using matrix multiplication
    visited = jnp.zeros(n, dtype=jnp.bool_)
    visited = visited.at[0].set(True)
    
    def bfs_step(visited, _):
        # Expand to neighbors
        new_visited = jnp.any(adj & visited[None, :], axis=1) | visited
        return new_visited, None
    
    visited, _ = lax.scan(bfs_step, visited, None, length=n)
    
    return jnp.all(visited)


@jax.jit
def is_articulation_point(positions: chex.Array, idx: int) -> bool:
    """
    Check if removing a module disconnects the graph.
    
    Args:
        positions: (n, 3) module positions
        idx: Index of module to check
        
    Returns:
        True if module is an articulation point
    """
    n = positions.shape[0]
    
    # Create mask excluding the module
    mask = jnp.arange(n) != idx
    remaining_positions = positions[mask]
    
    # Check if remaining modules are connected
    return ~check_connectivity(remaining_positions)


@jax.jit
def get_action_mask(
    state: OGMState,
    agent_idx: int
) -> chex.Array:
    """
    Compute valid action mask for an agent.
    
    Args:
        state: Current environment state
        agent_idx: Index of the acting agent
        
    Returns:
        (49,) boolean mask of valid actions
    """
    positions = state.module_positions
    n = state.n
    
    # Start with all actions invalid
    mask = jnp.zeros(NUM_ACTIONS, dtype=jnp.bool_)
    
    # No-op is always valid
    mask = mask.at[48].set(True)
    
    # Check each pivot action
    def check_action(action_idx: int) -> bool:
        if action_idx >= 48:
            return True  # No-op
            
        delta = get_action_deltas()[action_idx]
        new_pos = positions[agent_idx] + delta
        
        # Check 1: Position not already occupied
        occupied = jnp.any(jnp.all(positions == new_pos, axis=1))
        
        # Check 2: Would remain connected (simplified check)
        # For full check, would need to verify pivot zone
        
        return ~occupied
    
    mask = jax.vmap(check_action)(jnp.arange(NUM_ACTIONS))
    
    return mask


# ============================================
# Environment Step
# ============================================

@partial(jax.jit, static_argnums=(2,))
def step(
    state: OGMState,
    action: int,
    agent_idx: int
) -> Tuple[OGMState, chex.Array, bool, bool, dict]:
    """
    Execute one step in the environment.
    
    Args:
        state: Current state
        action: Action to take (0-48)
        agent_idx: Which agent is acting
        
    Returns:
        Tuple of (new_state, reward, terminated, truncated, info)
    """
    positions = state.module_positions
    
    # Apply action
    new_positions = lax.cond(
        action < 48,
        lambda: positions.at[agent_idx].add(get_action_deltas()[action]),
        lambda: positions
    )
    
    # Update state
    new_state = state._replace(
        module_positions=new_positions,
        step_count=state.step_count + 1
    )
    
    # Compute reward (will be expanded below)
    reward = compute_reward(state, new_state)
    
    # Check termination
    done = check_success(new_state)
    truncated = new_state.step_count >= new_state.max_steps
    
    info = {"success": done}
    
    return new_state, reward, done, truncated, info


# ============================================
# Reward Functions
# ============================================

@jax.jit
def compute_potential(state: OGMState, use_unlabeled: bool = True) -> float:
    """
    Compute potential-based reward.
    
    Args:
        state: Current state
        use_unlabeled: Whether to use unlabeled (shape) matching
        
    Returns:
        Potential value (lower is better, 0 is optimal)
    """
    curr_norms = compute_pairwise_norms(state.module_positions)
    final_norms = compute_pairwise_norms(state.final_positions)
    
    if use_unlabeled:
        # Use sorted signatures for unlabeled matching
        curr_sorted = jnp.sort(curr_norms, axis=1)
        final_sorted = jnp.sort(final_norms, axis=1)
        diff = jnp.abs(curr_sorted - final_sorted)
    else:
        diff = jnp.abs(curr_norms - final_norms)
    
    return jnp.sum(diff)


@jax.jit
def compute_reward(
    prev_state: OGMState,
    curr_state: OGMState,
    step_cost: float = -0.005,
    potential_scale: float = 1.0,
    success_bonus: float = 100.0
) -> float:
    """
    Compute shaped reward for transition.
    
    Args:
        prev_state: Previous state
        curr_state: Current state
        step_cost: Per-step penalty
        potential_scale: Scale for potential-based shaping
        success_bonus: Reward for reaching goal
        
    Returns:
        Total reward
    """
    # Base step cost
    reward = step_cost
    
    # Potential-based shaping (F = gamma * phi' - phi)
    prev_potential = compute_potential(prev_state)
    curr_potential = compute_potential(curr_state)
    shaping = potential_scale * (prev_potential - curr_potential)
    reward += shaping
    
    # Success bonus
    done = check_success(curr_state)
    reward = lax.cond(done, lambda: reward + success_bonus, lambda: reward)
    
    return reward


@jax.jit
def check_success(state: OGMState, tol: float = 1e-6) -> bool:
    """
    Check if goal configuration is reached.
    
    Uses unlabeled matching: checks if pairwise distance signatures match.
    """
    curr_norms = compute_pairwise_norms(state.module_positions)
    final_norms = compute_pairwise_norms(state.final_positions)
    
    # Sort each row to get shape signatures
    curr_sorted = jnp.sort(curr_norms, axis=1)
    final_sorted = jnp.sort(final_norms, axis=1)
    
    # Check if signatures match
    return jnp.allclose(curr_sorted, final_sorted, atol=tol)


# ============================================
# Reset / Initialization
# ============================================

def make_connected_configuration(
    key: chex.PRNGKey,
    n: int,
    grid_size: int = 50
) -> chex.Array:
    """
    Generate a random connected configuration of n modules.
    
    Args:
        key: JAX random key
        n: Number of modules
        grid_size: Size of the grid
        
    Returns:
        (n, 3) array of positions
    """
    # Start with one module at center
    center = grid_size // 2
    positions = jnp.array([[center, center, center]], dtype=jnp.int32)
    
    # Add modules one at a time, adjacent to existing ones
    def add_module(carry, key):
        positions, count = carry
        
        # Get all possible neighbor positions
        def get_neighbors_of(idx):
            pos = positions[idx]
            deltas = jnp.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]])
            return pos + deltas
        
        # Collect all neighbor positions
        all_neighbors = jax.vmap(get_neighbors_of)(jnp.arange(count))
        all_neighbors = all_neighbors.reshape(-1, 3)
        
        # Filter out occupied positions
        def is_free(pos):
            return ~jnp.any(jnp.all(positions[:count] == pos, axis=1))
        
        free_mask = jax.vmap(is_free)(all_neighbors)
        
        # Select random free position
        probs = free_mask.astype(jnp.float32)
        probs = probs / (probs.sum() + 1e-8)
        idx = jax.random.choice(key, jnp.arange(len(all_neighbors)), p=probs)
        new_pos = all_neighbors[idx]
        
        # Add to positions
        positions = positions.at[count].set(new_pos)
        
        return (positions, count + 1), None
    
    # Pre-allocate space
    positions = jnp.zeros((n, 3), dtype=jnp.int32)
    positions = positions.at[0].set(jnp.array([center, center, center]))
    
    keys = jax.random.split(key, n - 1)
    (positions, _), _ = lax.scan(add_module, (positions, 1), keys)
    
    return positions


def reset(
    key: chex.PRNGKey,
    config: OGMConfig
) -> OGMState:
    """
    Reset environment to a new random configuration.
    
    Args:
        key: JAX random key
        config: Environment configuration
        
    Returns:
        Initial state
    """
    key1, key2 = jax.random.split(key)
    
    initial_positions = make_connected_configuration(key1, config.n)
    final_positions = make_connected_configuration(key2, config.n)
    
    return OGMState(
        module_positions=initial_positions,
        final_positions=final_positions,
        n=config.n,
        step_count=0,
        max_steps=config.max_steps
    )


# ============================================
# Observation
# ============================================

@partial(jax.jit, static_argnums=(2, 3))
def get_observation(
    state: OGMState,
    agent_idx: int,
    use_four_band: bool = True,
    local_k: int = 7
) -> chex.Array:
    """
    Get observation for an agent.
    
    Args:
        state: Current state
        agent_idx: Which agent is observing
        use_four_band: Whether to use four-band reduction
        local_k: Size of local neighborhood
        
    Returns:
        Flat observation array
    """
    # Compute relative pairwise norms
    curr_norms = compute_pairwise_norms(state.module_positions)
    final_norms = compute_pairwise_norms(state.final_positions)
    rel_norms = compute_relative_norms(curr_norms, final_norms)
    
    # Apply four-band reduction
    if use_four_band:
        rel_norms = calc_four_band_reduction(rel_norms)
    
    # Apply local neighborhood
    obs = calc_local_neighborhood(rel_norms, agent_idx, local_k)
    
    # Flatten and normalize
    obs = obs.flatten()
    max_dist = jnp.sqrt(3.0) * 50  # Max possible distance in grid
    obs = obs / max_dist
    
    # Add agent encoding (fixed size = 4)
    agent_encoding = jnp.array([
        jnp.sin(2 * jnp.pi * agent_idx / state.n),
        jnp.cos(2 * jnp.pi * agent_idx / state.n),
        agent_idx / state.n,
        (state.n - agent_idx) / state.n
    ])
    
    return jnp.concatenate([obs, agent_encoding])


# ============================================
# Vectorized Environment (for PureJaxRL)
# ============================================

class VectorizedOGMEnv:
    """
    Vectorized environment for running many instances in parallel.
    
    Compatible with PureJaxRL-style training.
    """
    
    def __init__(self, config: OGMConfig):
        self.config = config
        self.obs_shape = (config.local_k * 4 + 4,)  # Four-band + agent encoding
        self.action_space = NUM_ACTIONS
        
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, OGMState]:
        """Reset environment and return initial observation."""
        state = reset(key, self.config)
        obs = get_observation(state, 0, True, self.config.local_k)
        return obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: OGMState,
        action: int,
        agent_idx: int
    ) -> Tuple[chex.Array, OGMState, float, bool, dict]:
        """Execute step and return (obs, state, reward, done, info)."""
        state, reward, done, truncated, info = step(state, action, agent_idx)
        
        # Next agent
        next_agent = (agent_idx + 1) % state.n
        obs = get_observation(state, next_agent, True, self.config.local_k)
        
        return obs, state, reward, done | truncated, info
    
    @partial(jax.jit, static_argnums=(0,))
    def get_action_mask(self, state: OGMState, agent_idx: int) -> chex.Array:
        """Get valid action mask for agent."""
        return get_action_mask(state, agent_idx)
