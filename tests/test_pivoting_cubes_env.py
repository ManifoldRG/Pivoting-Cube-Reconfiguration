import numpy as np
import pytest
from env.pivoting_cubes_env import PivotingCubesEnv

def test_env_basic():
    initial_positions = {1: (3, 3, 3), 2: (3, 4, 3)}
    final_positions = {1: (3, 3, 3), 2: (4, 3, 3)}
    n_modules = 2
    env = PivotingCubesEnv(initial_positions, final_positions, n_modules, empathy_lambda=0.1, max_episode_steps=5)
    obs, infos = env.reset()
    assert set(obs.keys()) == {"module_1", "module_2"}
    for agent, ob in obs.items():
        assert "observation" in ob and ob["observation"].shape == (5, 5, 5)
        assert "action_mask" in ob and ob["action_mask"].shape == (49,)
    # Take a step with NO-OP for both
    actions = {agent: 0 for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert all(isinstance(r, float) for r in rewards.values())
    assert all(isinstance(t, bool) for t in terminations.values())
    assert all(isinstance(t, bool) for t in truncations.values())

def test_env_connectivity():
    initial_positions = {1: (3, 3, 3), 2: (3, 4, 3)}
    final_positions = {1: (3, 3, 3), 2: (4, 3, 3)}
    n_modules = 2
    env = PivotingCubesEnv(initial_positions, final_positions, n_modules, empathy_lambda=0.1, max_episode_steps=5)
    obs, infos = env.reset()
    # Try to move both modules apart in a way that would disconnect them
    # Find a legal action for each agent (other than NO-OP)
    actions = {}
    for agent in env.agents:
        mask = obs[agent]["action_mask"]
        legal_actions = np.where(mask)[0]
        # Pick the first non-zero action if available
        non_noop = [a for a in legal_actions if a != 0]
        actions[agent] = non_noop[0] if non_noop else 0
    # Step should reject moves if they disconnect
    obs2, rewards, terminations, truncations, infos = env.step(actions)
    # The modules should not have moved apart if it would disconnect
    positions = env.ogm.module_positions
    dist = np.sum(np.abs(np.subtract(positions[1], positions[2])))
    assert dist <= 2, "Modules should remain connected after step"

def test_env_truncation():
    initial_positions = {1: (3, 3, 3), 2: (3, 4, 3)}
    final_positions = {1: (3, 3, 3), 2: (4, 3, 3)}
    n_modules = 2
    env = PivotingCubesEnv(initial_positions, final_positions, n_modules, empathy_lambda=0.1, max_episode_steps=2)
    obs, infos = env.reset()
    for _ in range(3):
        actions = {agent: 0 for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
    # After 2 steps, truncations should be True
    assert all(truncations.values()), "Truncation should occur after max_episode_steps" 