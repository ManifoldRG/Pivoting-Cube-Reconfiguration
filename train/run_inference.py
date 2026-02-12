"""
Inference script to run a trained model on custom initial/final configurations.

Usage:
    # Run with interactive input
    python train/run_inference.py --model runs/n8_improved/.../final_model.zip --num_agents 8

    # Run with config file
    python train/run_inference.py --model runs/model.zip --config configs/test_case.json

    # Run with inline configurations
    python train/run_inference.py --model runs/model.zip --num_agents 3 \
        --init "[[0,0,0], [1,0,0], [2,0,0]]" \
        --final "[[0,0,0], [0,1,0], [0,2,0]]"

Configuration format:
    - Each configuration is a list of [x, y, z] integer coordinates
    - Number of positions must match num_agents
    - Modules must be connected (6-neighborhood adjacency)
"""

import argparse
import json
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ogm.ogm_gym_env import OGMGymEnv
from ogm.occupancy_grid_map import OccupancyGridMap


class CustomConfigEnv(OGMGymEnv):
    """
    OGMGymEnv variant that allows setting custom configurations.
    """

    def __init__(self, init_config=None, final_config=None, **kwargs):
        super().__init__(**kwargs)
        self._custom_init = init_config
        self._custom_final = final_config

    def set_configurations(self, init_config, final_config):
        """Set custom initial and final configurations."""
        self._custom_init = init_config
        self._custom_final = final_config

    def reset(self, seed=None, options=None):
        """Reset with custom configurations if set."""
        if seed is not None:
            super().reset(seed=seed)

        if self._custom_init is not None and self._custom_final is not None:
            self.init_conf = self._custom_init
            self.final_conf = self._custom_final
        else:
            # Fall back to random configuration
            from ogm.random_configuration import random_configuration
            self.init_conf, self.final_conf, _ = random_configuration(self.num_agents)

        # Reset the OGM environment
        self.raw_obs = self.env.reset(self.init_conf, self.final_conf)

        # Reset internal state
        self.current_agent_idx = 0
        self.episode_done = False
        self.steps_taken = 0

        # Prepare grid map for first agent
        self.env.ogm.calc_pre_action_grid_map()

        obs = self._get_obs()
        info = {}

        return obs, info


def validate_configuration(config, num_agents):
    """Validate that a configuration is valid."""
    if len(config) != num_agents:
        raise ValueError(f"Configuration has {len(config)} positions but num_agents={num_agents}")

    # Check connectivity
    positions = set(tuple(p) for p in config)
    if len(positions) != num_agents:
        raise ValueError("Duplicate positions in configuration")

    # Simple connectivity check via BFS
    if num_agents > 1:
        visited = {tuple(config[0])}
        queue = [tuple(config[0])]
        neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

        while queue:
            pos = queue.pop(0)
            for dx, dy, dz in neighbors:
                neighbor = (pos[0]+dx, pos[1]+dy, pos[2]+dz)
                if neighbor in positions and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(visited) != num_agents:
            raise ValueError("Configuration is not connected (modules must share faces)")

    return True


def parse_config(config_str):
    """Parse configuration from string or file."""
    if os.path.isfile(config_str):
        with open(config_str, 'r') as f:
            return json.load(f)
    else:
        return json.loads(config_str)


def run_inference(model, env, max_steps=1000, verbose=True):
    """
    Run inference on a single episode.

    Returns:
        dict with results: success, steps, actions, positions_history
    """
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step = 0
    actions = []
    positions_history = [env.env.ogm.get_positions().copy()]

    if verbose:
        print(f"\nInitial configuration:")
        print(f"  {env.init_conf}")
        print(f"Target configuration:")
        print(f"  {env.final_conf}")
        print(f"\nRunning inference...")

    while not (done or truncated) and step < max_steps:
        # Get action mask
        action_masks = env.action_masks()

        # Predict action
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

        # Take step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        actions.append(int(action))

        # Record positions after each full phase (all agents acted)
        if env.current_agent_idx == 0:
            step += 1
            positions_history.append(env.env.ogm.get_positions().copy())

            if verbose and step % 50 == 0:
                success = env.env.ogm.check_final()
                print(f"  Step {step}: reward={total_reward:.3f}, success={success}")

    # Check final result
    success = env.env.ogm.check_final()

    if verbose:
        print(f"\nResult: {'SUCCESS' if success else 'FAILURE'}")
        print(f"  Steps: {step}")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"Final configuration:")
        print(f"  {env.env.ogm.get_positions()}")

    return {
        'success': success,
        'steps': step,
        'total_reward': total_reward,
        'actions': actions,
        'positions_history': positions_history,
        'init_config': env.init_conf,
        'final_config': env.final_conf,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained model on custom configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model arguments
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--vec_normalize", type=str, default=None,
        help="Path to VecNormalize stats (.pkl file). Auto-detected if not specified."
    )

    # Configuration arguments
    parser.add_argument(
        "--num_agents", type=int, required=True,
        help="Number of agents/modules"
    )
    parser.add_argument(
        "--init", type=str, default=None,
        help="Initial configuration as JSON array: [[x,y,z], ...]"
    )
    parser.add_argument(
        "--final", type=str, default=None,
        help="Final/target configuration as JSON array: [[x,y,z], ...]"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="JSON config file with 'init' and 'final' keys"
    )

    # Environment arguments (must match training settings)
    parser.add_argument(
        "--max_steps", type=int, default=1600,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--use_four_band_reduction", action="store_true", default=True,
        help="Use four-band dimension reduction"
    )
    parser.add_argument(
        "--use_local_neighborhood", action="store_true", default=True,
        help="Use local neighborhood reduction"
    )
    parser.add_argument(
        "--local_neighborhood_k", type=int, default=5,
        help="Local neighborhood size"
    )
    parser.add_argument(
        "--use_unlabeled_mode", action="store_true", default=True,
        help="Use unlabeled (label-agnostic) mode"
    )

    # Output arguments
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization GIF"
    )
    parser.add_argument(
        "--gif_path", type=str, default="inference_result.gif",
        help="Path for visualization GIF"
    )

    args = parser.parse_args()

    # Parse configurations
    init_config = None
    final_config = None

    if args.config:
        config_data = parse_config(args.config)
        init_config = config_data.get('init')
        final_config = config_data.get('final')
    elif args.init and args.final:
        init_config = parse_config(args.init)
        final_config = parse_config(args.final)

    # Validate configurations if provided
    if init_config is not None:
        validate_configuration(init_config, args.num_agents)
        print(f"Initial configuration validated: {len(init_config)} modules")
    if final_config is not None:
        validate_configuration(final_config, args.num_agents)
        print(f"Final configuration validated: {len(final_config)} modules")

    # Create environment
    print(f"\nCreating environment with {args.num_agents} agents...")
    env = CustomConfigEnv(
        init_config=init_config,
        final_config=final_config,
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        use_four_band_reduction=args.use_four_band_reduction,
        use_local_neighborhood=args.use_local_neighborhood,
        local_neighborhood_k=args.local_neighborhood_k,
        use_unlabeled_mode=args.use_unlabeled_mode,
        enable_potential_reward=True,
    )

    # Wrap in DummyVecEnv for SB3 compatibility
    vec_env = DummyVecEnv([lambda: env])

    # Load VecNormalize if available
    vec_normalize_path = args.vec_normalize
    if vec_normalize_path is None:
        # Try to auto-detect
        model_dir = os.path.dirname(args.model)
        potential_path = os.path.join(model_dir, "vec_normalize.pkl")
        if os.path.exists(potential_path):
            vec_normalize_path = potential_path

    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"Loading VecNormalize from: {vec_normalize_path}")
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False  # Disable updates during inference
        vec_env.norm_reward = False

    # Load model
    print(f"Loading model from: {args.model}")
    model = MaskablePPO.load(args.model, env=vec_env)

    # Get the unwrapped environment for direct access
    if hasattr(vec_env, 'envs'):
        base_env = vec_env.envs[0]
    else:
        base_env = vec_env.venv.envs[0]

    # Run inference
    results = run_inference(model, base_env, max_steps=args.max_steps, verbose=True)

    # Generate visualization if requested
    if args.visualize:
        try:
            from visualizer.step_visualizer import StepVisualizer
            print(f"\nGenerating visualization: {args.gif_path}")
            visualizer = StepVisualizer()
            visualizer.create_gif_from_positions(
                results['positions_history'],
                results['final_config'],
                args.gif_path
            )
            print(f"Visualization saved to: {args.gif_path}")
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")

    # Save results if requested
    if args.output:
        # Convert numpy arrays to lists for JSON serialization
        output_data = {
            'success': results['success'],
            'steps': results['steps'],
            'total_reward': results['total_reward'],
            'num_agents': args.num_agents,
            'init_config': [list(p) for p in results['init_config']],
            'final_config': [list(p) for p in results['final_config']],
            'actions': results['actions'],
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
