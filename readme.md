# Pivoting Cube Reconfiguration

## About the Project

PCR is a reinforcement learning framework for training modular robotic systems to reconfigure themselves on a 3D lattice. The project addresses the challenge of finding optimal action sequences that transform an initial configuration of connected modules into a desired target configuration through local pivot transformations.

### Problem Statement

The system operates on a 3D integer lattice where:
- Each position can be occupied by at most one module
- Modules maintain connectivity through 6-neighborhood adjacency (face-adjacent cells)
- Modules move via 49 distinct pivot operations (16 per plane: XY, XZ, YZ + 1 no-op)
- Success is defined by matching pairwise inter-module distances to the target configuration

The goal is to find a sequence of valid transformations that reconfigures modules from an initial state to the final state.

## Code Setup

### First Time Setup

1. Create the conda environment:
```bash
conda env create -f conda.yaml
```

2. Install additional dependencies for RL training:
```bash
pip install gymnasium sb3-contrib stable-baselines3
```

### Subsequent Usage

1. Activate the environment:
```bash
conda activate mssa
```

2. Set environment variables:
```bash
source env.sh
```

### Running Tests

Execute the test suite:
```bash
pytest
```

### Visualization

Visualize a reconfiguration path:
```bash
python tests/visualize_path.py
```

## OGM Architecture

The Occupancy Grid Map (OGM) is the core state representation and physics engine for the modular robot system.

### Components

**OccupancyGridMap** (`ogm/occupancy_grid_map.py`):
- 3D grid representation with dynamic sizing based on module count
- Module position tracking with automatic recentering
- Connectivity graph maintenance using 6-neighborhood adjacency
- Pairwise distance matrix computation and caching

**Key Operations**:

1. **Pivot Actions**: 49 local transformations
   - Each pivot moves a module around an adjacent module
   - Actions are validated for connectivity and collision
   
2. **Action Masking** (`init_actions`):
   - Pre-computes valid actions based on local geometry
   - Checks pivot zone constraints
   - Ensures post-action connectivity
   - Returns boolean mask of safe actions per module

3. **State Observations**:
   - Pairwise distance matrices (current and target)
   - Optional 4-band reduction: O(n) instead of O(n²) complexity
   - Optional local neighborhood masking for scalability
   - Agent ID encoding for multi-agent coordination

### Grid Management

- **Recentering**: Module 1 maintained at grid center to prevent boundary issues
- **Grid Sizing**: Dynamic sizing as `max(5, n*2+3)` where n is module count
- **Collision Detection**: Validated through grid occupancy checks
- **Connectivity Validation**: BFS-based connectivity verification after each action

## RL Architecture and Reward System

The project implements two complementary RL approaches for training reconfiguration policies.

### Training Frameworks

#### 1. Custom MAPPO (`agent/mappo_agent.py`, `train/train_mappo.py`)

Distance-based action selection with specialized architecture:

**Network Architecture**:
- Actor: `[obs + agent_id] → [hidden] → [hidden] → [n² target distances]`
- Critic: `[obs + agent_id] → [hidden] → [hidden] → [1 value]`
- Default: 2-layer MLP with 256 hidden units

**Action Selection Mechanism**:
1. Actor predicts target pairwise distance vector
2. Environment provides candidate distance vectors for all valid actions
3. Action selected by minimizing distance to target: `argmin ||candidate - target||`
4. Temperature parameter controls exploration vs. exploitation

**Key Features**:
- Auxiliary regression loss aligns predictions with chosen actions
- Sequential multi-agent execution
- Custom PPO implementation with GAE
- Distance-based inductive bias for better generalization

#### 2. Stable-Baselines3 (`train/train_sb3.py`)

Standard MaskablePPO with Gymnasium wrapper:

**Wrapper** (`ogm/ogm_gym_env.py`):
- Gymnasium-compatible interface
- Sequential agent management within single-agent paradigm
- Native action masking support
- Observation: flattened pairwise norms + one-hot agent ID

**Key Features**:
- Battle-tested SB3 implementation
- Rich logging and checkpointing
- Compatible with hyperparameter tuning tools
- Standard discrete action space (49 actions: 48 pivots + no-op)

### Reward System

The framework supports multiple composable reward mechanisms:

#### 1. Potential-Based Shaping (Primary)

```
r_potential = scale * (Φ(s') - Φ(s))
```
where `Φ(s) = -||D_curr - D_goal||_F` (negative Frobenius norm of distance difference)

**Parameters**:
- `--enable_potential_reward`: Enable/disable (default: True)
- `--potential_scale`: Scaling factor (default: 1.0)
- `--potential_normalize`: Normalization method ('n2', 'n', or 'none')

#### 2. Soft Matching Reward (Optional)

Based on the normalized soft pairwise score:
```
Φ(M) = (2 / n(n-1)) * Σ_{u<v} 1 / (1 + (D_uv - D_uv_goal)²)
```

Reward only for improvements:
```
r(t) = w(t) * max(0, Φ(M(t+1)) - Φ_max(t))
```
where `w(t) = β^t` is exponential decay

**Parameters**:
- `--enable_soft_matching_reward`: Enable soft matching
- `--soft_matching_scale`: Scale factor (default: 1.0)
- `--soft_matching_decay_beta`: Decay parameter (default: 0.999)

#### 3. Bounty Rewards (Optional)

Per-pair rewards for achieving target distances:
```
r_bounty = base_value * γ^(t-t_achieved) * η^(remaining_bounties)
```

**Parameters**:
- `--enable_bounty_reward`: Enable bounty system
- `--bounty_gamma`: Time decay (default: 0.999)
- `--bounty_eta`: Remaining bounties multiplier (default: 2.0)
- `--bounty_total_frac_of_success`: Fraction of success bonus (default: 0.2)

#### 4. Success Bonus

Large reward for achieving goal configuration:
```
r_success = success_bonus (if D_curr == D_goal)
```

**Parameter**: `--success_bonus` (default: 100.0)

#### 5. Step Cost

Encourages efficiency:
- Fixed: `--step_cost` (default: -0.005)
- Exponential decay: `--step_cost_initial`, `--step_cost_min`

### Observation Space Reductions

For scalability with large module counts:

1. **Four-Band Reduction**: Retain only 4 diagonals of distance matrix
   - Reduces from O(n²) to O(n) observations
   - Preserves sufficient geometric information
   - Enable with `--use_four_band_reduction`

2. **Local Neighborhood**: Further restrict to k-neighborhood around agent
   - Constant-size observation independent of n
   - Lossy: removes global context
   - Configure with `--use_local_neighborhood` and `--local_neighborhood_k`

## How to Run

### Training with Custom MAPPO

```bash
python train/train_mappo.py \
    --episodes 1000 \
    --num_agents 5 \
    --max_steps 500 \
    --lr 3e-4 \
    --gamma 0.99 \
    --hidden_dim 256 \
    --enable_potential_reward \
    --potential_scale 1.0 \
    --success_bonus 100.0 \
    --log_dir runs/mappo_experiment \
    --gif_interval 50
```

**Key Hyperparameters**:
- `--distance_temperature`: Controls action selection sharpness (default: 10.0)
- `--aux_coef`: Auxiliary loss weight (default: 0.1)
- `--clip`: PPO clip range (default: 0.2)
- `--epochs`: Update epochs per batch (default: 4)
- `--batch_size`: Mini-batch size (default: 64)

### Training with Stable-Baselines3

```bash
python train/train_sb3.py \
    --episodes 1000 \
    --num_agents 5 \
    --max_steps 500 \
    --n_steps 2048 \
    --lr 3e-4 \
    --gamma 0.99 \
    --hidden_dim 256 \
    --enable_potential_reward \
    --potential_scale 1.0 \
    --success_bonus 100.0 \
    --log_dir runs/sb3_experiment \
    --gif_interval 50
```

**Key Hyperparameters**:
- `--n_steps`: Steps per rollout (default: 2048)
- `--value_coef`: Value loss coefficient (default: 0.5)
- `--entropy_coef`: Entropy bonus (default: 0.01)

### Monitoring Training

Start TensorBoard:
```bash
tensorboard --logdir runs/
```

**MAPPO Metrics**:
- `reward/episode`: Total episode reward
- `loss/actor_loss`: Policy gradient loss
- `loss/critic_loss`: Value function loss
- `loss/entropy`: Policy entropy

**SB3 Metrics**:
- `rollout/ep_rew_mean`: Average episode reward
- `rollout/ep_len_mean`: Average episode length
- `train/approx_kl`: KL divergence
- `train/explained_variance`: Value function quality
- `custom/success_rate`: Success rate

### Evaluating Trained Models

For SB3 models:
```python
from sb3_contrib import MaskablePPO
from ogm.ogm_gym_env import make_ogm_env

model = MaskablePPO.load("runs/sb3/final_model")
env = make_ogm_env(num_agents=5, max_steps=500)

obs, info = env.reset()
done = False
while not done:
    action_mask = env.action_masks()
    action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Project Structure

```
MSSA/
├── agent/                      # Agent implementations
│   ├── base_agent.py          # Base agent interface
│   ├── mappo_agent.py         # Custom MAPPO agent
│   ├── simple_ppo_agent.py    # Simple PPO baseline
│   └── random_search_agent.py # Random search baseline
├── ogm/                        # Core OGM implementation
│   ├── occupancy_grid_map.py  # Grid map and physics
│   ├── ogm_env.py             # RL environment wrapper
│   ├── ogm_gym_env.py         # Gymnasium wrapper
│   └── random_configuration.py # Config generation
├── train/                      # Training scripts
│   ├── train_mappo.py         # MAPPO training
│   ├── train_sb3.py           # SB3 training
│   └── train_curriculum.py    # Curriculum learning
├── visualizer/                 # Visualization tools
│   ├── visualize_position.py  # 3D position plotting
│   └── step_visualizer.py     # Trajectory visualization
├── tests/                      # Unit and integration tests
│   ├── test_ogm_pivots.py     # Pivot operation tests
│   └── test_ogm_pairwise_norms.py
├── conda.yaml                  # Environment specification
└── README.md                   # This file
```

## Troubleshooting

**Low Success Rate**:
- Increase exploration: higher `--entropy_coef` or lower `--distance_temperature`
- Adjust reward scaling: tune `--potential_scale` and `--success_bonus`
- Increase training time: more episodes or curriculum learning

**Training Instability**:
- Reduce learning rate: `--lr`
- Increase gradient clipping: `--grad_clip`
- Decrease PPO clip range: `--clip`

**Action Masking Failures**:
- Verify connectivity constraints in `OccupancyGridMap.init_actions`
- Check pivot zone validation
- Ensure grid size is sufficient for configuration