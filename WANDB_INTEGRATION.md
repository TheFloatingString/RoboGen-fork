# Weights & Biases (wandb) Integration with RoboGen

This repository now includes comprehensive integration with Weights & Biases (wandb) for experiment tracking, visualization, and model management in reinforcement learning training.

## Features

### 🔧 Enhanced Training Logging
- **Real-time metrics**: Training rewards, episode lengths, FPS, and training time
- **Algorithm-specific metrics**:
  - **PPO**: Policy loss, value function loss, entropy, KL divergence
  - **SAC**: Actor loss, critic loss, alpha parameter
- **Evaluation metrics**: Return, episode length, best return tracking
- **Video logging**: Automatic logging of evaluation episodes and best trajectories as videos

### 📊 Comprehensive Experiment Tracking
- **Configuration logging**: All hyperparameters and environment settings
- **Model checkpointing**: Automatic tracking of best models
- **Progress visualization**: Real-time training curves and performance metrics
- **Experiment organization**: Proper project and experiment naming

### 🚀 Easy Integration
- **Backward compatible**: Existing code continues to work
- **Optional**: Can be disabled with `use_wandb=False`
- **Flexible configuration**: Customizable project names and experiment settings

## Setup

### Prerequisites
```bash
pip install wandb
```

### Initialize wandb (first time only)
```bash
wandb login
```

## Usage

### Basic Usage

```python
from RL.ray_learn import run_RL

# Run training with wandb logging
best_policy_path, rgbs, best_traj_state_paths = run_RL(
    task_config_path="path/to/task_config.yaml",
    solution_path="path/to/solution",
    task_name="my_task",
    last_restore_state_file="path/to/state.pkl",
    save_path="./models",
    
    # Wandb configuration
    use_wandb=True,                           # Enable wandb logging
    wandb_project="My-RoboGen-Project",      # Project name
    wandb_experiment_name="experiment_v1",    # Experiment name (optional)
    
    # Standard training parameters
    algo="sac",
    timesteps_total=1000000,
    seed=42,
)
```

### Advanced Configuration

```python
# Run multiple experiments for hyperparameter sweeping
configs = [
    {"algo": "sac", "seed": 42},
    {"algo": "sac", "seed": 123}, 
    {"algo": "ppo", "seed": 42},
]

for i, config in enumerate(configs):
    run_RL(
        # ... task configuration ...
        
        # Unique experiment name for each run
        wandb_experiment_name=f"sweep_{config['algo']}_seed{config['seed']}",
        wandb_project="RoboGen-Hyperparameter-Sweep",
        
        **config
    )
```

### Disable wandb

```python
# Disable wandb logging (default behavior preserved)
run_RL(
    # ... configuration ...
    use_wandb=False,  # No wandb logging
)
```

## Logged Metrics

### Training Metrics
- `training/iteration`: Training iteration number
- `training/timesteps_total`: Total timesteps completed
- `training/time_total_s`: Total training time in seconds
- `training/fps`: Frames per second
- `training/episode_reward_mean`: Mean episode reward
- `training/episode_reward_min/max`: Min/max episode rewards
- `training/episode_len_mean`: Mean episode length

### Algorithm-Specific Metrics

#### PPO
- `training/policy_loss`: Policy loss
- `training/value_function_loss`: Value function loss  
- `training/entropy`: Policy entropy
- `training/kl`: KL divergence

#### SAC
- `training/actor_loss`: Actor network loss
- `training/critic_loss`: Critic network loss
- `training/alpha`: Temperature parameter

### Evaluation Metrics
- `evaluation/return`: Episode return during evaluation
- `evaluation/episode_length`: Length of evaluation episode
- `evaluation/best_return`: Best return achieved so far
- `evaluation/episode_video`: Video of evaluation episode
- `evaluation/best_episode_video`: Video of best episode
- `evaluation/new_best_return`: New best return (when achieved)

### Final Metrics
- `final/best_policy_path`: Path to best saved model
- `final/training_completed`: Training completion flag
- `final/best_trajectory_video`: Final best trajectory video

## Wandb Dashboard Features

### 📈 Real-time Monitoring
- View training progress in real-time
- Compare different experiments side-by-side
- Track hyperparameter effects on performance

### 🎥 Video Visualization
- Watch agent behavior during evaluation
- Compare performance across different training stages
- Visual debugging of policy behavior

### 📊 Experiment Comparison
- Compare multiple runs with different algorithms
- Analyze the effect of hyperparameters
- Statistical comparison of performance

### 💾 Model Management
- Track best model checkpoints
- Download models directly from wandb
- Version control for trained policies

## Example Files

- `example_wandb_training.py`: Complete example showing basic usage and hyperparameter sweeping
- `RL/ray_learn.py`: Enhanced training functions with wandb integration

## Best Practices

### 🏷️ Naming Conventions
- Use descriptive project names: `"RoboGen-TaskName"`
- Include key parameters in experiment names: `"sac_randomized_seed42"`
- Use consistent naming across related experiments

### 🔧 Configuration Management
```python
# Good: Include key parameters in experiment name
experiment_name = f"{algo}_{task_name}_seed{seed}_{'rand' if randomize else 'norend'}"

# Log all relevant configuration
wandb_config = {
    "project": "RoboGen-Manipulation",
    "experiment_name": experiment_name,
    "extra_config": {
        "task_description": "Complex manipulation task",
        "environment_version": "v2.0",
        # ... other metadata
    }
}
```

### 📊 Experiment Organization
- Group related experiments in the same project
- Use tags to categorize experiments
- Include environment and task details in config

### 🚀 Performance Tips
- Videos are only logged during evaluation intervals
- Large video files are automatically handled by wandb
- Use appropriate evaluation intervals to balance logging frequency and performance

## Troubleshooting

### Common Issues

**1. wandb login required**
```bash
wandb login
```

**2. Video logging errors**
- Ensure `moviepy` is installed: `pip install moviepy`
- Check that evaluation generates frames

**3. Experiment not appearing in wandb**
- Verify internet connection
- Check wandb project permissions
- Ensure `use_wandb=True`

**4. Memory issues with video logging**
- Reduce evaluation frequency
- Use smaller video resolution if needed

### Debug Mode
```python
# Enable wandb debug logging
import wandb
wandb.init(mode="dryrun")  # For testing without uploading
```

## Migration Guide

### From Old Version
The integration is backward compatible. Simply add wandb parameters to existing `run_RL` calls:

```python
# Old code (still works)
run_RL(task_config, solution_path, task_name, state_file, save_path)

# Enhanced with wandb
run_RL(
    task_config, solution_path, task_name, state_file, save_path,
    use_wandb=True, 
    wandb_project="My-Project"
)
```

### Configuration Updates
- Remove old manual `wandb.init()` calls
- Replace manual logging with automatic integration  
- Update experiment scripts to use new parameters

---

For more information about Weights & Biases, visit [https://docs.wandb.ai/](https://docs.wandb.ai/)