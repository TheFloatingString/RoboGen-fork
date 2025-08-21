#!/usr/bin/env python3
"""
Example script showing how to use the enhanced wandb logging with RoboGen RL training.

This script demonstrates the new wandb integration features:
- Comprehensive training metrics logging
- Algorithm-specific metrics (PPO/SAC)
- Evaluation metrics and videos
- Best model tracking
- Proper experiment organization
"""

from RL.ray_learn import run_RL

def main():
    # Example task configuration
    task_config_path = "path/to/your/task_config.yaml"
    solution_path = "path/to/your/solution"
    task_name = "example_task"
    last_restore_state_file = "path/to/your/state.pkl"
    save_path = "./trained_models/example_experiment"
    
    # Run RL training with enhanced wandb logging
    best_policy_path, rgbs, best_traj_state_paths = run_RL(
        task_config_path=task_config_path,
        solution_path=solution_path,
        task_name=task_name,
        last_restore_state_file=last_restore_state_file,
        save_path=save_path,
        
        # Training configuration
        action_space="delta-translation",
        algo="sac",  # or "ppo"
        timesteps_total=1000000,
        seed=42,
        render=False,
        
        # Environment configuration
        randomize=True,
        use_bard=True,
        obj_id=0,
        use_gpt_size=True,
        use_gpt_joint_angle=True,
        use_gpt_spatial_relationship=True,
        use_distractor=False,
        
        # Wandb configuration
        use_wandb=True,
        wandb_project="RoboGen-Experiments",
        wandb_experiment_name="sac_example_task_v1",
    )
    
    print(f"Training completed! Best policy saved at: {best_policy_path}")


def run_experiment_sweep():
    """
    Example of running multiple experiments with different configurations
    for hyperparameter sweeping with wandb.
    """
    
    base_config = {
        "task_config_path": "path/to/your/task_config.yaml",
        "solution_path": "path/to/your/solution", 
        "task_name": "example_task",
        "last_restore_state_file": "path/to/your/state.pkl",
        "save_path": "./trained_models",
        "timesteps_total": 500000,
        "use_wandb": True,
        "wandb_project": "RoboGen-Sweep",
    }
    
    # Different configurations to test
    configs = [
        {"algo": "sac", "seed": 42, "randomize": True},
        {"algo": "sac", "seed": 123, "randomize": True},
        {"algo": "ppo", "seed": 42, "randomize": True},
        {"algo": "ppo", "seed": 123, "randomize": True},
    ]
    
    for i, config in enumerate(configs):
        experiment_config = {**base_config, **config}
        experiment_config["wandb_experiment_name"] = f"sweep_{config['algo']}_seed{config['seed']}_run{i}"
        experiment_config["save_path"] = f"./trained_models/sweep_run_{i}"
        
        print(f"Starting experiment {i+1}/{len(configs)}: {experiment_config['wandb_experiment_name']}")
        
        try:
            run_RL(**experiment_config)
            print(f"Experiment {i+1} completed successfully")
        except Exception as e:
            print(f"Experiment {i+1} failed with error: {e}")


if __name__ == "__main__":
    # Run single experiment
    main()
    
    # Uncomment to run parameter sweep
    # run_experiment_sweep()