"""
MAXIMUM PERFORMANCE + OPTUNA HPO for CarRacing-v3
Optimized for: Intel i7-14700 (20 cores), 64GB RAM, RTX 4060 8GB

Combines blazing fast training with intelligent hyperparameter optimization
"""

import gymnasium as gym
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import shutil
import json
from datetime import datetime
import argparse
import optuna.visualization as vis
from collections import deque
import time
from gymnasium.wrappers import RecordVideo


# ============================================================================
# BRAKING REWARD WRAPPERS - Encourage proper braking in tight corners
# ============================================================================

class BrakingRewardWrapper(gym.Wrapper):
    """
    Wrapper that adds reward shaping to encourage proper braking behavior
    
    The agent gets bonus rewards for:
    1. Braking when turning sharply (tight corners)
    2. Staying on track (penalty for going off-track)
    3. Maintaining good speed on straights
    """
    
    def __init__(self, env, brake_reward_scale=0.5, speed_penalty_scale=0.3):
        super().__init__(env)
        self.brake_reward_scale = brake_reward_scale
        self.speed_penalty_scale = speed_penalty_scale
        self.steps_off_track = 0
        
    def reset(self, **kwargs):
        self.steps_off_track = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract action components
        steering = action[0]  # -1 to 1
        gas = action[1]       # 0 to 1
        brake = action[2]     # 0 to 1
        
        # Check if on track (green channel in bottom of observation)
        bottom_section = obs[84:, :, 1]  # Green channel, bottom portion
        on_track = np.mean(bottom_section) > 100
        
        # Calculate turning sharpness
        turning_sharp = abs(steering) > 0.5
        
        # Reward shaping
        shaped_reward = reward
        
        # 1. Reward braking in sharp turns
        if turning_sharp and brake > 0.3:
            brake_bonus = brake * abs(steering) * self.brake_reward_scale
            shaped_reward += brake_bonus
        
        # 2. Penalty for going too fast into sharp turns without braking
        if turning_sharp and gas > 0.5 and brake < 0.1:
            shaped_reward -= 0.2 * self.speed_penalty_scale
        
        # 3. Penalty for going off track
        if not on_track:
            self.steps_off_track += 1
            if self.steps_off_track > 5:
                shaped_reward -= 0.5
        else:
            self.steps_off_track = 0
        
        # 4. Small reward for using gas on straights
        if abs(steering) < 0.3 and gas > 0.5:
            shaped_reward += 0.1
        
        return obs, shaped_reward, terminated, truncated, info


class NegativeTileWrapper(gym.Wrapper):
    """
    Adds extra penalty for going on grass (negative tiles)
    """
    
    def __init__(self, env, grass_penalty=0.5):
        super().__init__(env)
        self.grass_penalty = grass_penalty
        self.consecutive_grass = 0
    
    def reset(self, **kwargs):
        self.consecutive_grass = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if on grass
        bottom_section = obs[84:, :, 1]
        on_grass = np.mean(bottom_section) < 100
        
        if on_grass:
            self.consecutive_grass += 1
            # Increasing penalty the longer you stay on grass
            extra_penalty = min(self.grass_penalty * (self.consecutive_grass / 10), 2.0)
            reward -= extra_penalty
        else:
            self.consecutive_grass = 0
        
        return obs, reward, terminated, truncated, info


# ============================================================================

def make_env(seed=0, use_braking_reward=False, use_grass_penalty=False):
    """Create and wrap the CarRacing-v3 environment"""
    def _init():
        env = gym.make('CarRacing-v3', continuous=True, render_mode=None)
        
        # Apply reward shaping wrappers if requested
        if use_grass_penalty:
            env = NegativeTileWrapper(env, grass_penalty=0.5)
        
        if use_braking_reward:
            env = BrakingRewardWrapper(env, brake_reward_scale=0.5, speed_penalty_scale=0.3)
        
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


class TrialEvalCallback(EvalCallback):
    """
    Callback for evaluating and reporting to Optuna during training.
    Allows pruning of unpromising trials.
    """
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, **kwargs):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            
            # Report the mean reward to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def sample_ppo_params(trial, search_space="focused"):
    """
    Sample PPO hyperparameters for Optuna trial
    Optimized ranges for CarRacing-v3
    """
    if search_space == "minimal":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024]),
            "ent_coef": trial.suggest_float("ent_coef", 0.00001, 0.05, log=True),
        }
    
    elif search_space == "focused":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024]),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "n_epochs": trial.suggest_int("n_epochs", 5, 15),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999, log=True),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "ent_coef": trial.suggest_float("ent_coef", 0.00001, 0.05, log=True),
        }
    
    else:  # "full"
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [256, 512, 1024, 2048]),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "n_epochs": trial.suggest_int("n_epochs", 5, 20),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999, log=True),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "ent_coef": trial.suggest_float("ent_coef", 0.00001, 0.05, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.9),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        }


def objective(
    trial,
    n_envs=40,
    total_timesteps=800_000,  # Reduced for HPO speed
    search_space="focused",
    use_braking_reward=False,
    use_grass_penalty=False
):
    """
    Objective function for Optuna optimization
    Uses max performance settings with your hardware
    """
    
    # Sample hyperparameters
    sampled_params = sample_ppo_params(trial, search_space)
    
    # Create trial-specific directory
    trial_dir = f"./optuna_trials_maxperf/trial_{trial.number}"
    log_dir = f"{trial_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    try:
        # Enable GPU optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Create vectorized environments
        env = SubprocVecEnv([make_env(i, use_braking_reward, use_grass_penalty) for i in range(n_envs)], start_method='spawn')
        env = VecMonitor(env, log_dir)
        env = VecFrameStack(env, n_stack=4)
        
        # Create evaluation environment (use SubprocVecEnv to match training env type)
        eval_env = SubprocVecEnv([make_env(42, use_braking_reward, use_grass_penalty)], start_method='spawn')
        eval_env = VecMonitor(eval_env)
        eval_env = VecFrameStack(eval_env, n_stack=4)
        
        # Policy kwargs
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            normalize_images=True,
        )
        
        # Create model with sampled hyperparameters
        model = PPO(
            "CnnPolicy",
            env,
            verbose=0,
            tensorboard_log=None,  # Disable for speed
            device="auto",
            policy_kwargs=policy_kwargs,
            **sampled_params
        )
        
        # Create evaluation callback with pruning
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=3,  # Quick evaluation
            eval_freq=max(20_000 // n_envs, 1),
            deterministic=True,
            render=False
        )
        
        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
        
        # Clean up
        env.close()
        eval_env.close()
        
        # Return mean reward (or raise if pruned)
        if eval_callback.is_pruned:
            raise optuna.TrialPruned()
        
        return eval_callback.last_mean_reward
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise
    
    finally:
        # Clean up trial directory to save space
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir, ignore_errors=True)


def save_best_params(study, study_name):
    """
    Callback to save best parameters after each trial
    This ensures we never lose progress if the process crashes
    """
    try:
        best_trial = study.best_trial
        
        # Save best params so far
        results = {
            "best_reward": float(best_trial.value),
            "best_params": best_trial.params,
            "best_trial_number": best_trial.number,
            "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save to main results file
        results_file = f"./optuna_trials_maxperf/{study_name}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Also save a backup with trial number
        backup_file = f"./optuna_trials_maxperf/{study_name}_best_trial_{best_trial.number}.json"
        with open(backup_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved best params (Trial {best_trial.number}, Reward: {best_trial.value:.2f})")
        
    except Exception as e:
        print(f"Warning: Could not save best params: {e}")


def optimize_hyperparameters(
    n_trials=20,
    n_jobs=1,  # Parallel Optuna trials
    n_envs=40,
    total_timesteps=800_000,  # Per trial
    search_space="focused",
    study_name="carracing_maxperf_hpo",
    use_braking_reward=False,
    use_grass_penalty=False,
):
    """
    Run hyperparameter optimization with max performance settings
    
    Args:
        n_trials: Number of Optuna trials
        n_jobs: Number of parallel Optuna studies (careful with memory!)
        n_envs: Environments per trial (40 is optimal for your CPU)
        total_timesteps: Timesteps per trial (reduced for speed)
        search_space: "minimal", "focused", or "full"
        study_name: Name for the study
    """
    
    # Create directories
    os.makedirs("./optuna_trials_maxperf", exist_ok=True)
    
    print("="*70)
    print("MAXIMUM PERFORMANCE HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print(f"Hardware: Intel i7-14700 + RTX 4060 + 64GB RAM")
    print(f"Parallel environments per trial: {n_envs}")
    print(f"Timesteps per trial: {total_timesteps:,}")
    print(f"Total trials: {n_trials}")
    print(f"Search space: {search_space}")
    print(f"Parallel Optuna jobs: {n_jobs}")
    print(f"Braking reward: {'✓ ENABLED' if use_braking_reward else '✗ Disabled'}")
    print(f"Grass penalty: {'✓ ENABLED' if use_grass_penalty else '✗ Disabled'}")
    
    # Estimate time
    minutes_per_trial = (total_timesteps / (n_envs * 30)) / 60  # Rough estimate
    total_minutes = minutes_per_trial * n_trials / n_jobs
    
    print(f"\nEstimated time per trial: ~{minutes_per_trial:.0f} minutes")
    print(f"Total estimated time: ~{total_minutes:.0f} minutes (~{total_minutes/60:.1f} hours)")
    print("="*70 + "\n")
    
    # Create Optuna study with pruning
    sampler = TPESampler(n_startup_trials=5, seed=42)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, n_envs, total_timesteps, search_space, use_braking_reward, use_grass_penalty),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        callbacks=[lambda study, trial: save_best_params(study, study_name)]
    )
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\nTrials completed: {len(study.trials)}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    print("\n" + "-"*70)
    print("BEST TRIAL:")
    trial = study.best_trial
    
    print(f"  Trial number: {trial.number}")
    print(f"  Reward: {trial.value:.2f}")
    print("\n  Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save final results (this also happened after each trial via callback)
    results_file = f"./optuna_trials_maxperf/{study_name}_results.json"
    results = {
        "best_reward": float(trial.value),
        "best_params": trial.params,
        "best_trial_number": trial.number,
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Note: Best params were also saved after each trial to prevent data loss")
    
    # Generate visualizations
    try:
        
        fig = vis.plot_optimization_history(study)
        fig.write_html(f"./optuna_trials_maxperf/{study_name}_history.html")
        
        fig = vis.plot_param_importances(study)
        fig.write_html(f"./optuna_trials_maxperf/{study_name}_importance.html")
        
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(f"./optuna_trials_maxperf/{study_name}_parallel.html")
        
        print(f"Visualizations saved to: ./optuna_trials_maxperf/")
        
    except Exception as e:
        print(f"Could not generate visualizations: {e}")
    
    print("="*70 + "\n")
    
    return study


def train_with_best_params(
    study_name="carracing_maxperf_hpo",
    total_timesteps=2_500_000,
    n_envs=40,
    use_braking_reward=False,
    use_grass_penalty=False,
):
    """
    Train final model using best hyperparameters from Optuna
    """
    
    # Load best parameters
    results_file = f"./optuna_trials_maxperf/{study_name}_results.json"
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}\nRun optimization first!")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    best_params = results['best_params']
    best_reward = results['best_reward']
    
    print("="*70)
    print("FINAL TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("="*70)
    print(f"Best trial reward: {best_reward:.2f}")
    print(f"Braking reward: {'✓ ENABLED' if use_braking_reward else '✗ Disabled'}")
    print(f"Grass penalty: {'✓ ENABLED' if use_grass_penalty else '✗ Disabled'}")
    print("\nOptimized hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")
    
    # Setup directories
    log_dir = "./logs_maxperf_optimized"
    model_dir = "./models_maxperf_optimized"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Enable GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ GPU optimizations enabled\n")
    
    # Create environments
    print(f"Creating {n_envs} training environments...")
    env = SubprocVecEnv([make_env(i, use_braking_reward, use_grass_penalty) for i in range(n_envs)], start_method='spawn')
    env = VecMonitor(env, log_dir)
    env = VecFrameStack(env, n_stack=4)
    
    eval_env = SubprocVecEnv([make_env(42, use_braking_reward, use_grass_penalty)], start_method='spawn')
    eval_env = VecMonitor(eval_env, f"{log_dir}/eval")
    eval_env = VecFrameStack(eval_env, n_stack=4)
    print("✓ Environments created\n")
    
    # Create callbacks
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=model_dir,
        name_prefix="ppo_optimized",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=f"{log_dir}/eval",
        eval_freq=25_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    # Policy kwargs
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        normalize_images=True,
    )
    
    # Create model with best params
    print("Initializing PPO model with optimized hyperparameters...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
        policy_kwargs=policy_kwargs,
        **best_params
    )
    print("✓ Model initialized\n")
    
    # Train
    print(f"Starting final training for {total_timesteps:,} timesteps...")
    print(f"Monitor: tensorboard --logdir {log_dir}\n")
    print("="*70 + "\n")
    
    try:
        import time
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        elapsed = time.time() - start_time
        
        # Save final model
        final_path = f"{model_dir}/ppo_optimized_final"
        model.save(final_path)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Training time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"Throughput: {total_timesteps/elapsed:.0f} timesteps/second")
        print(f"\nModels saved:")
        print(f"  Final: {final_path}.zip")
        print(f"  Best: {model_dir}/best_model/best_model.zip")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        model.save(f"{model_dir}/ppo_optimized_interrupted")
    
    finally:
        env.close()
        eval_env.close()
    
    return model

def test_model(
    model_path,
    n_episodes=10,
    render=True,
    record_video=False,
    video_folder="./videos",
    deterministic=True,
):
    """
    Test a trained model and visualize results
    
    Args:
        model_path: Path to the trained model (.zip file)
        n_episodes: Number of test episodes
        render: Whether to render the environment (human view)
        record_video: Whether to record video of the episodes
        video_folder: Folder to save videos
        deterministic: Use deterministic policy (recommended for testing)
    """
        
    print("="*70)
    print("TESTING TRAINED MODEL")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"Render: {render}")
    print(f"Record video: {record_video}")
    print("="*70 + "\n")
    
    # Load the model
    print("Loading model...")
    model = PPO.load(model_path)
    print("✓ Model loaded\n")
    
    # Create environment
    render_mode = "human" if render else "rgb_array" if record_video else None
    
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        env = gym.make('CarRacing-v3', continuous=True, render_mode="rgb_array")
        env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)
    else:
        env = gym.make('CarRacing-v3', continuous=True, render_mode=render_mode)
    
    # Frame stacking wrapper to match training
    class FrameStackWrapper(gym.Wrapper):
        def __init__(self, env, num_stack=4):
            super().__init__(env)
            self.num_stack = num_stack
            self.frames = deque(maxlen=num_stack)
            
            original_shape = self.observation_space.shape
            new_shape = (original_shape[0], original_shape[1], original_shape[2] * num_stack)
            
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=new_shape,
                dtype=np.uint8
            )
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            for _ in range(self.num_stack):
                self.frames.append(obs)
            return self._get_obs(), info
        
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.frames.append(obs)
            return self._get_obs(), reward, terminated, truncated, info
        
        def _get_obs(self):
            assert len(self.frames) == self.num_stack
            return np.concatenate(list(self.frames), axis=-1)
    
    env = FrameStackWrapper(env, num_stack=4)
    
    # Test the model
    rewards = []
    episode_lengths = []
    
    print("Starting evaluation...\n")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        start_time = time.time()
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render and episode_length % 10 == 0:
                # Optional: display current stats on screen
                pass
        
        elapsed = time.time() - start_time
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Determine success (reward >= 900 is considered solved)
        status = "✓ SOLVED" if episode_reward >= 900 else "✗ Not solved"
        
        print(f"Episode {episode + 1:2d}/{n_episodes}: "
              f"Reward = {episode_reward:7.2f} | "
              f"Length = {episode_length:4d} steps | "
              f"Time = {elapsed:5.1f}s | "
              f"{status}")
    
    env.close()
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    mean_length = np.mean(episode_lengths)
    
    solved_count = sum(1 for r in rewards if r >= 900)
    success_rate = (solved_count / n_episodes) * 100
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Episodes evaluated: {n_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean reward:   {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min reward:    {min_reward:.2f}")
    print(f"  Max reward:    {max_reward:.2f}")
    print(f"\nPerformance:")
    print(f"  Mean episode length: {mean_length:.1f} steps")
    print(f"  Solved episodes (≥900): {solved_count}/{n_episodes} ({success_rate:.1f}%)")
    
    if mean_reward >= 900:
        print(f"\n🏆 ENVIRONMENT SOLVED! Average reward: {mean_reward:.2f}")
    elif mean_reward >= 850:
        print(f"\n🎯 Near solution! Average reward: {mean_reward:.2f}")
    elif mean_reward >= 700:
        print(f"\n📈 Good performance! Average reward: {mean_reward:.2f}")
    elif mean_reward >= 500:
        print(f"\n🔄 Moderate performance. Average reward: {mean_reward:.2f}")
    else:
        print(f"\n⚠️  Needs more training. Average reward: {mean_reward:.2f}")
    
    if record_video:
        print(f"\n📹 Videos saved to: {video_folder}")
    
    print("="*70 + "\n")
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "rewards": rewards,
    }

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Max Performance + Optuna HPO for CarRacing-v3")
    parser.add_argument("--mode", type=str, default="optimize",
                        choices=["optimize", "train_best", "quick_optimize", "test"],
                        help="Mode: optimize hyperparameters, train with best, or test model")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Parallel Optuna jobs (use 1 unless you have >32GB RAM)")
    parser.add_argument("--n_envs", type=int, default=40,
                        help="Parallel environments per trial")
    parser.add_argument("--timesteps_per_trial", type=int, default=800_000,
                        help="Timesteps per trial")
    parser.add_argument("--search_space", type=str, default="focused",
                        choices=["minimal", "focused", "full"],
                        help="Search space size")
    parser.add_argument("--final_timesteps", type=int, default=2_500_000,
                        help="Timesteps for final training with best params")
    
    # Test mode arguments
    parser.add_argument("--model_path", type=str, default="./models_maxperf_optimized/best_model/best_model.zip",
                        help="Path to model for testing")
    parser.add_argument("--n_test_episodes", type=int, default=10,
                        help="Number of test episodes")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable rendering during testing")
    parser.add_argument("--record_video", action="store_true",
                        help="Record video of test episodes")
    parser.add_argument("--video_folder", type=str, default="./videos",
                        help="Folder to save videos")
    
    # Braking reward options
    parser.add_argument("--braking_reward", action="store_true",
                        help="Enable braking reward shaping (helps with tight corners)")
    parser.add_argument("--grass_penalty", action="store_true",
                        help="Enable extra grass penalty")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        # Test a trained model
        test_model(
            model_path=args.model_path,
            n_episodes=args.n_test_episodes,
            render=not args.no_render,
            record_video=args.record_video,
            video_folder=args.video_folder,
        )
        
    elif args.mode == "quick_optimize":
        # Quick optimization: fewer trials, shorter training
        print("Quick optimization mode: 10 trials, 500K timesteps each\n")
        study = optimize_hyperparameters(
            n_trials=10,
            n_jobs=args.n_jobs,
            n_envs=args.n_envs,
            total_timesteps=500_000,
            search_space="minimal",
            use_braking_reward=args.braking_reward,
            use_grass_penalty=args.grass_penalty,
        )
        print("\n✓ Quick optimization complete!")
        print("Run with --mode train_best to train final model\n")
        
    elif args.mode == "optimize":
        study = optimize_hyperparameters(
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            n_envs=args.n_envs,
            total_timesteps=args.timesteps_per_trial,
            search_space=args.search_space,
            use_braking_reward=args.braking_reward,
            use_grass_penalty=args.grass_penalty,
        )
        print("\n✓ Optimization complete!")
        print("Run with --mode train_best to train final model\n")
        
    else:  # train_best
        train_with_best_params(
            total_timesteps=args.final_timesteps,
            n_envs=args.n_envs,
            use_braking_reward=args.braking_reward,
            use_grass_penalty=args.grass_penalty,
        )