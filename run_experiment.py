"""
Main Experiment Runner

# RECOMMENDED: Run with subprocess isolation (prevents freezing/state leakage)
python run_experiment.py --isolated --env-only starpilot

# Run only DQN on CartPole (isolated)
python run_experiment.py --isolated --algo-only dqn --env-only starpilot
# Run everything for CartPole (all algos, no isolation)
python run_experiment.py --env-only cartpole

# Run only PPO on CartPole
python run_experiment.py --algo-only ppo --env-only cartpole

# Run only ES on CartPole
python run_experiment.py --algo-only es --env-only cartpole

# Run single experiment
python run_experiment.py --algo dqn --technique baseline --env cartpole

# Analyze results and generate plots
python run_experiment.py --analyze

Techniques:
  baseline - No regularization, no GAR
  reg      - L2 weight decay (1e-4)
  gar      - Gradient Agreement Regularization
  gar+reg  - Both GAR and weight decay

Flags:
  --isolated  Run each algorithm in a separate subprocess (prevents freezing)
"""

import os
import sys
import json
import argparse
import time
import warnings
import gc
import random
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.dqn import DQN
from agents.ppo import PPO
from agents.es import ES
from envs.cartpole_var import CartPoleVarEnv
from envs.minigrid_var import MiniGridVarEnv
from envs.minatar_var import MinAtarVarEnv
from evaluation.grs import compute_grs, plot_grs_curve, create_profile_card
from evaluation.metrics import MetricsTracker, plot_learning_curves, plot_grs_heatmap, create_experiment_summary
from envs.wrappers import FrameStack


ALGORITHMS = ["dqn", "ppo", "es"]
# Techniques can be combined with "+" (e.g., "gar+reg" for both GAR and regularization)
TECHNIQUES = ["baseline", "reg", "gar", "gar+reg"]
ENVIRONMENTS = ["cartpole", "minigrid", "space_invaders"]


def parse_technique(technique: str) -> Dict[str, Any]:
    """
    Parse technique string into component settings.

    Examples:
        "baseline" -> {"weight_decay": 0.0, "use_gar": False}
        "reg"      -> {"weight_decay": 1e-4, "use_gar": False}
        "gar"      -> {"weight_decay": 0.0, "use_gar": True}
        "gar+reg"  -> {"weight_decay": 1e-4, "use_gar": True}
    """
    components = technique.lower().split("+")

    settings = {
        "weight_decay": 0.0,
        "use_gar": False,
    }

    for comp in components:
        comp = comp.strip()
        if comp == "reg":
            settings["weight_decay"] = 1e-4
        elif comp == "gar":
            settings["use_gar"] = True
        elif comp == "baseline":
            pass  # No changes
        else:
            raise ValueError(f"Unknown technique component: {comp}")

    return settings

TRAINING_CONFIG = {
    "cartpole": {
        "total_steps": 50000,
        "buffer_size": 100000,
        "eval_episodes": 20,
        "max_steps": 500,
    },
    "minigrid": {
        "total_steps": 1000000,
        "buffer_size": 100000,
        "eval_episodes": 20,
        "max_steps": 100,
    },
    "starpilot": {
        "total_steps": 3000000,
        "buffer_size": 22000,  # Max for 10GB: (10GB - 1.5GB overhead) / 384KB per transition
        "eval_episodes": 20,
        "max_steps": 1000,
    },
    "space_invaders": {
        "total_steps": 500000,  # Match MiniGrid for consistent comparison
        "buffer_size": 100000,  # Small observations (10x10x6) = ~2.4KB per transition
        "eval_episodes": 20,
        "max_steps": 5000,
    },
}

SHIFT_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Environment-specific hyperparameters
# Different environments have vastly different observation spaces and dynamics,
# requiring tailored hyperparameters for optimal performance.
HYPERPARAMS = {
    "cartpole": {
        # CartPole: 4-dim state vector, simple dynamics, fast learning
        "dqn": {
            "learning_rate": 1e-4,
            "hidden_dims": [64, 64],  # Small MLP sufficient
            "batch_size": 64,
            "target_update_freq": 2000,
        },
        "ppo": {
            "learning_rate": 3e-4,
            "hidden_dims": [64, 64],
            "n_steps": 2048,
            "n_epochs": 10,
            "batch_size": 64,
            "ent_coef": 0.01,
        },
        "es": {
            "population_size": 50,
            "sigma": 0.1,
            "learning_rate": 0.03,
            "hidden_dims": [64, 64],
        },
    },
    "minigrid": {
        # MiniGrid: 7x7 partial observation (49 dims flattened)
        # Uses MLP with normalized discrete observations (cell types 0-4)
        # Sparse rewards require slower epsilon decay and more exploration
        "dqn": {
            "learning_rate": 1e-4,  # Higher LR works with normalized obs
            "hidden_dims": [128, 128],  # Medium network
            "batch_size": 64,
            "target_update_freq": 1000,  # More frequent target updates for grid world
            "epsilon_decay": 500000,  # Slow decay for sparse rewards (50% of training)
            "epsilon_end": 0.1,  # Higher minimum for continued exploration
        },
        "ppo": {
            "learning_rate": 2e-4,
            "hidden_dims": [128, 128],
            "n_steps": 2048,
            "n_epochs": 10,
            "batch_size": 64,
            "ent_coef": 0.05,  # Higher entropy for exploration in sparse reward
        },
        "es": {
            "population_size": 40,
            "sigma": 0.1,
            "learning_rate": 0.02,
            "hidden_dims": [128, 128],
        },
    },
    "starpilot": {
        # Starpilot (Procgen): 64x64 RGB images, complex visual environment
        # Needs larger CNN, lower learning rates for stability, higher entropy
        "dqn": {
            "learning_rate": 2.5e-5,  # Lower LR for image-based learning
            "hidden_dims": [256, 256],  # Larger network for visual processing
            "batch_size": 32,  # Smaller batch for memory efficiency
            "target_update_freq": 5000,  # Less frequent updates for stability
        },
        "ppo": {
            "learning_rate": 5e-5,  # Lower for image stability
            "hidden_dims": [256, 256],
            "n_steps": 2048,
            "n_epochs": 3,  # Fewer epochs for Procgen (per OpenAI recommendations)
            "batch_size": 64,
            "ent_coef": 0.01,  # Higher entropy for exploration
        },
        "es": {
            # WARNING: ES on image-based envs is computationally expensive.
            # Reduced population size to make it tractable.
            "population_size": 20,  # Reduced from 50 for speed
            "sigma": 0.02,  # Lower sigma for high-dimensional parameter space
            "learning_rate": 0.01,  # Conservative LR
            "hidden_dims": [256, 256],
        },
    },
    "space_invaders": {
        # MinAtar Space Invaders: 10x10x6 small images, fast to train
        # Uses CNN but much smaller than Procgen
        "dqn": {
            "learning_rate": 1e-4,
            "hidden_dims": [128, 128],
            "batch_size": 64,
            "target_update_freq": 2000,
        },
        "ppo": {
            "learning_rate": 3e-4,
            "hidden_dims": [128, 128],
            "n_steps": 1024,
            "n_epochs": 4,
            "batch_size": 64,
            "ent_coef": 0.01,
        },
        "es": {
            "population_size": 50,
            "sigma": 0.1,
            "learning_rate": 0.03,
            "hidden_dims": [128, 128],
        },
    },
}


def make_env(env_name: str, variation_level: float = 0.0, use_frame_stack: bool = True):
    """
    Create environment with appropriate wrappers.

    Args:
        env_name: Name of the environment (cartpole, minigrid, starpilot)
        variation_level: Distribution shift level for generalization testing
        use_frame_stack: Whether to apply frame stacking (for Starpilot/Procgen)

    Returns:
        Wrapped gymnasium environment
    """
    if env_name == "cartpole":
        return CartPoleVarEnv(variation_level=variation_level)
    elif env_name == "minigrid":
        # Use partial observability (7x7 agent view = 49 dims) for better learning
        # Full observability with padding is 89% walls and hard to learn from
        return MiniGridVarEnv(variation_level=variation_level, fully_observable=False)
    elif env_name == "starpilot":
        from envs.starpilot_var import StarpilotVarEnv
        env = StarpilotVarEnv(variation_level=variation_level)
        # Apply frame stacking for temporal information (standard for Procgen)
        # Most Procgen benchmarks use 4 stacked frames
        if use_frame_stack:
            env = FrameStack(env, n_frames=4)
        return env
    elif env_name == "space_invaders":
        return MinAtarVarEnv(variation_level=variation_level, game="space_invaders")
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def get_env_class(env_name: str):
    if env_name == "cartpole":
        return CartPoleVarEnv
    elif env_name == "minigrid":
        return MiniGridVarEnv
    elif env_name == "starpilot":
        from envs.starpilot_var import StarpilotVarEnv
        return StarpilotVarEnv
    elif env_name == "space_invaders":
        return MinAtarVarEnv
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def train_agent(
    algo: str,
    technique: str,
    env_name: str,
    seed: int,
    config: Dict,
    verbose: bool = True,
) -> Dict[str, Any]:
    # Aggressive cleanup before starting new experiment
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        # For full determinism (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Reset any lingering torch state
    torch.set_grad_enabled(True)

    env = make_env(env_name, variation_level=0.0)
    # Seed environment for reproducibility
    env.reset(seed=seed)

    # Get environment-specific hyperparameters
    hp = HYPERPARAMS.get(env_name, HYPERPARAMS["cartpole"]).get(algo, {})

    # Parse technique into settings (supports combinations like "gar+reg")
    tech_settings = parse_technique(technique)
    weight_decay = tech_settings["weight_decay"]
    use_gar = tech_settings["use_gar"]

    if verbose:
        print(f"  Technique settings: weight_decay={weight_decay}, use_gar={use_gar}")

    # Get environment class and kwargs for GAR
    env_class = get_env_class(env_name) if use_gar else None
    env_kwargs = {"fully_observable": False} if env_name == "minigrid" else {}
    gar_variation_levels = [0.0, 0.25, 0.5]  # Levels to sample from during training

    # Warn about ES on image-based environments
    if algo == "es" and env_name == "starpilot":
        warnings.warn(
            "ES on image-based environments (Starpilot) is computationally expensive. "
            f"Using reduced population_size={hp.get('population_size', 20)} for tractability. "
            "Consider using DQN or PPO for faster training.",
            RuntimeWarning
        )

    # GAR on image-based envs is very expensive - disable for PPO (runs every step)
    if use_gar and env_name == "starpilot" and algo == "ppo":
        warnings.warn(
            "GAR with PPO on Starpilot is extremely slow (runs 3 Procgen envs every step). "
            "Disabling GAR for this experiment. Use DQN+GAR instead (optimized for images).",
            RuntimeWarning
        )
        use_gar = False
        env_class = None

    # Create env_wrapper for GAR environments (e.g., FrameStack for Starpilot)
    env_wrapper = None
    if env_name == "starpilot":
        env_wrapper = lambda e: FrameStack(e, n_frames=4)

    # Create agent with environment-specific hyperparameters
    if algo == "dqn":
        agent = DQN(
            env=env,
            learning_rate=hp.get("learning_rate", 1e-4),
            buffer_size=config["buffer_size"],
            batch_size=hp.get("batch_size", 64),
            learning_starts=1000,
            target_update_freq=hp.get("target_update_freq", 2000),
            epsilon_decay=hp.get("epsilon_decay", int(config["total_steps"] * 0.7)),
            epsilon_end=hp.get("epsilon_end", 0.05),
            weight_decay=weight_decay,
            hidden_dims=hp.get("hidden_dims", [64, 64]),
            # GAR parameters
            use_gar=use_gar,
            gar_variation_levels=gar_variation_levels,
            env_class=env_class,
            env_kwargs=env_kwargs,
            env_wrapper=env_wrapper,
        )
    elif algo == "ppo":
        agent = PPO(
            env=env,
            learning_rate=hp.get("learning_rate", 3e-4),
            n_steps=hp.get("n_steps", 2048),
            n_epochs=hp.get("n_epochs", 10),
            batch_size=hp.get("batch_size", 64),
            weight_decay=weight_decay,
            hidden_dims=hp.get("hidden_dims", [64, 64]),
            ent_coef=hp.get("ent_coef", 0.01),
            # GAR parameters
            use_gar=use_gar,
            gar_variation_levels=gar_variation_levels,
            env_class=env_class,
            env_kwargs=env_kwargs,
            env_wrapper=env_wrapper,
        )
    elif algo == "es":
        # Note: ES does not support GAR (evolutionary strategies don't use gradients)
        if use_gar:
            warnings.warn(
                "GAR is not supported for ES (Evolutionary Strategies) as it doesn't use gradients. "
                "Running ES without GAR.",
                RuntimeWarning
            )
        agent = ES(
            env=env,
            population_size=hp.get("population_size", 50),
            sigma=hp.get("sigma", 0.1),
            learning_rate=hp.get("learning_rate", 0.03),
            weight_decay=weight_decay,
            hidden_dims=hp.get("hidden_dims", [64, 64]),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    if verbose:
        print(f"Training {algo.upper()} with {technique} on {env_name}...")
    
    start_time = time.time()
    
    if algo == "es":
        num_generations = config["total_steps"] // 1000
        # MiniGrid has sparse rewards - need more episodes for stable fitness signal
        es_episodes = 5 if env_name == "minigrid" else 1
        history = agent.learn(
            total_generations=num_generations,
            episodes_per_eval=es_episodes,
            max_steps=config["max_steps"],
            log_interval=num_generations // 10,
            eval_interval=num_generations // 5,
        )
    elif algo == "ppo":
        # PPO logs per UPDATE, not per step
        # num_updates = total_steps // n_steps, so calculate log_interval accordingly
        n_steps = hp.get("n_steps", 2048)
        num_updates = config["total_steps"] // n_steps
        history = agent.learn(
            total_timesteps=config["total_steps"],
            log_interval=max(1, num_updates // 10),  # Log ~10 times during training
            eval_interval=max(1, num_updates // 5),  # Eval ~5 times during training
            eval_episodes=5,
        )
    else:
        # DQN logs per step
        history = agent.learn(
            total_timesteps=config["total_steps"],
            log_interval=config["total_steps"] // 10,
            eval_interval=config["total_steps"] // 5,
            eval_episodes=5,
        )
    
    train_time = time.time() - start_time
    
    if verbose:
        print(f"Training completed in {train_time:.1f}s")
    
    env.close()
    
    return {
        "agent": agent,
        "history": history,
        "train_time": train_time,
    }


def run_experiment(
    algo: str,
    technique: str,
    env_name: str,
    seed: int,
    results_dir: str = "results",
    metrics_tracker: Optional[MetricsTracker] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    experiment_id = f"{algo}_{technique}_{env_name}_seed{seed}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_id}")
        print(f"{'='*60}")

    config = TRAINING_CONFIG[env_name]

    # Initialize metrics tracker if not provided
    if metrics_tracker is None:
        metrics_tracker = MetricsTracker(results_dir)

    train_result = train_agent(
        algo=algo,
        technique=technique,
        env_name=env_name,
        seed=seed,
        config=config,
        verbose=verbose,
    )

    agent = train_result["agent"]
    history = train_result["history"]

    # Save training history and generate learning curve plot
    metrics_tracker.save_training_history(history, algo, technique, env_name, seed)

    plot_path = os.path.join(results_dir, "plots", f"{experiment_id}_learning_curves.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plot_learning_curves(history, algo, technique, env_name, save_path=plot_path, show=False)

    # Evaluate and check if this is the best model for this architecture
    eval_reward = agent.evaluate(num_episodes=config["eval_episodes"])
    metrics_tracker.update_best_model(agent, algo, env_name, eval_reward, technique, seed)

    if verbose:
        print("Computing GRS...")

    env_class = get_env_class(env_name)
    env_kwargs = {"fully_observable": False} if env_name == "minigrid" else {}

    # For Starpilot, we need to apply the same FrameStack wrapper used during training
    env_wrapper = None
    if env_name == "starpilot":
        env_wrapper = lambda env: FrameStack(env, n_frames=4)

    grs_result = compute_grs(
        agent=agent,
        env_class=env_class,
        shift_levels=SHIFT_LEVELS,
        num_episodes=config["eval_episodes"],
        max_steps=config["max_steps"],
        env_kwargs=env_kwargs,
        env_wrapper=env_wrapper,
    )

    # Save GRS curve plot
    grs_plot_path = os.path.join(results_dir, "plots", f"{experiment_id}_grs_curve.png")
    plot_grs_curve(grs_result, title=f"GRS: {algo.upper()} + {technique} on {env_name}",
                   save_path=grs_plot_path, show=False)

    results = {
        "experiment_id": experiment_id,
        "algorithm": algo,
        "technique": technique,
        "environment": env_name,
        "seed": seed,
        "train_time": train_result["train_time"],
        "eval_reward": eval_reward,
        "grs": grs_result["grs"],
        "grs_curve": {
            "shift_levels": grs_result["shift_levels"],
            "performances": grs_result["performances"],
            "normalized": grs_result["normalized"],
        },
        "train_performance": grs_result["baseline"],
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{experiment_id}.json")

    results_json = json.loads(
        json.dumps(results, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    )

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    if verbose:
        print(f"GRS: {results['grs']:.3f}")
        print(f"Eval Reward: {eval_reward:.2f}")
        print(f"Saved to: {results_path}")

    return results


def run_all_experiments_isolated(
    seeds: List[int] = [0, 1, 2],
    results_dir: str = "results",
    envs: List[str] = None,
    algos: List[str] = None,
):
    """
    Run experiments with subprocess isolation per algorithm.
    Each algorithm runs in a fresh Python process to prevent state leakage.
    """
    envs = envs or ENVIRONMENTS
    algos = algos or ALGORITHMS

    total_algos = len(algos) * len(envs)
    print("\n" + "="*60)
    print("RUNNING WITH SUBPROCESS ISOLATION")
    print(f"Each algorithm runs in a fresh Python process")
    print(f"Algorithms: {algos}")
    print(f"Environments: {envs}")
    print(f"Total subprocess runs: {total_algos}")
    print("="*60)

    script_path = os.path.abspath(__file__)
    completed = 0

    for env_name in envs:
        for algo in algos:
            completed += 1
            print(f"\n{'='*60}")
            print(f"[{completed}/{total_algos}] Starting {algo.upper()} on {env_name} (fresh subprocess)")
            print(f"{'='*60}\n")

            cmd = [
                sys.executable, script_path,
                "--algo-only", algo,
                "--env-only", env_name,
                "--seeds", *map(str, seeds),
                "--results-dir", results_dir,
            ]

            # Run in subprocess - completely fresh Python state
            result = subprocess.run(cmd, cwd=os.path.dirname(script_path))

            if result.returncode != 0:
                print(f"WARNING: {algo} on {env_name} exited with code {result.returncode}")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE (with isolation)")
    print("="*60)


def run_all_experiments(
    seeds: List[int] = [0, 1, 2],
    results_dir: str = "results",
    envs: List[str] = None,
    algos: List[str] = None,
    techniques: List[str] = None,
):
    os.makedirs(results_dir, exist_ok=True)

    envs = envs or ENVIRONMENTS
    algos = algos or ALGORITHMS
    techniques = techniques or TECHNIQUES

    total = len(algos) * len(techniques) * len(envs) * len(seeds)

    print(f"\nRunning {total} experiments...")
    print(f"Algorithms: {algos}")
    print(f"Techniques: {techniques}")
    print(f"Environments: {envs}")
    print(f"Seeds: {seeds}\n")

    # Shared metrics tracker for best model tracking
    metrics_tracker = MetricsTracker(results_dir)
    all_results = []

    # Build experiment list
    experiments = [
        (env_name, algo, technique, seed)
        for env_name in envs
        for algo in algos
        for technique in techniques
        for seed in seeds
    ]

    # Main progress bar
    pbar = tqdm(experiments, desc="Experiments", unit="exp",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for env_name, algo, technique, seed in pbar:
        exp_name = f"{algo}_{technique}_{env_name}_s{seed}"
        pbar.set_description(f"{exp_name}")

        try:
            result = run_experiment(
                algo=algo,
                technique=technique,
                env_name=env_name,
                seed=seed,
                results_dir=results_dir,
                metrics_tracker=metrics_tracker,
            )
            all_results.append(result)
        except Exception as e:
            tqdm.write(f"ERROR in {exp_name}: {e}")
            import traceback
            traceback.print_exc()

        # Clear memory after each algorithm completes all techniques for an env
        if technique == techniques[-1] and seed == seeds[-1]:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*60)

    # Save summary
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    # Generate final visualizations
    print("\nGenerating final visualizations...")

    # GRS heatmap
    if all_results:
        heatmap_path = os.path.join(results_dir, "plots", "grs_heatmap.png")
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        plot_grs_heatmap(all_results, save_path=heatmap_path, show=False)
        print(f"  GRS heatmap: {heatmap_path}")

        # Text summary
        summary_txt_path = os.path.join(results_dir, "experiment_summary.txt")
        create_experiment_summary(all_results, save_path=summary_txt_path)

    # Print best models
    print("\n" + "="*60)
    print("BEST MODELS PER ARCHITECTURE")
    print("="*60)
    for key, info in metrics_tracker.best_scores.items():
        print(f"  {key}: score={info['score']:.2f}, technique={info['technique']}, seed={info['seed']}")

    print(f"\nAll experiments completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"Best models saved to: {metrics_tracker.models_dir}")

    return all_results


def analyze_results(results_dir: str = "results"):
    import glob

    results = []
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        if "summary" in path or "history" in path:
            continue
        with open(path, 'r') as f:
            try:
                results.append(json.load(f))
            except json.JSONDecodeError:
                continue

    if not results:
        print("No results found!")
        return

    aggregated = {}
    for r in results:
        key = (r["algorithm"], r["technique"], r["environment"])
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(r["grs"])

    all_algos = sorted(set(r["algorithm"] for r in results))
    all_techniques = sorted(set(r["technique"] for r in results))
    all_envs = sorted(set(r["environment"] for r in results))

    print("\n" + "=" * 80)
    print("PATTERN TABLE: Mean GRS (± std)")
    print("=" * 80)

    header = f"{'Algorithm + Technique':<25}"
    for env in all_envs:
        header += f" {env:>15}"
    print(header)
    print("-" * 80)

    for algo in all_algos:
        for tech in all_techniques:
            row = f"{algo.upper()} + {tech:<15}"
            for env in all_envs:
                key = (algo, tech, env)
                if key in aggregated:
                    values = aggregated[key]
                    mean = np.mean(values)
                    std = np.std(values)
                    row += f" {mean:>6.3f}±{std:.2f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)
        print()

    print("\nBest Configuration per Environment:")
    print("-" * 40)
    for env in all_envs:
        best_key = None
        best_grs = -1
        for algo in all_algos:
            for tech in all_techniques:
                key = (algo, tech, env)
                if key in aggregated:
                    mean = np.mean(aggregated[key])
                    if mean > best_grs:
                        best_grs = mean
                        best_key = key
        if best_key:
            print(f"  {env}: {best_key[0].upper()} + {best_key[1]} (GRS={best_grs:.3f})")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # GRS heatmap
    heatmap_path = os.path.join(plots_dir, "grs_heatmap.png")
    plot_grs_heatmap(results, save_path=heatmap_path, show=False)
    print(f"  GRS heatmap saved: {heatmap_path}")

    # Text summary
    summary_path = os.path.join(results_dir, "experiment_summary.txt")
    create_experiment_summary(results, save_path=summary_path)

    # Check for best models
    metrics_tracker = MetricsTracker(results_dir)
    if metrics_tracker.best_scores:
        print("\nBest Models Saved:")
        for key, info in metrics_tracker.best_scores.items():
            print(f"  {key}: score={info['score']:.2f}, technique={info['technique']}")


def main():
    parser = argparse.ArgumentParser(description="RL Generalization Experiments")
    
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--isolated", action="store_true", help="Run with subprocess isolation per algorithm (recommended)")
    parser.add_argument("--algo", type=str, choices=ALGORITHMS, help="Algorithm")
    parser.add_argument("--algo-only", type=str, choices=ALGORITHMS, help="Run all techniques/envs for one algorithm")
    parser.add_argument("--technique", type=str, choices=TECHNIQUES, help="Technique")
    parser.add_argument("--env", type=str, choices=ENVIRONMENTS + ["starpilot"], help="Environment")
    parser.add_argument("--env-only", type=str, choices=ENVIRONMENTS + ["starpilot"], help="Run all for one env")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_results(args.results_dir)
    elif args.isolated:
        # Run with subprocess isolation - each algorithm in fresh Python process
        run_all_experiments_isolated(
            seeds=args.seeds,
            results_dir=args.results_dir,
            envs=[args.env_only] if args.env_only else None,
            algos=[args.algo_only] if args.algo_only else None,
        )
    elif args.all:
        run_all_experiments(
            seeds=args.seeds,
            results_dir=args.results_dir,
        )
    elif args.env_only and args.algo_only:
        # Run single algo on single env (all techniques)
        run_all_experiments(
            seeds=args.seeds,
            results_dir=args.results_dir,
            envs=[args.env_only],
            algos=[args.algo_only],
        )
    elif args.env_only:
        run_all_experiments(
            seeds=args.seeds,
            results_dir=args.results_dir,
            envs=[args.env_only],
        )
    elif args.algo_only:
        # Run single algo on all envs (all techniques)
        run_all_experiments(
            seeds=args.seeds,
            results_dir=args.results_dir,
            algos=[args.algo_only],
        )
    elif args.algo and args.technique and args.env:
        for seed in args.seeds:
            run_experiment(
                algo=args.algo,
                technique=args.technique,
                env_name=args.env,
                seed=seed,
                results_dir=args.results_dir,
            )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python run_experiment.py --all")
        print("  python run_experiment.py --env-only cartpole")
        print("  python run_experiment.py --algo-only dqn                    # Run all DQN experiments")
        print("  python run_experiment.py --algo-only ppo --env-only cartpole  # Run PPO on cartpole only")
        print("  python run_experiment.py --algo ppo --technique gar --env cartpole")
        print("  python run_experiment.py --analyze")


if __name__ == "__main__":
    main()