"""
Comprehensive Metrics Tracking and Visualization

Provides:
- Training metrics logging
- Best model tracking per architecture
- Learning curves and performance plots
- Comparison visualizations
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
import pickle


class MetricsTracker:
    """Tracks training metrics and saves best models per architecture."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.models_dir = os.path.join(results_dir, "best_models")
        self.plots_dir = os.path.join(results_dir, "plots")
        self.metrics_dir = os.path.join(results_dir, "metrics")

        for d in [self.results_dir, self.models_dir, self.plots_dir, self.metrics_dir]:
            os.makedirs(d, exist_ok=True)

        # Track best performance per architecture per environment
        self.best_scores: Dict[str, Dict[str, float]] = {}
        self._load_best_scores()

    def _load_best_scores(self):
        """Load existing best scores if available."""
        scores_path = os.path.join(self.models_dir, "best_scores.json")
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                self.best_scores = json.load(f)

    def _save_best_scores(self):
        """Save best scores."""
        scores_path = os.path.join(self.models_dir, "best_scores.json")
        with open(scores_path, 'w') as f:
            json.dump(self.best_scores, f, indent=2)

    def update_best_model(
        self,
        agent,
        algo: str,
        env_name: str,
        eval_reward: float,
        technique: str,
        seed: int,
    ) -> bool:
        """
        Check if this is the best model for this architecture+environment.
        If so, save it.

        Returns True if this was a new best.
        """
        key = f"{algo}_{env_name}"

        if key not in self.best_scores:
            self.best_scores[key] = {"score": float('-inf'), "technique": "", "seed": -1}

        if eval_reward > self.best_scores[key]["score"]:
            self.best_scores[key] = {
                "score": eval_reward,
                "technique": technique,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
            }

            # Save the model
            model_path = os.path.join(self.models_dir, f"{key}_best.pt")
            agent.save(model_path)

            self._save_best_scores()
            print(f"  New best {algo.upper()} on {env_name}: {eval_reward:.2f} (technique={technique}, seed={seed})")
            return True

        return False

    def save_training_history(
        self,
        history: Dict[str, List],
        algo: str,
        technique: str,
        env_name: str,
        seed: int,
    ):
        """Save detailed training history."""
        filename = f"{algo}_{technique}_{env_name}_seed{seed}_history.json"
        path = os.path.join(self.metrics_dir, filename)

        # Convert numpy arrays to lists
        history_json = {}
        for k, v in history.items():
            if isinstance(v, np.ndarray):
                history_json[k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                history_json[k] = [x.tolist() for x in v]
            else:
                history_json[k] = v

        with open(path, 'w') as f:
            json.dump(history_json, f, indent=2)

    def get_best_model_path(self, algo: str, env_name: str) -> Optional[str]:
        """Get path to best model for an architecture+environment."""
        key = f"{algo}_{env_name}"
        model_path = os.path.join(self.models_dir, f"{key}_best.pt")
        if os.path.exists(model_path):
            return model_path
        return None

    def get_best_score(self, algo: str, env_name: str) -> Optional[Dict]:
        """Get best score info for an architecture+environment."""
        key = f"{algo}_{env_name}"
        return self.best_scores.get(key)


def plot_learning_curves(
    history: Dict[str, List],
    algo: str,
    technique: str,
    env_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot comprehensive learning curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Determine which metrics are available
    has_episode_rewards = "episode_rewards" in history and len(history["episode_rewards"]) > 0
    has_losses = "losses" in history and len(history["losses"]) > 0
    has_policy_losses = "policy_losses" in history and len(history["policy_losses"]) > 0
    has_eval_rewards = "eval_rewards" in history and len(history["eval_rewards"]) > 0
    has_epsilons = "epsilons" in history and len(history["epsilons"]) > 0
    has_entropies = "entropies" in history and len(history["entropies"]) > 0
    has_generation_fitness = "generation_fitness" in history and len(history["generation_fitness"]) > 0

    # Plot 1: Episode Rewards / Fitness
    ax1 = axes[0, 0]
    if has_episode_rewards:
        rewards = history["episode_rewards"]
        ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
        # Smoothed
        window = min(50, len(rewards) // 10 + 1)
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=2, label=f'Smoothed (w={window})')
        ax1.set_ylabel("Episode Reward")
        ax1.legend()
    elif has_generation_fitness:
        fitness = history["generation_fitness"]
        ax1.plot(fitness, color='blue', linewidth=2)
        ax1.set_ylabel("Generation Fitness")
    ax1.set_xlabel("Episode / Generation")
    ax1.set_title("Training Reward")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss
    ax2 = axes[0, 1]
    if has_losses:
        losses = history["losses"]
        ax2.plot(losses, alpha=0.3, color='red', label='Raw')
        window = min(100, len(losses) // 10 + 1)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(losses)), smoothed, color='red', linewidth=2, label=f'Smoothed')
        ax2.set_ylabel("Loss")
        ax2.legend()
    elif has_policy_losses:
        policy_losses = history["policy_losses"]
        value_losses = history.get("value_losses", [])
        ax2.plot(policy_losses, color='red', linewidth=2, label='Policy Loss')
        if value_losses:
            ax2.plot(value_losses, color='orange', linewidth=2, label='Value Loss')
        ax2.set_ylabel("Loss")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No loss data", ha='center', va='center', transform=ax2.transAxes)
    ax2.set_xlabel("Update Step")
    ax2.set_title("Training Loss")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Eval Rewards
    ax3 = axes[1, 0]
    if has_eval_rewards:
        eval_rewards = history["eval_rewards"]
        ax3.plot(eval_rewards, marker='o', color='green', linewidth=2, markersize=6)
        ax3.axhline(y=np.max(eval_rewards), color='green', linestyle='--', alpha=0.5, label=f'Best: {np.max(eval_rewards):.2f}')
        ax3.set_ylabel("Eval Reward")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No eval data", ha='center', va='center', transform=ax3.transAxes)
    ax3.set_xlabel("Evaluation #")
    ax3.set_title("Evaluation Performance")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Epsilon / Entropy
    ax4 = axes[1, 1]
    if has_epsilons:
        ax4.plot(history["epsilons"], color='purple', linewidth=2)
        ax4.set_ylabel("Epsilon")
        ax4.set_title("Exploration Rate")
    elif has_entropies:
        ax4.plot(history["entropies"], color='purple', linewidth=2)
        ax4.set_ylabel("Entropy")
        ax4.set_title("Policy Entropy")
    else:
        ax4.text(0.5, 0.5, "No exploration data", ha='center', va='center', transform=ax4.transAxes)
    ax4.set_xlabel("Episode / Update")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"{algo.upper()} + {technique} on {env_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved learning curves: {save_path}")

    if show:
        plt.show()

    plt.close()


def plot_comparison(
    all_histories: Dict[str, Dict[str, List]],
    env_name: str,
    metric: str = "eval_rewards",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Compare multiple experiments on the same plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))

    for (name, history), color in zip(all_histories.items(), colors):
        if metric in history and len(history[metric]) > 0:
            data = history[metric]
            ax.plot(data, label=name, color=color, linewidth=2)

    ax.set_xlabel("Evaluation #" if "eval" in metric else "Step")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Comparison on {env_name}")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def plot_grs_heatmap(
    results: List[Dict],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Create heatmap of GRS scores across algorithms, techniques, and environments."""
    # Organize data
    algos = sorted(set(r["algorithm"] for r in results))
    techniques = sorted(set(r["technique"] for r in results))
    envs = sorted(set(r["environment"] for r in results))

    n_configs = len(algos) * len(techniques)
    n_envs = len(envs)

    data = np.zeros((n_configs, n_envs))
    labels = []

    for i, algo in enumerate(algos):
        for j, tech in enumerate(techniques):
            labels.append(f"{algo.upper()}+{tech}")
            for k, env in enumerate(envs):
                matching = [r for r in results if r["algorithm"] == algo and r["technique"] == tech and r["environment"] == env]
                if matching:
                    data[i * len(techniques) + j, k] = np.mean([r["grs"] for r in matching])

    fig, ax = plt.subplots(figsize=(10, max(6, n_configs * 0.5)))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(n_envs))
    ax.set_xticklabels(envs)
    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(labels)

    # Add text annotations
    for i in range(n_configs):
        for j in range(n_envs):
            text = ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                           color="black" if 0.3 < data[i, j] < 0.7 else "white")

    ax.set_title("GRS Scores Heatmap", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='GRS Score')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def create_experiment_summary(
    results: List[Dict],
    save_path: Optional[str] = None,
) -> str:
    """Create a text summary of all experiments."""
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total experiments: {len(results)}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Group by environment
    envs = sorted(set(r["environment"] for r in results))
    algos = sorted(set(r["algorithm"] for r in results))
    techniques = sorted(set(r["technique"] for r in results))

    for env in envs:
        lines.append(f"\n{'='*70}")
        lines.append(f"Environment: {env.upper()}")
        lines.append(f"{'='*70}")

        env_results = [r for r in results if r["environment"] == env]

        # Table header
        header = f"{'Config':<20} {'GRS':>8} {'Train Perf':>12} {'Time (s)':>10}"
        lines.append(header)
        lines.append("-" * 55)

        # Sort by GRS
        env_results.sort(key=lambda x: x.get("grs", 0), reverse=True)

        for r in env_results:
            config = f"{r['algorithm'].upper()}+{r['technique']}"
            grs = r.get("grs", 0)
            train_perf = r.get("train_performance", 0)
            train_time = r.get("train_time", 0)
            lines.append(f"{config:<20} {grs:>8.3f} {train_perf:>12.2f} {train_time:>10.1f}")

        # Best per architecture
        lines.append("\nBest per architecture:")
        for algo in algos:
            algo_results = [r for r in env_results if r["algorithm"] == algo]
            if algo_results:
                best = max(algo_results, key=lambda x: x.get("grs", 0))
                lines.append(f"  {algo.upper()}: {best['technique']} (GRS={best.get('grs', 0):.3f})")

    summary = "\n".join(lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary)
        print(f"Saved summary to: {save_path}")

    return summary


if __name__ == "__main__":
    print("Testing metrics module...")

    # Create mock history
    history = {
        "episode_rewards": list(np.random.randn(100).cumsum() + 100),
        "losses": list(np.exp(-np.linspace(0, 3, 500)) + np.random.randn(500) * 0.1),
        "eval_rewards": list(np.linspace(50, 150, 10) + np.random.randn(10) * 10),
        "epsilons": list(np.linspace(1.0, 0.05, 100)),
    }

    plot_learning_curves(
        history,
        algo="dqn",
        technique="baseline",
        env_name="cartpole",
        show=True,
    )

    print("Metrics module tests passed!")
