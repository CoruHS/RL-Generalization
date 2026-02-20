"""
GRS: Generalization Robustness Score

Measures how gracefully an agent's performance degrades as the environment
shifts further from training distribution.

GRS = Area under the normalized performance curve
- GRS ≈ 1.0: No degradation (perfect generalization)
- GRS ≈ 0.5: Linear degradation
- GRS ≈ 0.0: Immediate failure
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Callable
import json
import os


def _is_auto_reset_env(env) -> bool:
    """
    Detect if the environment auto-resets (like Procgen).

    Auto-resetting envs don't require explicit reset() calls between episodes.
    When an episode ends, the returned observation is from the new episode.
    """
    # Check class name first (most reliable)
    class_name = env.__class__.__name__.lower()
    if 'procgen' in class_name or 'starpilot' in class_name or 'coinrun' in class_name:
        return True

    # Check for Procgen-specific attributes
    if hasattr(env, 'num_levels') and hasattr(env, 'distribution_mode'):
        return True

    # Check if wrapped - with depth limit to prevent infinite loops
    check_env = env
    depth = 0
    max_depth = 10
    while hasattr(check_env, 'env') and depth < max_depth:
        if hasattr(check_env, 'env_name') or 'procgen' in str(type(check_env)).lower():
            return True
        check_env = check_env.env
        depth += 1

    return False


def evaluate_policy(
    agent,
    env,
    num_episodes: int = 20,
    max_steps: int = 1000,
) -> Tuple[float, float]:
    """
    Evaluate a policy on an environment.

    Handles both standard environments and auto-resetting environments like Procgen.

    Args:
        agent: Agent with predict(obs) method
        env: Environment to evaluate on
        num_episodes: Number of episodes
        max_steps: Max steps per episode

    Returns:
        mean_reward, std_reward
    """
    rewards = []
    is_auto_reset = _is_auto_reset_env(env)

    if is_auto_reset:
        # For auto-resetting envs, run continuously and count done signals
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0

        while len(rewards) < num_episodes:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1

            if terminated or truncated or step_count >= max_steps:
                rewards.append(episode_reward)
                episode_reward = 0
                step_count = 0
                # obs is already the first observation of the new episode
    else:
        # Standard evaluation loop for non-auto-resetting envs
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=ep)
            episode_reward = 0

            for step in range(max_steps):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            rewards.append(episode_reward)

    return np.mean(rewards), np.std(rewards)


def compute_grs(
    agent,
    env_class,
    shift_levels: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    num_episodes: int = 20,
    max_steps: int = 1000,
    env_kwargs: Optional[Dict] = None,
    env_wrapper: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Compute Generalization Robustness Score.

    Args:
        agent: Trained agent with predict() method
        env_class: Environment class with variation_level parameter
        shift_levels: List of shift levels to evaluate
        num_episodes: Episodes per evaluation
        max_steps: Max steps per episode
        env_kwargs: Additional kwargs for environment
        env_wrapper: Optional function to wrap environment (e.g., FrameStack)

    Returns:
        Dictionary with GRS score and curve data
    """
    env_kwargs = env_kwargs or {}

    performances = []
    stds = []

    for shift in shift_levels:
        # Create environment at this shift level
        env = env_class(variation_level=shift, **env_kwargs)
        # Apply wrapper if provided (e.g., FrameStack for Procgen)
        if env_wrapper is not None:
            env = env_wrapper(env)
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            agent, env, num_episodes, max_steps
        )
        
        performances.append(mean_reward)
        stds.append(std_reward)
        env.close()
    
    # Normalize by training performance (shift=0)
    baseline = performances[0]

    # Handle edge cases for baseline performance
    # If baseline is negative or near-zero, the agent failed to learn
    # In this case, GRS should be 0 (no generalization possible without learning)
    min_baseline_threshold = 0.01  # Minimum positive performance to consider "learned"

    if baseline <= min_baseline_threshold:
        # Agent didn't learn - can't measure generalization
        # Set GRS to 0 and normalized to show actual ratios for debugging
        if baseline != 0:
            normalized = [p / baseline for p in performances]
        else:
            normalized = [0.0 for _ in performances]
        grs = 0.0
    else:
        # Agent learned - compute normalized performance
        normalized = [p / baseline for p in performances]

        # Clamp normalized values to [0, 1] for cleaner GRS
        normalized_clamped = [max(0, min(1, n)) for n in normalized]

        # GRS = area under normalized curve (trapezoidal integration)
        grs = float(np.trapz(normalized_clamped, shift_levels))

    return {
        "grs": grs,
        "shift_levels": shift_levels,
        "performances": performances,
        "stds": stds,
        "normalized": normalized,
        "baseline": baseline,
    }


def compute_ggr(
    agent,
    env_class,
    train_shift: float = 0.0,
    test_shift: float = 0.5,
    num_episodes: int = 20,
    max_steps: int = 1000,
    env_kwargs: Optional[Dict] = None,
) -> float:
    """
    Compute Generalization Gap Ratio.
    
    GGR = test_performance / train_performance
    
    Simpler than GRS but less informative.
    """
    env_kwargs = env_kwargs or {}
    
    # Train environment
    train_env = env_class(variation_level=train_shift, **env_kwargs)
    train_perf, _ = evaluate_policy(agent, train_env, num_episodes, max_steps)
    train_env.close()
    
    # Test environment
    test_env = env_class(variation_level=test_shift, **env_kwargs)
    test_perf, _ = evaluate_policy(agent, test_env, num_episodes, max_steps)
    test_env.close()
    
    if train_perf == 0:
        return 0.0
    
    return test_perf / train_perf


def interpret_grs(grs: float) -> str:
    """Human-readable interpretation of GRS."""
    if grs >= 0.8:
        return "Excellent - Graceful degradation"
    elif grs >= 0.6:
        return "Good - Moderate degradation"
    elif grs >= 0.4:
        return "Fair - Significant degradation"
    else:
        return "Poor - Rapid failure"


def plot_grs_curve(
    result: Dict[str, Any],
    title: str = "Generalization Robustness Curve",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot GRS degradation curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw performance
    ax1 = axes[0]
    ax1.errorbar(
        result["shift_levels"],
        result["performances"],
        yerr=result["stds"],
        marker='o',
        capsize=5,
        linewidth=2,
        markersize=8,
        color='blue'
    )
    ax1.set_xlabel("Shift Level", fontsize=12)
    ax1.set_ylabel("Mean Reward", fontsize=12)
    ax1.set_title("Raw Performance", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    
    # Normalized with GRS shading
    ax2 = axes[1]
    ax2.fill_between(
        result["shift_levels"],
        result["normalized"],
        alpha=0.3,
        color='green',
        label=f'GRS = {result["grs"]:.3f}'
    )
    ax2.plot(
        result["shift_levels"],
        result["normalized"],
        marker='o',
        linewidth=2,
        markersize=8,
        color='green'
    )
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Shift Level", fontsize=12)
    ax2.set_ylabel("Normalized Performance", fontsize=12)
    ax2.set_title("Normalized (GRS = shaded area)", fontsize=14)
    ax2.legend(loc='lower left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.15)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def compare_grs(
    results: Dict[str, Dict[str, Any]],
    title: str = "GRS Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Compare GRS curves for multiple agents."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, result), color in zip(results.items(), colors):
        ax.plot(
            result["shift_levels"],
            result["normalized"],
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'{name} (GRS={result["grs"]:.3f})',
            color=color,
        )
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Shift Level", fontsize=12)
    ax.set_ylabel("Normalized Performance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def create_profile_card(
    grs_by_variation: Dict[str, float],
    name: str,
) -> str:
    """Create text-based generalization profile card."""
    def bar(val, width=10):
        filled = int(val * width)
        return "█" * filled + "░" * (width - filled)
    
    lines = [
        "┌" + "─" * 44 + "┐",
        f"│  AGENT: {name:<33} │",
        "├" + "─" * 44 + "┤",
    ]
    
    for var_type, grs in grs_by_variation.items():
        lines.append(f"│  {var_type:<15} {bar(grs)} {grs:.2f}  │")
    
    avg = np.mean(list(grs_by_variation.values()))
    best = max(grs_by_variation, key=grs_by_variation.get)
    worst = min(grs_by_variation, key=grs_by_variation.get)
    
    lines.extend([
        "├" + "─" * 44 + "┤",
        f"│  Overall GRS: {avg:.2f}                          │",
        f"│  Best for:    {best:<28} │",
        f"│  Avoid for:   {worst:<28} │",
        "└" + "─" * 44 + "┘",
    ])
    
    return "\n".join(lines)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing GRS module...")
    
    # Mock agent for testing
    class MockAgent:
        def __init__(self, skill=0.5):
            self.skill = skill
        
        def predict(self, obs, deterministic=True):
            # Returns random action, simulating degradation with shift
            return np.random.randint(2), None
    
    # Mock environment
    class MockEnv:
        def __init__(self, variation_level=0.0):
            self.variation_level = variation_level
            self.steps = 0
        
        def reset(self, seed=None):
            self.steps = 0
            return np.zeros(4), {}
        
        def step(self, action):
            self.steps += 1
            # Performance degrades with variation
            reward = 1.0 * (1 - 0.5 * self.variation_level)
            done = self.steps >= 100
            return np.zeros(4), reward, done, False, {}
        
        def close(self):
            pass
    
    # Test GRS computation
    agent = MockAgent()
    result = compute_grs(
        agent,
        MockEnv,
        shift_levels=[0.0, 0.25, 0.5, 0.75, 1.0],
        num_episodes=3,
        max_steps=100,
    )
    
    print(f"GRS: {result['grs']:.3f}")
    print(f"Performances: {result['performances']}")
    print(f"Interpretation: {interpret_grs(result['grs'])}")
    
    # Test profile card
    print("\n" + create_profile_card(
        {"Dynamics": 0.72, "Layout": 0.45, "Visual": 0.33},
        "PPO + GAR"
    ))
    
    print("\nGRS tests passed!")