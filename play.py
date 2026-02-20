#!/usr/bin/env python3
"""
Watch trained agents play!

Usage:
    python play.py --env cartpole --algo dqn
    python play.py --env minigrid --algo ppo
    python play.py --env space_invaders --algo dqn

    # Train a quick agent if no saved model exists
    python play.py --env cartpole --algo dqn --train

    # Specify number of episodes
    python play.py --env cartpole --algo ppo --episodes 5
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym


def play_cartpole(agent, num_episodes=3, delay=0.02):
    """Play CartPole with pygame rendering."""
    from envs.cartpole_var import CartPoleVarEnv

    env = gym.make("CartPole-v1", render_mode="human")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            time.sleep(delay)

        print(f"Episode {ep+1}: {total_reward:.0f} reward, {steps} steps")

    env.close()


def play_minigrid(agent, num_episodes=3, delay=0.2):
    """Play MiniGrid with pygame rendering."""
    from envs.minigrid_var import MiniGridVarEnv

    env = MiniGridVarEnv(variation_level=0.0, use_compact_obs=True, render_mode="human")

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        steps = 0

        print(f"\nEpisode {ep+1} starting...")

        while not done:
            env.render()

            # Get action
            action, _ = agent.predict(obs, deterministic=True)
            action_names = ["LEFT", "RIGHT", "FORWARD"]

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if reward > 0.5:
                print(f"  Step {steps}: GOAL REACHED! +{reward:.2f}")

            time.sleep(delay)

        result = "GOAL!" if total_reward > 0.5 else "FAILED"
        print(f"Episode {ep+1} {result}: {total_reward:.2f} reward, {steps} steps")
        time.sleep(0.5)

    env.close()


def play_space_invaders(agent, num_episodes=3, delay=0.05, use_gui=True):
    """Play Space Invaders with matplotlib or text rendering."""
    from envs.minatar_var import MinAtarVarEnv
    import matplotlib
    matplotlib.use('TkAgg')  # Use interactive backend
    import matplotlib.pyplot as plt

    env = MinAtarVarEnv(variation_level=0.0, game="space_invaders")

    # Channel indices for text fallback
    CANNON = 0
    ALIEN = 1
    FRIENDLY_BULLET = 4
    ENEMY_BULLET = 5

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        steps = 0

        print(f"\nEpisode {ep+1} starting...")

        if use_gui:
            plt.ion()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_title("Space Invaders")

        while not done:
            if use_gui:
                # Use matplotlib to display
                ax.clear()
                ax.set_title(f"Space Invaders | Score: {total_reward:.0f} | Step: {steps}")
                # Create RGB image from channels (obs is float32, use np.maximum for combining)
                img = np.zeros((10, 10, 3))
                img[:, :, 2] = obs[:, :, CANNON] * 255  # Blue = cannon
                # Combine alien channels with maximum (they're float, not bool)
                aliens = np.maximum(np.maximum(obs[:, :, ALIEN], obs[:, :, 2]), obs[:, :, 3])
                img[:, :, 0] = aliens * 255  # Red = aliens
                img[:, :, 1] = obs[:, :, FRIENDLY_BULLET] * 255  # Green = friendly bullet
                img[:, :, 0] += obs[:, :, ENEMY_BULLET] * 128  # Red-ish = enemy bullet
                img = np.clip(img, 0, 255).astype(np.uint8)
                ax.imshow(img, interpolation='nearest')
                ax.axis('off')
                plt.pause(delay)
            else:
                # Text rendering fallback
                os.system('clear' if os.name == 'posix' else 'cls')
                print(f"SPACE INVADERS | Step {steps} | Score: {total_reward:.0f}")
                print("-" * 22)
                for row in range(10):
                    line = "|"
                    for col in range(10):
                        if obs[row, col, CANNON] > 0.5:
                            line += "A "
                        elif obs[row, col, ALIEN] > 0.5 or obs[row, col, 2] > 0.5 or obs[row, col, 3] > 0.5:
                            line += "M "
                        elif obs[row, col, FRIENDLY_BULLET] > 0.5:
                            line += "^ "
                        elif obs[row, col, ENEMY_BULLET] > 0.5:
                            line += "v "
                        else:
                            line += "  "
                    print(line + "|")
                print("-" * 22)
                time.sleep(delay)

            # Get action
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        if use_gui:
            plt.ioff()
            plt.close(fig)

        print(f"Episode {ep+1} GAME OVER! Score: {total_reward:.0f} in {steps} steps")
        time.sleep(0.5)

    env.close()


def quick_train(env_name, algo, steps=10000):
    """Quickly train an agent for demo purposes."""
    from run_experiment import make_env, HYPERPARAMS

    env = make_env(env_name, variation_level=0.0)
    hp = HYPERPARAMS.get(env_name, {}).get(algo, {})

    print(f"Quick training {algo.upper()} on {env_name} for {steps} steps...")

    if algo == "dqn":
        from agents.dqn import DQN
        agent = DQN(
            env=env,
            learning_rate=hp.get("learning_rate", 1e-4),
            buffer_size=min(steps, 50000),
            hidden_dims=hp.get("hidden_dims", [64, 64]),
        )
        agent.learn(total_timesteps=steps, log_interval=steps//5, eval_interval=0)

    elif algo == "ppo":
        from agents.ppo import PPO
        agent = PPO(
            env=env,
            learning_rate=hp.get("learning_rate", 3e-4),
            n_steps=hp.get("n_steps", 512),
            hidden_dims=hp.get("hidden_dims", [64, 64]),
        )
        agent.learn(total_timesteps=steps, log_interval=1, eval_interval=0)

    elif algo == "es":
        from agents.es import ES
        agent = ES(
            env=env,
            population_size=hp.get("population_size", 30),
            learning_rate=hp.get("learning_rate", 0.03),
            hidden_dims=hp.get("hidden_dims", [64, 64]),
        )
        generations = max(50, steps // 500)
        agent.learn(total_generations=generations, log_interval=10, eval_interval=0)

    print(f"Training done! Episodes: {getattr(agent, 'episodes', 'N/A')}")
    return agent


def load_or_train(env_name, algo, train_if_missing=False):
    """Load a saved model or train a new one."""
    from run_experiment import make_env, HYPERPARAMS

    # Look for saved models
    results_dir = "results/best_models"
    model_patterns = [
        f"{algo}_*_{env_name}_*.pth",
        f"{algo}_{env_name}*.pth",
    ]

    model_path = None
    if os.path.exists(results_dir):
        for f in os.listdir(results_dir):
            # Check for both .pt and .pth extensions
            if f.startswith(f"{algo}_") and env_name in f and (f.endswith(".pt") or f.endswith(".pth")):
                model_path = os.path.join(results_dir, f)
                break

    # Also check results directory
    if model_path is None and os.path.exists("results"):
        for f in os.listdir("results"):
            if f.startswith(f"{algo}_") and env_name in f and (f.endswith(".pt") or f.endswith(".pth")):
                model_path = os.path.join("results", f)
                break

    env = make_env(env_name, variation_level=0.0)
    hp = HYPERPARAMS.get(env_name, {}).get(algo, {})

    # Create agent
    if algo == "dqn":
        from agents.dqn import DQN
        agent = DQN(
            env=env,
            learning_rate=hp.get("learning_rate", 1e-4),
            buffer_size=1000,  # Small buffer for loading
            hidden_dims=hp.get("hidden_dims", [64, 64]),
        )
    elif algo == "ppo":
        from agents.ppo import PPO
        agent = PPO(
            env=env,
            learning_rate=hp.get("learning_rate", 3e-4),
            hidden_dims=hp.get("hidden_dims", [64, 64]),
        )
    elif algo == "es":
        from agents.es import ES
        agent = ES(
            env=env,
            population_size=hp.get("population_size", 30),
            hidden_dims=hp.get("hidden_dims", [64, 64]),
        )

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load(model_path)
        return agent
    elif train_if_missing:
        print(f"No saved model found. Training a quick demo agent...")
        return quick_train(env_name, algo)
    else:
        print(f"No saved model found at {results_dir}")
        print(f"Run with --train to train a quick demo agent")
        print(f"Or run experiments first: python run_experiment.py --env-only {env_name} --algo-only {algo}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Watch trained agents play!")
    parser.add_argument("--env", type=str, required=True,
                        choices=["cartpole", "minigrid", "space_invaders"],
                        help="Environment to play")
    parser.add_argument("--algo", type=str, required=True,
                        choices=["dqn", "ppo", "es"],
                        help="Algorithm to use")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to play")
    parser.add_argument("--train", action="store_true",
                        help="Train a quick agent if no saved model exists")
    parser.add_argument("--delay", type=float, default=None,
                        help="Delay between frames (seconds)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Use text rendering instead of GUI (for SSH)")

    args = parser.parse_args()

    # Load or train agent
    agent = load_or_train(args.env, args.algo, train_if_missing=args.train)
    if agent is None:
        return

    # Play!
    print(f"\nWatching {args.algo.upper()} play {args.env}...")
    print("Press Ctrl+C to stop\n")
    time.sleep(1)

    try:
        if args.env == "cartpole":
            delay = args.delay if args.delay else 0.02
            play_cartpole(agent, args.episodes, delay)
        elif args.env == "minigrid":
            delay = args.delay if args.delay else 0.2
            play_minigrid(agent, args.episodes, delay)
        elif args.env == "space_invaders":
            delay = args.delay if args.delay else 0.05
            play_space_invaders(agent, args.episodes, delay, use_gui=not args.no_gui)
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    main()
