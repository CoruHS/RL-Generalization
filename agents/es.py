"""
ES (Evolution Strategies) Algorithm
"""

import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple, List
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.networks import ESNetwork


class ES:
    def __init__(
        self,
        env: gym.Env,
        population_size: int = 50,
        sigma: float = 0.1,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        hidden_dims: list = [64, 64],
        device: str = "cpu",
    ):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        
        self.device = device
        
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.use_cnn = len(self.obs_shape) == 3
        obs_dim = self.obs_shape if self.use_cnn else self.obs_shape[0]
        
        self.policy = ESNetwork(
            obs_dim=obs_dim,
            action_dim=self.n_actions,
            hidden_dims=hidden_dims,
            use_cnn=self.use_cnn,
        ).to(self.device)


        self.param_count = sum(p.numel() for p in self.policy.parameters())
        print(f"Total parameters: {self.param_count}")
        
        self.total_steps = 0
        self.generations = 0
    
    def evaluate_policy(self, policy: ESNetwork, num_episodes: int = 1, max_steps: int = 1000) -> float:
        """
        Evaluate a policy's performance over multiple episodes.

        Handles both standard environments and auto-resetting environments like Procgen.

        Args:
            policy: The policy network to evaluate
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode

        Returns:
            Mean reward across all episodes
        """
        total_rewards = []
        is_auto_reset = self._is_auto_reset_env()

        if is_auto_reset:
            # For auto-resetting envs, run continuously and count done signals
            obs, _ = self.env.reset()
            episode_reward = 0
            step_count = 0

            while len(total_rewards) < num_episodes:
                obs_tensor = torch.from_numpy(obs).float().to(self.device)
                action = policy.get_action(obs_tensor, deterministic=True)

                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                step_count += 1

                if terminated or truncated or step_count >= max_steps:
                    total_rewards.append(episode_reward)
                    episode_reward = 0
                    step_count = 0
                    # obs is already the first observation of the new episode
        else:
            # Standard evaluation loop for non-auto-resetting envs
            for _ in range(num_episodes):
                obs, _ = self.env.reset()
                episode_reward = 0

                for step in range(max_steps):
                    obs_tensor = torch.from_numpy(obs).float().to(self.device)
                    action = policy.get_action(obs_tensor, deterministic=True)

                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    episode_reward += reward

                    if terminated or truncated:
                        break

                total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def _is_auto_reset_env(self) -> bool:
        """
        Detect if the environment auto-resets (like Procgen).

        Auto-resetting envs don't require explicit reset() calls between episodes.
        When an episode ends, the returned observation is from the new episode.
        """
        # Check class name first (most reliable)
        class_name = self.env.__class__.__name__.lower()
        if 'procgen' in class_name or 'starpilot' in class_name or 'coinrun' in class_name:
            return True

        # Check for Procgen-specific attributes (need BOTH to be sure)
        if hasattr(self.env, 'num_levels') and hasattr(self.env, 'distribution_mode'):
            return True

        # Check if wrapped - with depth limit to prevent infinite loops
        env = self.env
        depth = 0
        max_depth = 10
        while hasattr(env, 'env') and depth < max_depth:
            if hasattr(env, 'env_name') or 'procgen' in str(type(env)).lower():
                return True
            env = env.env
            depth += 1

        return False
    
    def generate_perturbations(self) -> List[np.ndarray]:
        perturbations = []
        for _ in range(self.population_size):
            perturbation = np.random.randn(self.param_count).astype(np.float32)
            perturbations.append(perturbation)
        return perturbations
    
    def apply_perturbation(self, base_params: np.ndarray, perturbation: np.ndarray, direction: int = 1) -> ESNetwork:
        obs_dim = self.obs_shape if self.use_cnn else self.obs_shape[0]
        perturbed_policy = ESNetwork(
            obs_dim=obs_dim,
            action_dim=self.n_actions,
            use_cnn=self.use_cnn,
        ).to(self.device)
        
        new_params = base_params + direction * self.sigma * perturbation
        perturbed_policy.set_flat_params(new_params)
        
        return perturbed_policy
    
    def update(self, perturbations: List[np.ndarray], fitnesses: np.ndarray):
        fitnesses = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        
        weighted_sum = np.zeros(self.param_count, dtype=np.float32)
        for i, perturbation in enumerate(perturbations):
            weighted_sum += fitnesses[i] * perturbation
        
        current_params = self.policy.get_flat_params()
        gradient_estimate = weighted_sum / (self.population_size * self.sigma)
        
        # Apply weight decay
        gradient_estimate -= self.weight_decay * current_params
        
        new_params = current_params + self.learning_rate * gradient_estimate
        self.policy.set_flat_params(new_params)
    
    def train_generation(self, episodes_per_eval: int = 1, max_steps: int = 1000) -> Dict[str, float]:
        base_params = self.policy.get_flat_params()
        perturbations = self.generate_perturbations()
        fitnesses = []
        
        for perturbation in perturbations:
            policy_plus = self.apply_perturbation(base_params, perturbation, direction=+1)
            fitness_plus = self.evaluate_policy(policy_plus, episodes_per_eval, max_steps)
            
            policy_minus = self.apply_perturbation(base_params, perturbation, direction=-1)
            fitness_minus = self.evaluate_policy(policy_minus, episodes_per_eval, max_steps)
            
            fitnesses.append(fitness_plus - fitness_minus)
            self.total_steps += episodes_per_eval * 2 * max_steps
        
        fitnesses = np.array(fitnesses)
        self.update(perturbations, fitnesses)
        self.generations += 1
        
        current_fitness = self.evaluate_policy(self.policy, episodes_per_eval, max_steps)
        
        return {
            "mean_fitness": current_fitness,
            "max_fitness": fitnesses.max(),
            "min_fitness": fitnesses.min(),
            "std_fitness": fitnesses.std(),
        }
    
    def learn(
        self,
        total_generations: int,
        episodes_per_eval: int = 1,
        max_steps: int = 1000,
        log_interval: int = 1,
        eval_interval: int = 10,
        eval_episodes: int = 10,
        callback=None,
    ) -> Dict[str, list]:
        history = {
            "generation_fitness": [],
            "eval_rewards": [],
        }
        
        for gen in range(total_generations):
            stats = self.train_generation(episodes_per_eval, max_steps)
            history["generation_fitness"].append(stats["mean_fitness"])
            
            if (gen + 1) % log_interval == 0:
                print(f"Generation {gen + 1}/{total_generations} | "
                      f"Fitness: {stats['mean_fitness']:.2f} | "
                      f"Max: {stats['max_fitness']:.2f} | "
                      f"Std: {stats['std_fitness']:.2f}")
            
            if eval_interval > 0 and (gen + 1) % eval_interval == 0:
                eval_reward = self.evaluate(eval_episodes, max_steps)
                history["eval_rewards"].append(eval_reward)
                print(f"  Eval reward: {eval_reward:.2f}")
            
            if callback is not None:
                callback(self, gen)
        
        return history
    
    def evaluate(self, num_episodes: int = 10, max_steps: int = 1000) -> float:
        return self.evaluate_policy(self.policy, num_episodes, max_steps)
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        action = self.policy.get_action(obs_tensor, deterministic=deterministic)
        return action, None
    
    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "generations": self.generations,
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.generations = checkpoint["generations"]
        self.total_steps = checkpoint["total_steps"]


class OpenAIES(ES):
    """OpenAI's ES with Adam optimizer and rank-based fitness."""
    
    def __init__(
        self,
        env: gym.Env,
        population_size: int = 50,
        sigma: float = 0.1,
        learning_rate: float = 0.01,
        weight_decay: float = 0.01,
        hidden_dims: list = [64, 64],
        device: str = "cpu",
    ):
        super().__init__(
            env=env,
            population_size=population_size,
            sigma=sigma,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_dims=hidden_dims,
            device=device,
        )
        
        self.m = np.zeros(self.param_count, dtype=np.float32)
        self.v = np.zeros(self.param_count, dtype=np.float32)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
    
    def compute_ranks(self, fitnesses: np.ndarray) -> np.ndarray:
        ranks = np.empty(len(fitnesses), dtype=np.float32)
        ranks[fitnesses.argsort()] = np.arange(len(fitnesses))
        ranks = ranks / (len(fitnesses) - 1) - 0.5
        return ranks
    
    def update(self, perturbations: List[np.ndarray], fitnesses: np.ndarray):
        ranked_fitnesses = self.compute_ranks(fitnesses)
        
        gradient = np.zeros(self.param_count, dtype=np.float32)
        for i, perturbation in enumerate(perturbations):
            gradient += ranked_fitnesses[i] * perturbation
        gradient = gradient / (self.population_size * self.sigma)
        
        current_params = self.policy.get_flat_params()
        gradient = gradient - self.weight_decay * current_params
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** (self.generations + 1))
        v_hat = self.v / (1 - self.beta2 ** (self.generations + 1))
        
        step = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        new_params = current_params + step
        
        self.policy.set_flat_params(new_params)


if __name__ == "__main__":
    print("Testing ES...")
    
    env = gym.make("CartPole-v1")
    agent = ES(
        env=env,
        population_size=20,
        sigma=0.1,
        learning_rate=0.03,
        weight_decay=0.01,
    )
    
    history = agent.learn(
        total_generations=20,
        episodes_per_eval=1,
        max_steps=500,
        log_interval=5,
        eval_interval=10,
    )
    
    final_reward = agent.evaluate(num_episodes=10)
    print(f"Final mean reward: {final_reward:.2f}")
    
    env.close()
    print("ES test complete!")