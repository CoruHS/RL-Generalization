"""
DQN (Deep Q-Network) Algorithm
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.networks import DQNNetwork
from core.replay_buffer import ReplayBuffer
from core.gar import (
    GARBuffer,
    GARStats,
    compute_gradient_dict,
    compute_gradient_agreement,
    compute_mean_agreement,
    apply_gradients,
)


class DQN:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        target_update_freq: int = 2000,
        train_freq: int = 4,
        learning_starts: int = 1000,
        weight_decay: float = 0.0,
        hidden_dims: list = [64, 64],
        device: str = "auto",
        # GAR parameters
        use_gar: bool = False,
        gar_variation_levels: list = [0.0, 0.25, 0.5],
        gar_buffer_size: int = 5000,
        gar_min_samples: int = 100,
        env_class: type = None,
        env_kwargs: dict = None,
        env_wrapper: callable = None,  # Wrapper to apply to GAR envs (e.g., FrameStack)
    ):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.n_actions = env.action_space.n

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq
        self.learning_starts = learning_starts

        self.use_cnn = len(self.obs_shape) == 3
        obs_dim = self.obs_shape if self.use_cnn else self.obs_shape[0]

        # Detect if observations need normalization (e.g., MiniGrid with discrete cell types)
        # Check observation space bounds to determine normalization factor
        obs_high = env.observation_space.high
        if not self.use_cnn and obs_high is not None:
            max_val = obs_high.max() if hasattr(obs_high, 'max') else obs_high
            # If max value is small (like 4 for MiniGrid cell types), normalize
            self.obs_normalize = max_val if max_val > 1.0 and max_val < 256 else 1.0
        else:
            self.obs_normalize = 1.0
        
        self.q_network = DQNNetwork(
            obs_dim=obs_dim,
            action_dim=self.n_actions,
            hidden_dims=hidden_dims,
            use_cnn=self.use_cnn,
        ).to(self.device)

        self.target_network = DQNNetwork(
            obs_dim=obs_dim,
            action_dim=self.n_actions,
            hidden_dims=hidden_dims,
            use_cnn=self.use_cnn,
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        # AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.q_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            obs_shape=self.obs_shape,
            device=self.device,
        )
        
        self.total_steps = 0
        self.episodes = 0
        self.training_logs = []

        # GAR setup
        self.use_gar = use_gar
        self.gar_variation_levels = gar_variation_levels
        self.gar_min_samples = gar_min_samples
        self.env_class = env_class
        self.env_kwargs = env_kwargs or {}
        self.env_wrapper = env_wrapper  # e.g., FrameStack for Procgen/Starpilot

        if self.use_gar:
            # Create GAR buffer for storing experiences from shifted environments
            self.gar_buffer = GARBuffer(
                capacity_per_level=gar_buffer_size,
                obs_shape=self.obs_shape,
                device=self.device,
            )
            self.gar_stats = GARStats()

            # Create shifted environments for collecting experiences
            if self.env_class is not None:
                self.gar_envs = {}
                for level in self.gar_variation_levels:
                    if level != 0.0:  # Don't duplicate the main env
                        env = self.env_class(variation_level=level, **self.env_kwargs)
                        # Apply wrapper if provided (e.g., FrameStack for Starpilot)
                        if self.env_wrapper is not None:
                            env = self.env_wrapper(env)
                        self.gar_envs[level] = env
            else:
                self.gar_envs = {}
        else:
            self.gar_buffer = None
            self.gar_stats = None
            self.gar_envs = {}
    
    def get_epsilon(self) -> float:
        progress = min(1.0, self.total_steps / self.epsilon_decay)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
    
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> int:
        epsilon = 0.0 if evaluate else self.get_epsilon()

        if np.random.random() < epsilon:
            return self.env.action_space.sample()

        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        # Normalize observations if needed (e.g., MiniGrid cell types 0-4 -> 0-1)
        if self.obs_normalize > 1.0:
            obs_tensor = obs_tensor / self.obs_normalize

        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
            return q_values.argmax(dim=1).item()
    
    def _compute_dqn_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DQN loss for a batch of transitions."""
        # Normalize observations if needed
        if self.obs_normalize > 1.0:
            obs = obs / self.obs_normalize
            next_obs = next_obs / self.obs_normalize

        current_q = self.q_network(obs)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_obs)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Use Huber loss for more stable training (less sensitive to outliers)
        return F.smooth_l1_loss(current_q, target_q)

    def train_step(self) -> Optional[float]:
        if not self.buffer.is_ready(self.batch_size):
            return None

        if self.use_gar:
            return self._train_step_gar()
        else:
            return self._train_step_standard()

    def _train_step_standard(self) -> float:
        """Standard DQN training step without GAR."""
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        loss = self._compute_dqn_loss(obs, actions, rewards, next_obs, dones)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def _train_step_gar(self) -> float:
        """
        GAR training step: compute gradients from multiple variation levels
        and only update in directions where gradients agree.
        """
        # Get available variation levels with enough samples
        available_levels = self.gar_buffer.get_available_levels(
            min_samples=self.gar_min_samples
        )

        # Always include the main buffer (level 0.0)
        if 0.0 not in available_levels:
            available_levels = [0.0] + available_levels

        # Compute gradients from each variation level
        grads_list = []
        losses = []

        for level in available_levels:
            if level == 0.0:
                # Sample from main buffer
                obs, actions, rewards, next_obs, dones = self.buffer.sample(
                    self.batch_size
                )
            else:
                # Sample from GAR buffer
                batch = self.gar_buffer.sample(level, self.batch_size)
                if batch is None:
                    continue
                obs, actions, rewards, next_obs, dones = batch

            # Compute loss and gradients
            loss = self._compute_dqn_loss(obs, actions, rewards, next_obs, dones)
            losses.append(loss.item())

            # Record loss per variation
            if self.gar_stats is not None:
                self.gar_stats.record_variation_loss(level, loss.item())

            # Compute gradients without updating
            grad_dict = compute_gradient_dict(loss, self.q_network)
            grads_list.append(grad_dict)

        if len(grads_list) == 0:
            # Fall back to standard training if no gradients available
            return self._train_step_standard()

        # Compute gradient agreement
        if len(grads_list) > 1:
            agreement = compute_mean_agreement(grads_list)
            if self.gar_stats is not None:
                self.gar_stats.record_agreement(agreement)

        # Compute agreed gradients
        agreed_grads = compute_gradient_agreement(grads_list)

        # Apply gradients
        self.optimizer.zero_grad()
        apply_gradients(self.q_network, agreed_grads, max_grad_norm=10.0)
        self.optimizer.step()

        return np.mean(losses)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _collect_gar_experiences(self, num_steps: int = 10):
        """
        Collect experiences from shifted environments for GAR.

        This runs the current policy on shifted environments and stores
        the experiences in the GAR buffer.

        Args:
            num_steps: Number of steps to collect per shifted environment
        """
        if not self.use_gar or not self.gar_envs:
            return

        for level, env in self.gar_envs.items():
            # Initialize or get current state for this env
            if not hasattr(self, '_gar_env_states'):
                self._gar_env_states = {}
            if level not in self._gar_env_states:
                obs, _ = env.reset()
                self._gar_env_states[level] = obs

            obs = self._gar_env_states[level]

            for _ in range(num_steps):
                # Use current policy to select action
                action = self.select_action(obs, evaluate=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store in GAR buffer
                self.gar_buffer.add(level, obs, action, reward, next_obs, done)

                if done:
                    obs, _ = env.reset()
                else:
                    obs = next_obs

            self._gar_env_states[level] = obs
    
    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1000,
        eval_interval: int = 10000,
        eval_episodes: int = 10,
        callback=None,
    ) -> Dict[str, list]:
        history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "epsilons": [],
            "eval_rewards": [],
        }

        # Add GAR-specific tracking if enabled
        if self.use_gar:
            history["gar_agreements"] = []

        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        # GAR collection frequency - much less frequent for image-based envs (expensive)
        if self.use_cnn:
            gar_collect_freq = 1000  # CNN envs are slow - collect less often
            gar_steps_per_collect = 20  # Fewer steps per collection
        else:
            gar_collect_freq = 100
            gar_steps_per_collect = 50

        for step in range(total_timesteps):
            self.total_steps += 1

            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(obs, action, reward, next_obs, done)

            # Also add to GAR buffer at level 0.0 (the base environment)
            if self.use_gar and self.gar_buffer is not None:
                self.gar_buffer.add(0.0, obs, action, reward, next_obs, done)

            # Periodically collect experiences from shifted environments
            if self.use_gar and self.total_steps % gar_collect_freq == 0:
                self._collect_gar_experiences(num_steps=gar_steps_per_collect)
            
            episode_reward += reward
            episode_length += 1
            
            if self.total_steps >= self.learning_starts and self.total_steps % self.train_freq == 0:
                loss = self.train_step()
                if loss is not None:
                    history["losses"].append(loss)
            
            if self.total_steps % self.target_update_freq == 0:
                self.update_target_network()
            
            if done:
                history["episode_rewards"].append(episode_reward)
                history["episode_lengths"].append(episode_length)
                history["epsilons"].append(self.get_epsilon())
                self.episodes += 1
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            if self.total_steps % log_interval == 0:
                recent_rewards = history["episode_rewards"][-10:] if history["episode_rewards"] else [0]
                recent_loss = np.mean(history["losses"][-100:]) if history["losses"] else 0
                log_msg = (f"Step {self.total_steps}/{total_timesteps} | "
                           f"Episodes: {self.episodes} | "
                           f"Reward: {np.mean(recent_rewards):.2f} | "
                           f"Loss: {recent_loss:.4f} | "
                           f"Epsilon: {self.get_epsilon():.3f}")

                # Add GAR statistics if enabled
                if self.use_gar and self.gar_stats is not None:
                    gar_stats = self.gar_stats.get_stats()
                    if "mean_agreement" in gar_stats:
                        log_msg += f" | GAR Agree: {gar_stats['mean_agreement']:.3f}"
                        history["gar_agreements"].append(gar_stats["mean_agreement"])

                print(log_msg)
            
            if eval_interval > 0 and self.total_steps % eval_interval == 0:
                eval_reward = self.evaluate(eval_episodes)
                history["eval_rewards"].append(eval_reward)
                print(f"  Eval reward: {eval_reward:.2f}")
                obs, _ = self.env.reset()
            
            if callback is not None:
                callback(self, step)
        
        return history
    
    def evaluate(self, num_episodes: int = 10) -> float:
        """
        Evaluate the agent's performance over multiple episodes.

        Handles both standard environments (where reset() starts a new episode)
        and auto-resetting environments like Procgen (where reset() may be ignored
        and done signals indicate the start of a new episode).

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Mean reward across all episodes
        """
        rewards = []

        # Detect if this is an auto-resetting environment (like Procgen)
        is_auto_reset = self._is_auto_reset_env()

        if is_auto_reset:
            # For auto-resetting envs, we run continuously and count done signals
            obs, _ = self.env.reset()
            episode_reward = 0

            while len(rewards) < num_episodes:
                action = self.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    rewards.append(episode_reward)
                    episode_reward = 0
                    # obs is already the first observation of the new episode
        else:
            # Standard evaluation loop for non-auto-resetting envs
            for _ in range(num_episodes):
                obs, _ = self.env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action = self.select_action(obs, evaluate=True)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    episode_reward += reward

                rewards.append(episode_reward)

        return np.mean(rewards)

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
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[int, None]:
        action = self.select_action(obs, evaluate=deterministic)
        return action, None
    
    def save(self, path: str):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.episodes = checkpoint["episodes"]


if __name__ == "__main__":
    print("Testing DQN...")
    
    env = gym.make("CartPole-v1")
    agent = DQN(
        env=env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=500,
        batch_size=32,
        target_update_freq=2000,
        weight_decay=1e-4,
    )
    
    print(f"Device: {agent.device}")
    
    history = agent.learn(
        total_timesteps=5000,
        log_interval=1000,
        eval_interval=2500,
    )
    
    final_reward = agent.evaluate(num_episodes=10)
    print(f"Final mean reward: {final_reward:.2f}")
    
    env.close()
    print("DQN test complete!")