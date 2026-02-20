"""
PPO (Proximal Policy Optimization) Algorithm
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

from core.networks import PPONetwork
from core.rollout_buffer import RolloutBuffer
from core.gar import (
    GARStats,
    compute_gradient_dict,
    compute_gradient_agreement,
    compute_mean_agreement,
    apply_gradients,
)


class PPO:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        weight_decay: float = 0.0,
        hidden_dims: list = [64, 64],
        device: str = "auto",
        # GAR parameters
        use_gar: bool = False,
        gar_variation_levels: list = [0.0, 0.25, 0.5],
        env_class: type = None,
        env_kwargs: dict = None,
        env_wrapper: callable = None,  # Wrapper to apply to GAR envs (e.g., FrameStack)
    ):
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.n_actions = env.action_space.n

        # Detect vectorized environment
        self.num_envs = getattr(env, 'num_envs', 1)
        self.is_vectorized = self.num_envs > 1

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

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

        self.policy = PPONetwork(
            obs_dim=obs_dim,
            action_dim=self.n_actions,
            hidden_dims=hidden_dims,
            use_cnn=self.use_cnn,
        ).to(self.device)

        # AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Buffer size accounts for vectorized envs: n_steps * num_envs transitions
        buffer_size = n_steps * self.num_envs
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            obs_shape=self.obs_shape,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=self.device,
        )

        self.total_steps = 0
        self.episodes = 0
        self.updates = 0

        # GAR setup
        self.use_gar = use_gar
        self.gar_variation_levels = gar_variation_levels
        self.env_class = env_class
        self.env_kwargs = env_kwargs or {}
        self.env_wrapper = env_wrapper  # e.g., FrameStack for Procgen/Starpilot

        if self.use_gar:
            self.gar_stats = GARStats()

            # Create shifted environments and their rollout buffers
            self.gar_envs = {}
            self.gar_buffers = {}
            if self.env_class is not None:
                for level in self.gar_variation_levels:
                    if level != 0.0:  # Don't duplicate the main env
                        env = self.env_class(variation_level=level, **self.env_kwargs)
                        # Apply wrapper if provided (e.g., FrameStack for Starpilot)
                        if self.env_wrapper is not None:
                            env = self.env_wrapper(env)
                        self.gar_envs[level] = env
                        self.gar_buffers[level] = RolloutBuffer(
                            buffer_size=n_steps,
                            obs_shape=self.obs_shape,
                            gamma=gamma,
                            gae_lambda=gae_lambda,
                            device=self.device,
                        )
        else:
            self.gar_stats = None
            self.gar_envs = {}
            self.gar_buffers = {}
    
    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> Tuple[int, float, float]:
        """Select action for a single observation. Returns scalars."""
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        # Apply observation normalization if needed
        if self.obs_normalize > 1.0:
            obs_tensor = obs_tensor / self.obs_normalize
        action, log_prob, value = self.policy.get_action(obs_tensor, deterministic=evaluate)
        return action, log_prob, value

    def select_actions_batch(self, obs: np.ndarray, evaluate: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for a batch of observations. Returns numpy arrays."""
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        # Apply observation normalization if needed
        if self.obs_normalize > 1.0:
            obs_tensor = obs_tensor / self.obs_normalize

        with torch.no_grad():
            action_probs, values = self.policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=action_probs)

            if evaluate:
                actions = action_probs.argmax(dim=-1)
            else:
                actions = dist.sample()

            log_probs = dist.log_prob(actions)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.squeeze(-1).cpu().numpy()
    
    def _compute_ppo_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PPO loss components for a batch."""
        new_log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        if self.clip_range_vf is not None:
            old_values = returns - advantages
            values_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_range_vf, self.clip_range_vf
            )
            value_loss1 = F.mse_loss(values, returns)
            value_loss2 = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = F.mse_loss(values, returns)

        entropy_loss = -entropy.mean()

        return policy_loss, value_loss, entropy_loss

    def update(self) -> Dict[str, float]:
        if self.use_gar:
            return self._update_gar()
        else:
            return self._update_standard()

    def _update_standard(self) -> Dict[str, float]:
        """Standard PPO update without GAR."""
        last_obs = self.buffer.observations[self.buffer.idx - 1] if self.buffer.idx > 0 else self.buffer.observations[-1]
        last_obs_tensor = torch.from_numpy(last_obs).float().unsqueeze(0).to(self.device)
        if self.obs_normalize > 1.0:
            last_obs_tensor = last_obs_tensor / self.obs_normalize

        with torch.no_grad():
            _, last_value = self.policy(last_obs_tensor)
            last_value = last_value.item()

        self.buffer.compute_advantages(last_value, last_done=False)

        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs, actions, old_log_probs, batch_advantages, returns = batch
                # Normalize observations if needed
                if self.obs_normalize > 1.0:
                    obs = obs / self.obs_normalize

                policy_loss, value_loss, entropy_loss = self._compute_ppo_loss(
                    obs, actions, old_log_probs, batch_advantages, returns
                )

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        self.buffer.reset()
        self.updates += 1

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def _update_gar(self) -> Dict[str, float]:
        """
        GAR-enabled PPO update: compute gradients from multiple variation levels
        and only update in directions where gradients agree.
        """
        # Prepare main buffer
        last_obs = self.buffer.observations[self.buffer.idx - 1] if self.buffer.idx > 0 else self.buffer.observations[-1]
        last_obs_tensor = torch.from_numpy(last_obs).float().unsqueeze(0).to(self.device)
        if self.obs_normalize > 1.0:
            last_obs_tensor = last_obs_tensor / self.obs_normalize

        with torch.no_grad():
            _, last_value = self.policy(last_obs_tensor)
            last_value = last_value.item()

        self.buffer.compute_advantages(last_value, last_done=False)
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages

        # Prepare GAR buffers
        gar_buffers_ready = {}
        for level, buffer in self.gar_buffers.items():
            if buffer.is_full():
                # Compute advantages for this buffer
                gar_last_obs = buffer.observations[buffer.idx - 1] if buffer.idx > 0 else buffer.observations[-1]
                gar_last_obs_tensor = torch.from_numpy(gar_last_obs).float().unsqueeze(0).to(self.device)
                if self.obs_normalize > 1.0:
                    gar_last_obs_tensor = gar_last_obs_tensor / self.obs_normalize

                with torch.no_grad():
                    _, gar_last_value = self.policy(gar_last_obs_tensor)
                    gar_last_value = gar_last_value.item()

                buffer.compute_advantages(gar_last_value, last_done=False)
                adv = buffer.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                buffer.advantages = adv
                gar_buffers_ready[level] = buffer

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        total_agreement = 0
        agreement_count = 0

        for epoch in range(self.n_epochs):
            # Get batches from main buffer
            main_batches = list(self.buffer.get_batches(self.batch_size))

            # Get batches from GAR buffers
            gar_batch_lists = {
                level: list(buffer.get_batches(self.batch_size))
                for level, buffer in gar_buffers_ready.items()
            }

            for batch_idx, main_batch in enumerate(main_batches):
                obs, actions, old_log_probs, batch_advantages, returns = main_batch
                # Normalize observations if needed
                if self.obs_normalize > 1.0:
                    obs = obs / self.obs_normalize

                # Compute loss and gradients from main buffer
                policy_loss, value_loss, entropy_loss = self._compute_ppo_loss(
                    obs, actions, old_log_probs, batch_advantages, returns
                )
                main_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                grads_list = [compute_gradient_dict(main_loss, self.policy)]

                # Compute gradients from GAR buffers
                for level, gar_batches in gar_batch_lists.items():
                    if batch_idx < len(gar_batches):
                        gar_batch = gar_batches[batch_idx]
                        gar_obs, gar_actions, gar_old_log_probs, gar_advantages, gar_returns = gar_batch
                        # Normalize GAR observations if needed
                        if self.obs_normalize > 1.0:
                            gar_obs = gar_obs / self.obs_normalize

                        gar_policy_loss, gar_value_loss, gar_entropy_loss = self._compute_ppo_loss(
                            gar_obs, gar_actions, gar_old_log_probs, gar_advantages, gar_returns
                        )
                        gar_loss = gar_policy_loss + self.vf_coef * gar_value_loss + self.ent_coef * gar_entropy_loss

                        # Record loss per variation
                        if self.gar_stats is not None:
                            self.gar_stats.record_variation_loss(level, gar_loss.item())

                        grads_list.append(compute_gradient_dict(gar_loss, self.policy))

                # Compute gradient agreement
                if len(grads_list) > 1:
                    agreement = compute_mean_agreement(grads_list)
                    total_agreement += agreement
                    agreement_count += 1

                    if self.gar_stats is not None:
                        self.gar_stats.record_agreement(agreement)

                # Compute agreed gradients
                agreed_grads = compute_gradient_agreement(grads_list)

                # Apply gradients
                self.optimizer.zero_grad()
                apply_gradients(self.policy, agreed_grads, max_grad_norm=self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        # Reset buffers
        self.buffer.reset()
        for buffer in self.gar_buffers.values():
            buffer.reset()
        self.updates += 1

        result = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

        if agreement_count > 0:
            result["gar_agreement"] = total_agreement / agreement_count

        return result
    
    def collect_rollout(self) -> Tuple[float, int]:
        """Collect rollout from environment(s). Supports both single and vectorized envs."""
        if self.is_vectorized:
            return self._collect_rollout_vectorized()
        else:
            return self._collect_rollout_single()

    def _collect_rollout_single(self) -> Tuple[float, int]:
        """Collect rollout from a single environment."""
        obs, _ = self.env.reset()
        episode_rewards = []
        current_reward = 0

        # Initialize GAR environment states if needed
        if self.use_gar and not hasattr(self, '_gar_env_obs'):
            self._gar_env_obs = {}
            for level, env in self.gar_envs.items():
                self._gar_env_obs[level], _ = env.reset()

        for step in range(self.n_steps):
            action, log_prob, value = self.select_action(obs)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(obs, action, reward, value, log_prob, done)

            # Collect experiences from GAR environments in parallel
            if self.use_gar:
                for level, env in self.gar_envs.items():
                    gar_obs = self._gar_env_obs[level]
                    gar_action, gar_log_prob, gar_value = self.select_action(gar_obs)

                    gar_next_obs, gar_reward, gar_term, gar_trunc, _ = env.step(gar_action)
                    gar_done = gar_term or gar_trunc

                    self.gar_buffers[level].add(
                        gar_obs, gar_action, gar_reward, gar_value, gar_log_prob, gar_done
                    )

                    if gar_done:
                        self._gar_env_obs[level], _ = env.reset()
                    else:
                        self._gar_env_obs[level] = gar_next_obs

            current_reward += reward
            self.total_steps += 1

            if done:
                episode_rewards.append(current_reward)
                current_reward = 0
                self.episodes += 1
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        return mean_reward, len(episode_rewards)

    def _collect_rollout_vectorized(self) -> Tuple[float, int]:
        """Collect rollout from vectorized environments (faster with multiple envs)."""
        obs, _ = self.env.reset()  # Shape: [num_envs, *obs_shape]
        episode_rewards = []
        current_rewards = np.zeros(self.num_envs)

        for step in range(self.n_steps):
            # Get actions for all environments at once
            actions, log_probs, values = self.select_actions_batch(obs)

            # Step all environments
            next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            dones = terminateds | truncateds

            # Store transitions for each environment
            for env_idx in range(self.num_envs):
                self.buffer.add(
                    obs[env_idx],
                    actions[env_idx],
                    rewards[env_idx],
                    values[env_idx],
                    log_probs[env_idx],
                    dones[env_idx]
                )

            # Track rewards and episodes
            current_rewards += rewards
            self.total_steps += self.num_envs

            # Handle episode completions (vec env auto-resets)
            for env_idx in range(self.num_envs):
                if dones[env_idx]:
                    episode_rewards.append(current_rewards[env_idx])
                    current_rewards[env_idx] = 0
                    self.episodes += 1

            obs = next_obs

        mean_reward = np.mean(episode_rewards) if episode_rewards else 0
        return mean_reward, len(episode_rewards)
    
    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        eval_interval: int = 10,
        eval_episodes: int = 10,
        callback=None,
    ) -> Dict[str, list]:
        history = {
            "episode_rewards": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "eval_rewards": [],
        }

        # Add GAR-specific tracking if enabled
        if self.use_gar:
            history["gar_agreements"] = []

        num_updates = total_timesteps // self.n_steps

        for update in range(num_updates):
            mean_reward, num_episodes = self.collect_rollout()
            history["episode_rewards"].append(mean_reward)

            losses = self.update()
            history["policy_losses"].append(losses["policy_loss"])
            history["value_losses"].append(losses["value_loss"])
            history["entropies"].append(losses["entropy"])

            # Track GAR agreement if available
            if self.use_gar and "gar_agreement" in losses:
                history["gar_agreements"].append(losses["gar_agreement"])

            if (update + 1) % log_interval == 0:
                log_msg = (f"Update {update + 1}/{num_updates} | "
                           f"Steps: {self.total_steps} | "
                           f"Episodes: {self.episodes} | "
                           f"Reward: {mean_reward:.2f} | "
                           f"Policy Loss: {losses['policy_loss']:.4f} | "
                           f"Value Loss: {losses['value_loss']:.4f}")

                # Add GAR statistics if enabled
                if self.use_gar and "gar_agreement" in losses:
                    log_msg += f" | GAR Agree: {losses['gar_agreement']:.3f}"

                print(log_msg)

            if eval_interval > 0 and (update + 1) % eval_interval == 0:
                eval_reward = self.evaluate(eval_episodes)
                history["eval_rewards"].append(eval_reward)
                print(f"  Eval reward: {eval_reward:.2f}")

            if callback is not None:
                callback(self, update)

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
                action, _, _ = self.select_action(obs, evaluate=True)
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
                    action, _, _ = self.select_action(obs, evaluate=True)
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
        action, _, _ = self.select_action(obs, evaluate=deterministic)
        return action, None
    
    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "updates": self.updates,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.episodes = checkpoint["episodes"]
        self.updates = checkpoint["updates"]


if __name__ == "__main__":
    print("Testing PPO...")
    
    env = gym.make("CartPole-v1")
    agent = PPO(
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        n_epochs=4,
        batch_size=64,
        weight_decay=1e-4,
    )
    
    print(f"Device: {agent.device}")
    
    history = agent.learn(
        total_timesteps=5000,
        log_interval=5,
        eval_interval=10,
    )
    
    final_reward = agent.evaluate(num_episodes=10)
    print(f"Final mean reward: {final_reward:.2f}")
    
    env.close()
    print("PPO test complete!")