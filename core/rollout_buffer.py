"""
Rollout Buffer for PPO

Stores complete trajectories for on-policy learning.
Unlike DQN's replay buffer, this stores sequential data and is cleared after each update.

Also computes advantages using GAE (Generalized Advantage Estimation).
"""

import numpy as np
import torch
from typing import Tuple, Generator, Optional


class RolloutBuffer:
    """
    Rollout buffer for PPO.
    
    Stores trajectories and computes advantages.
    
    Args:
        buffer_size: Number of steps to collect before update
        obs_shape: Shape of observations
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        device: Device for tensors
    """
    
    def __init__(
        self,
        buffer_size: int = 2048,
        obs_shape: Tuple = (4,),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Storage arrays
        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed after rollout
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        # Pointer
        self.idx = 0
        self.full = False
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Add a single timestep to the buffer."""
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.values[self.idx] = value
        self.log_probs[self.idx] = log_prob
        self.dones[self.idx] = float(done)
        
        self.idx += 1
        if self.idx >= self.buffer_size:
            self.full = True
    
    def compute_advantages(self, last_value: float, last_done: bool):
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        
        GAE provides a good bias-variance tradeoff for advantage estimation.
        
        Args:
            last_value: Value estimate of the last state
            last_done: Whether the last state was terminal
        """
        last_gae = 0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            
            # TD error: r + gamma * V(s') - V(s)
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            
            # GAE: sum of discounted TD errors
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        
        # Returns = advantages + values
        self.returns = self.advantages + self.values
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate minibatches for PPO update.
        
        Args:
            batch_size: Size of each minibatch
            shuffle: Whether to shuffle data
            
        Yields:
            Tuples of (obs, actions, old_log_probs, advantages, returns)
        """
        indices = np.arange(self.buffer_size)
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                torch.from_numpy(self.observations[batch_indices]).to(self.device),
                torch.from_numpy(self.actions[batch_indices]).to(self.device),
                torch.from_numpy(self.log_probs[batch_indices]).to(self.device),
                torch.from_numpy(self.advantages[batch_indices]).to(self.device),
                torch.from_numpy(self.returns[batch_indices]).to(self.device),
            )
    
    def get_all(self) -> Tuple[torch.Tensor, ...]:
        """Get all data as tensors."""
        return (
            torch.from_numpy(self.observations).to(self.device),
            torch.from_numpy(self.actions).to(self.device),
            torch.from_numpy(self.log_probs).to(self.device),
            torch.from_numpy(self.advantages).to(self.device),
            torch.from_numpy(self.returns).to(self.device),
        )
    
    def reset(self):
        """Clear the buffer for next rollout."""
        self.idx = 0
        self.full = False
    
    def is_full(self) -> bool:
        """Check if buffer is full and ready for update."""
        return self.full


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Rollout Buffer...")
    
    buffer = RolloutBuffer(buffer_size=64, obs_shape=(4,))
    
    # Simulate a rollout
    for i in range(64):
        obs = np.random.randn(4).astype(np.float32)
        action = np.random.randint(2)
        reward = 1.0  # Constant reward for testing
        value = np.random.randn()
        log_prob = np.random.randn()
        done = i == 63  # Last step is terminal
        
        buffer.add(obs, action, reward, value, log_prob, done)
    
    print(f"Buffer full: {buffer.is_full()}")
    assert buffer.is_full()
    
    # Compute advantages
    last_value = 0.0  # Terminal state value
    buffer.compute_advantages(last_value, last_done=True)
    
    print(f"Advantages shape: {buffer.advantages.shape}")
    print(f"Advantages mean: {buffer.advantages.mean():.3f}")
    print(f"Advantages std: {buffer.advantages.std():.3f}")
    print(f"Returns shape: {buffer.returns.shape}")
    
    # Test batch generation
    print("\nTesting batch generation...")
    batch_count = 0
    for batch in buffer.get_batches(batch_size=16):
        obs, actions, log_probs, advantages, returns = batch
        batch_count += 1
        print(f"  Batch {batch_count}: obs={obs.shape}, actions={actions.shape}")
    
    assert batch_count == 4  # 64 / 16 = 4 batches
    
    # Test reset
    buffer.reset()
    print(f"\nAfter reset - Buffer full: {buffer.is_full()}")
    assert not buffer.is_full()
    
    print("\nAll rollout buffer tests passed!")