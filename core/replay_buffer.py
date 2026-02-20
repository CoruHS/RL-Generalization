"""
Replay Buffer for DQN

Stores transitions (s, a, r, s', done) and samples random batches for training.
This breaks correlation between consecutive samples.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from collections import deque
import random


class ReplayBuffer:
    """
    Simple replay buffer using deque.
    
    Stores transitions and samples random batches.
    
    Args:
        capacity: Maximum number of transitions to store
        obs_shape: Shape of observations
        device: Device to put tensors on when sampling
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        obs_shape: Tuple = (4,),
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device
        
        # Storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Pointers
        self.idx = 0
        self.size = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add a single transition to the buffer."""
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.dones[self.idx] = float(done)
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.
        
        Returns:
            observations: (batch_size, *obs_shape)
            actions: (batch_size,)
            rewards: (batch_size,)
            next_observations: (batch_size, *obs_shape)
            dones: (batch_size,)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.from_numpy(self.observations[indices]).to(self.device),
            torch.from_numpy(self.actions[indices]).to(self.device),
            torch.from_numpy(self.rewards[indices]).to(self.device),
            torch.from_numpy(self.next_observations[indices]).to(self.device),
            torch.from_numpy(self.dones[indices]).to(self.device),
        )
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return self.size >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions based on TD-error priority.
    Higher error = more likely to be sampled.
    
    Args:
        capacity: Maximum number of transitions
        obs_shape: Shape of observations
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        beta: Importance sampling exponent (starts low, anneals to 1)
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        obs_shape: Tuple = (4,),
        alpha: float = 0.6,
        beta: float = 0.4,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        # Storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Priorities
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        # Pointers
        self.idx = 0
        self.size = 0
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add transition with max priority."""
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.dones[self.idx] = float(done)
        self.priorities[self.idx] = self.max_priority
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample based on priorities."""
        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        return (
            torch.from_numpy(self.observations[indices]).to(self.device),
            torch.from_numpy(self.actions[indices]).to(self.device),
            torch.from_numpy(self.rewards[indices]).to(self.device),
            torch.from_numpy(self.next_observations[indices]).to(self.device),
            torch.from_numpy(self.dones[indices]).to(self.device),
            torch.from_numpy(weights.astype(np.float32)).to(self.device),
            indices,
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6  # Small constant to avoid zero
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing Replay Buffer...")
    
    # Test basic buffer
    buffer = ReplayBuffer(capacity=1000, obs_shape=(4,))
    
    # Add some transitions
    for i in range(100):
        obs = np.random.randn(4).astype(np.float32)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_obs = np.random.randn(4).astype(np.float32)
        done = np.random.random() < 0.1
        
        buffer.add(obs, action, reward, next_obs, done)
    
    print(f"Buffer size: {len(buffer)}")
    assert len(buffer) == 100
    
    # Sample a batch
    obs, actions, rewards, next_obs, dones = buffer.sample(batch_size=32)
    print(f"Sampled batch shapes:")
    print(f"  obs: {obs.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  rewards: {rewards.shape}")
    print(f"  next_obs: {next_obs.shape}")
    print(f"  dones: {dones.shape}")
    
    assert obs.shape == (32, 4)
    assert actions.shape == (32,)
    
    # Test prioritized buffer
    print("\nTesting Prioritized Replay Buffer...")
    pri_buffer = PrioritizedReplayBuffer(capacity=1000, obs_shape=(4,))
    
    for i in range(100):
        obs = np.random.randn(4).astype(np.float32)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_obs = np.random.randn(4).astype(np.float32)
        done = np.random.random() < 0.1
        
        pri_buffer.add(obs, action, reward, next_obs, done)
    
    result = pri_buffer.sample(batch_size=32)
    obs, actions, rewards, next_obs, dones, weights, indices = result
    print(f"Prioritized sample weights shape: {weights.shape}")
    print(f"Indices shape: {len(indices)}")
    
    # Update priorities
    td_errors = np.random.randn(32)
    pri_buffer.update_priorities(indices, td_errors)
    print(f"Updated priorities successfully")
    
    print("\nAll replay buffer tests passed!")