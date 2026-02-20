"""
Gradient Agreement Regularization (GAR) for RL Generalization

This module provides utilities for computing gradient agreement across multiple
environment variations. The key idea is to only update in directions where
gradients agree across variations, forcing the agent to learn features that
generalize.

The gradient agreement algorithm:
1. Sample experiences from multiple environment variations
2. Compute gradients for each variation
3. Weight gradients by their agreement (cosine similarity with mean)
4. Only update in directions where gradients agree

This encourages learning features that are useful across all variations,
rather than overfitting to specific environment characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque


def compute_gradient_agreement(grads_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Compute agreed gradients from a list of gradient dictionaries.

    For each parameter, we:
    1. Stack gradients from all variations
    2. Compute the mean gradient direction
    3. Weight each gradient by its cosine similarity with the mean
    4. Return the weighted sum

    This ensures that updates only happen in directions where gradients agree.

    Args:
        grads_list: List of gradient dictionaries, one per environment variation.
                   Each dict maps parameter names to gradient tensors.

    Returns:
        Dictionary of agreed gradients with the same keys as input.
    """
    if len(grads_list) == 0:
        return {}

    if len(grads_list) == 1:
        return grads_list[0]

    agreed_grads = {}

    for key in grads_list[0]:
        # Stack gradients for this parameter: shape (num_variations, *param_shape)
        stacked = torch.stack([g[key] for g in grads_list])

        # Compute mean gradient
        mean_grad = stacked.mean(dim=0)

        # Compute cosine similarity of each gradient with the mean
        similarities = []
        for g in stacked:
            # Flatten for cosine similarity computation
            g_flat = g.flatten()
            mean_flat = mean_grad.flatten()

            # Handle zero gradients
            g_norm = g_flat.norm()
            mean_norm = mean_flat.norm()

            if g_norm < 1e-8 or mean_norm < 1e-8:
                cos_sim = torch.tensor(0.0, device=g.device)
            else:
                cos_sim = F.cosine_similarity(
                    g_flat.unsqueeze(0),
                    mean_flat.unsqueeze(0),
                    dim=1
                ).squeeze()

            # Only count positive agreement (gradients pointing in same direction)
            similarities.append(cos_sim.clamp(min=0))

        # Convert to weights
        weights = torch.stack(similarities)
        weight_sum = weights.sum() + 1e-8
        weights = weights / weight_sum

        # Compute weighted sum of gradients
        # Reshape weights for broadcasting: (num_variations, 1, 1, ...)
        weight_shape = [-1] + [1] * len(stacked.shape[1:])
        weights_reshaped = weights.view(*weight_shape)

        agreed_grads[key] = (stacked * weights_reshaped).sum(dim=0)

    return agreed_grads


def compute_gradient_dict(
    loss: torch.Tensor,
    model: nn.Module,
    create_graph: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute gradients of a loss with respect to model parameters.

    Args:
        loss: Scalar loss tensor
        model: Neural network model
        create_graph: Whether to create computation graph (for higher-order derivatives)

    Returns:
        Dictionary mapping parameter names to gradient tensors
    """
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )

    grad_dict = {}
    for (name, param), grad in zip(model.named_parameters(), grads):
        if grad is not None:
            grad_dict[name] = grad
        else:
            # Parameter not used in this loss - use zero gradient
            grad_dict[name] = torch.zeros_like(param)

    return grad_dict


def apply_gradients(
    model: nn.Module,
    grad_dict: Dict[str, torch.Tensor],
    max_grad_norm: Optional[float] = None,
) -> None:
    """
    Apply gradients to model parameters.

    This sets the .grad attribute of each parameter, which can then be
    used by an optimizer.

    Args:
        model: Neural network model
        grad_dict: Dictionary mapping parameter names to gradients
        max_grad_norm: Optional gradient clipping threshold
    """
    # Set gradients on parameters
    for name, param in model.named_parameters():
        if name in grad_dict:
            param.grad = grad_dict[name].clone()
        else:
            param.grad = torch.zeros_like(param)

    # Apply gradient clipping if specified
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


class GARBuffer:
    """
    Buffer for storing experiences from multiple environment variations.

    This maintains separate mini-buffers for each variation level, allowing
    the agent to sample experiences from different variations during training.

    Args:
        capacity_per_level: Maximum experiences to store per variation level
        obs_shape: Shape of observations
        device: Device for tensors
    """

    def __init__(
        self,
        capacity_per_level: int = 1000,
        obs_shape: Tuple = (4,),
        device: str = "cpu",
    ):
        self.capacity_per_level = capacity_per_level
        self.obs_shape = obs_shape
        self.device = device

        # Storage for each variation level
        self.buffers: Dict[float, Dict[str, deque]] = {}

    def _ensure_level_exists(self, level: float):
        """Create buffer storage for a variation level if it doesn't exist."""
        if level not in self.buffers:
            self.buffers[level] = {
                "observations": deque(maxlen=self.capacity_per_level),
                "actions": deque(maxlen=self.capacity_per_level),
                "rewards": deque(maxlen=self.capacity_per_level),
                "next_observations": deque(maxlen=self.capacity_per_level),
                "dones": deque(maxlen=self.capacity_per_level),
            }

    def add(
        self,
        level: float,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer for a specific variation level."""
        self._ensure_level_exists(level)

        self.buffers[level]["observations"].append(obs.copy())
        self.buffers[level]["actions"].append(action)
        self.buffers[level]["rewards"].append(reward)
        self.buffers[level]["next_observations"].append(next_obs.copy())
        self.buffers[level]["dones"].append(float(done))

    def sample(
        self,
        level: float,
        batch_size: int,
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Sample a batch from a specific variation level.

        Returns None if not enough samples are available.
        """
        if level not in self.buffers:
            return None

        buffer = self.buffers[level]
        if len(buffer["observations"]) < batch_size:
            return None

        # Sample random indices
        indices = np.random.randint(0, len(buffer["observations"]), size=batch_size)

        # Convert to arrays and sample
        obs_array = np.array(buffer["observations"])
        actions_array = np.array(buffer["actions"])
        rewards_array = np.array(buffer["rewards"])
        next_obs_array = np.array(buffer["next_observations"])
        dones_array = np.array(buffer["dones"])

        return (
            torch.from_numpy(obs_array[indices]).float().to(self.device),
            torch.from_numpy(actions_array[indices]).long().to(self.device),
            torch.from_numpy(rewards_array[indices]).float().to(self.device),
            torch.from_numpy(next_obs_array[indices]).float().to(self.device),
            torch.from_numpy(dones_array[indices]).float().to(self.device),
        )

    def get_available_levels(self, min_samples: int = 32) -> List[float]:
        """Get list of variation levels with enough samples."""
        return [
            level for level, buffer in self.buffers.items()
            if len(buffer["observations"]) >= min_samples
        ]

    def clear(self):
        """Clear all buffers."""
        self.buffers.clear()

    def __len__(self) -> int:
        """Total number of transitions across all levels."""
        return sum(
            len(buffer["observations"])
            for buffer in self.buffers.values()
        )


class GARStats:
    """
    Track statistics for Gradient Agreement Regularization.

    Monitors gradient agreement, effective learning, and per-variation losses.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Track various statistics
        self.agreement_scores: deque = deque(maxlen=window_size)
        self.variation_losses: Dict[float, deque] = {}
        self.gradient_norms: deque = deque(maxlen=window_size)
        self.update_magnitudes: deque = deque(maxlen=window_size)

    def record_agreement(self, score: float):
        """Record a gradient agreement score."""
        self.agreement_scores.append(score)

    def record_variation_loss(self, level: float, loss: float):
        """Record loss for a specific variation level."""
        if level not in self.variation_losses:
            self.variation_losses[level] = deque(maxlen=self.window_size)
        self.variation_losses[level].append(loss)

    def record_gradient_norm(self, norm: float):
        """Record gradient norm."""
        self.gradient_norms.append(norm)

    def record_update_magnitude(self, magnitude: float):
        """Record update magnitude."""
        self.update_magnitudes.append(magnitude)

    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        stats = {}

        if self.agreement_scores:
            stats["mean_agreement"] = np.mean(list(self.agreement_scores))
            stats["min_agreement"] = np.min(list(self.agreement_scores))

        if self.gradient_norms:
            stats["mean_grad_norm"] = np.mean(list(self.gradient_norms))

        if self.update_magnitudes:
            stats["mean_update_mag"] = np.mean(list(self.update_magnitudes))

        for level, losses in self.variation_losses.items():
            if losses:
                stats[f"loss_level_{level:.2f}"] = np.mean(list(losses))

        return stats


def compute_mean_agreement(grads_list: List[Dict[str, torch.Tensor]]) -> float:
    """
    Compute the mean pairwise cosine similarity across all gradients.

    This is a measure of how much the gradients from different variations agree.
    Higher values indicate better generalization potential.

    Args:
        grads_list: List of gradient dictionaries

    Returns:
        Mean cosine similarity (0 to 1)
    """
    if len(grads_list) < 2:
        return 1.0

    # Flatten all gradients into vectors
    flat_grads = []
    for grad_dict in grads_list:
        flat = torch.cat([g.flatten() for g in grad_dict.values()])
        flat_grads.append(flat)

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(len(flat_grads)):
        for j in range(i + 1, len(flat_grads)):
            g_i = flat_grads[i]
            g_j = flat_grads[j]

            norm_i = g_i.norm()
            norm_j = g_j.norm()

            if norm_i < 1e-8 or norm_j < 1e-8:
                sim = 0.0
            else:
                sim = F.cosine_similarity(
                    g_i.unsqueeze(0),
                    g_j.unsqueeze(0),
                    dim=1
                ).item()

            similarities.append(max(0, sim))  # Only count positive agreement

    return np.mean(similarities) if similarities else 1.0


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing GAR module...")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )

    print("\nTest 1: Compute gradient dict")
    x = torch.randn(32, 4)
    y = torch.randn(32, 2)
    pred = model(x)
    loss = F.mse_loss(pred, y)

    grad_dict = compute_gradient_dict(loss, model)
    print(f"  Number of gradient entries: {len(grad_dict)}")
    for name, grad in grad_dict.items():
        print(f"    {name}: {grad.shape}")

    print("\nTest 2: Gradient agreement with similar targets")
    grads_list = []
    for _ in range(3):
        x = torch.randn(32, 4)
        y = torch.ones(32, 2)  # Same target direction
        pred = model(x)
        loss = F.mse_loss(pred, y)
        grads_list.append(compute_gradient_dict(loss, model))

    agreement = compute_mean_agreement(grads_list)
    print(f"  Agreement score: {agreement:.4f}")

    agreed_grads = compute_gradient_agreement(grads_list)
    print(f"  Agreed gradients computed: {len(agreed_grads)} entries")

    print("\nTest 3: Gradient agreement with different targets")
    grads_list = []
    targets = [torch.ones(32, 2), -torch.ones(32, 2), torch.zeros(32, 2)]
    for target in targets:
        x = torch.randn(32, 4)
        pred = model(x)
        loss = F.mse_loss(pred, target)
        grads_list.append(compute_gradient_dict(loss, model))

    agreement = compute_mean_agreement(grads_list)
    print(f"  Agreement score: {agreement:.4f}")

    print("\nTest 4: Apply gradients")
    model.zero_grad()
    apply_gradients(model, agreed_grads, max_grad_norm=10.0)

    has_grads = all(p.grad is not None for p in model.parameters())
    print(f"  All parameters have gradients: {has_grads}")

    print("\nTest 5: GAR Buffer")
    buffer = GARBuffer(capacity_per_level=100, obs_shape=(4,))

    # Add samples at different levels
    for level in [0.0, 0.25, 0.5]:
        for _ in range(50):
            obs = np.random.randn(4).astype(np.float32)
            action = np.random.randint(2)
            reward = 1.0
            next_obs = np.random.randn(4).astype(np.float32)
            done = False
            buffer.add(level, obs, action, reward, next_obs, done)

    print(f"  Total samples: {len(buffer)}")
    print(f"  Available levels: {buffer.get_available_levels(min_samples=32)}")

    batch = buffer.sample(0.25, batch_size=16)
    if batch is not None:
        obs, actions, rewards, next_obs, dones = batch
        print(f"  Sampled batch shapes: obs={obs.shape}, actions={actions.shape}")

    print("\nTest 6: GAR Stats")
    stats = GARStats()

    for _ in range(50):
        stats.record_agreement(np.random.uniform(0.5, 1.0))
        stats.record_gradient_norm(np.random.uniform(0.1, 1.0))
        stats.record_variation_loss(0.0, np.random.uniform(0.1, 0.5))
        stats.record_variation_loss(0.5, np.random.uniform(0.2, 0.6))

    summary = stats.get_stats()
    print(f"  Stats: {summary}")

    print("\nAll GAR tests passed!")
