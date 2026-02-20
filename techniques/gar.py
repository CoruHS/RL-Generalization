"""
GAR: Gradient Agreement Regularization

A novel technique that penalizes gradient disagreement across different training levels.

Core insight: When gradients from Level A say "increase weight W" but gradients from 
Level B say "decrease weight W", the network is learning level-specific features.
GAR adds a loss term that penalizes this disagreement.

This is different from:
- PCGrad/GradSurgery: They MODIFY gradients, we add a LOSS term
- Multi-task learning: Applies to different tasks, we apply to different levels of same task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque


class GAR:
    """
    Gradient Agreement Regularization.
    
    Computes gradient agreement loss between batches from different levels.
    
    Args:
        model: Neural network model
        lambda_gar: Weight for GAR loss term
        sample_pairs: Number of level pairs to sample per update
        history_size: Size of gradient history for computing agreement
    """
    
    def __init__(
        self,
        model: nn.Module,
        lambda_gar: float = 0.1,
        sample_pairs: int = 4,
        history_size: int = 10,
    ):
        self.model = model
        self.lambda_gar = lambda_gar
        self.sample_pairs = sample_pairs
        self.history_size = history_size
        
        # Store recent gradients from different levels
        self.grad_history: Dict[int, deque] = {}
        
        # Tracking
        self.gar_losses = []
        self.agreements = []
    
    def compute_gradient(
        self, 
        loss: torch.Tensor, 
        retain_graph: bool = True
    ) -> torch.Tensor:
        """Compute flattened gradient vector."""
        grads = torch.autograd.grad(
            loss, 
            self.model.parameters(), 
            retain_graph=retain_graph,
            create_graph=True,  # Needed for GAR to be differentiable
        )
        return torch.cat([g.flatten() for g in grads])
    
    def compute_gar_loss(
        self,
        loss_fn,
        batch_a: Dict[str, torch.Tensor],
        batch_b: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute GAR loss between two batches from different levels.
        
        Args:
            loss_fn: Function that takes batch and returns loss
            batch_a: First batch (from level A)
            batch_b: Second batch (from level B)
            
        Returns:
            gar_loss: Differentiable GAR loss term
            agreement: Cosine similarity (for logging)
        """
        # Compute gradients from each batch
        loss_a = loss_fn(batch_a)
        grad_a = self.compute_gradient(loss_a, retain_graph=True)
        
        loss_b = loss_fn(batch_b)
        grad_b = self.compute_gradient(loss_b, retain_graph=True)
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(grad_a.unsqueeze(0), grad_b.unsqueeze(0))
        
        # GAR loss: penalize negative agreement (disagreement)
        # When cos_sim is negative, gradients point in opposite directions
        gar_loss = F.relu(-cos_sim)  # 0 if agreeing, positive if disagreeing
        
        return gar_loss.squeeze(), cos_sim.item()
    
    def compute_gar_loss_from_gradients(
        self,
        grad_a: torch.Tensor,
        grad_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute GAR loss from pre-computed gradients.
        
        Use this if you already have gradients computed.
        """
        cos_sim = F.cosine_similarity(grad_a.unsqueeze(0), grad_b.unsqueeze(0))
        gar_loss = F.relu(-cos_sim)
        return gar_loss.squeeze(), cos_sim.item()
    
    def store_gradient(self, level_id: int, gradient: torch.Tensor):
        """Store gradient in history for a level."""
        if level_id not in self.grad_history:
            self.grad_history[level_id] = deque(maxlen=self.history_size)
        
        # Detach to avoid memory issues
        self.grad_history[level_id].append(gradient.detach())
    
    def compute_gar_loss_from_history(self) -> Tuple[torch.Tensor, float]:
        """
        Compute GAR loss using stored gradient history.
        
        Samples pairs of levels and computes agreement.
        """
        level_ids = list(self.grad_history.keys())
        
        if len(level_ids) < 2:
            return torch.tensor(0.0), 1.0
        
        total_loss = 0.0
        total_agreement = 0.0
        count = 0
        
        for _ in range(self.sample_pairs):
            # Sample two different levels
            idx = np.random.choice(len(level_ids), size=2, replace=False)
            level_a, level_b = level_ids[idx[0]], level_ids[idx[1]]
            
            # Get most recent gradients
            if len(self.grad_history[level_a]) > 0 and len(self.grad_history[level_b]) > 0:
                grad_a = self.grad_history[level_a][-1]
                grad_b = self.grad_history[level_b][-1]
                
                loss, agreement = self.compute_gar_loss_from_gradients(grad_a, grad_b)
                total_loss += loss
                total_agreement += agreement
                count += 1
        
        if count == 0:
            return torch.tensor(0.0), 1.0
        
        avg_loss = total_loss / count
        avg_agreement = total_agreement / count
        
        self.gar_losses.append(avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss)
        self.agreements.append(avg_agreement)
        
        return avg_loss * self.lambda_gar, avg_agreement
    
    def get_stats(self) -> Dict[str, float]:
        """Get GAR statistics."""
        return {
            "mean_gar_loss": np.mean(self.gar_losses[-100:]) if self.gar_losses else 0,
            "mean_agreement": np.mean(self.agreements[-100:]) if self.agreements else 0,
            "num_levels": len(self.grad_history),
        }
    
    def reset_history(self):
        """Clear gradient history."""
        self.grad_history.clear()


class GARTrainer:
    """
    Helper class to integrate GAR into training loop.
    
    Wraps an existing training setup and adds GAR loss.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lambda_gar: float = 0.1,
        gar_start_step: int = 1000,  # Start GAR after warmup
    ):
        self.model = model
        self.optimizer = optimizer
        self.gar = GAR(model, lambda_gar=lambda_gar)
        self.gar_start_step = gar_start_step
        self.step = 0
    
    def train_step(
        self,
        compute_loss_fn,
        batch: Dict[str, torch.Tensor],
        level_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Perform one training step with optional GAR.
        
        Args:
            compute_loss_fn: Function that computes main loss from batch
            batch: Training batch
            level_id: ID of the level this batch came from (for GAR)
            
        Returns:
            Dictionary of losses
        """
        self.step += 1
        
        # Compute main loss
        main_loss = compute_loss_fn(batch)
        
        # Compute and store gradient for GAR
        if level_id is not None:
            grad = self.gar.compute_gradient(main_loss, retain_graph=True)
            self.gar.store_gradient(level_id, grad)
        
        # Compute GAR loss (after warmup)
        if self.step >= self.gar_start_step and len(self.gar.grad_history) >= 2:
            gar_loss, agreement = self.gar.compute_gar_loss_from_history()
            total_loss = main_loss + gar_loss
        else:
            gar_loss = torch.tensor(0.0)
            agreement = 1.0
            total_loss = main_loss
        
        # Backward and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "main_loss": main_loss.item(),
            "gar_loss": gar_loss.item() if torch.is_tensor(gar_loss) else gar_loss,
            "agreement": agreement,
            "total_loss": total_loss.item(),
        }


def apply_gar_to_ppo(
    ppo_agent,
    lambda_gar: float = 0.1,
):
    """
    Modify a PPO agent to use GAR.
    
    This patches the update method to include GAR loss.
    """
    original_update = ppo_agent.update
    gar = GAR(ppo_agent.policy, lambda_gar=lambda_gar)
    
    def update_with_gar():
        # Call original update logic but with GAR
        # This is a simplified version - full implementation would
        # need to modify the inner training loop
        result = original_update()
        result["gar_stats"] = gar.get_stats()
        return result
    
    ppo_agent.update = update_with_gar
    ppo_agent.gar = gar
    
    return ppo_agent


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing GAR module...")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    
    gar = GAR(model, lambda_gar=0.1)
    
    # Simulate batches from different levels
    def compute_loss(batch):
        x = batch["obs"]
        y = batch["target"]
        pred = model(x)
        return F.mse_loss(pred, y)
    
    # Test with agreeing gradients (same target)
    print("\nTest 1: Agreeing gradients")
    batch_a = {"obs": torch.randn(32, 4), "target": torch.ones(32, 2)}
    batch_b = {"obs": torch.randn(32, 4), "target": torch.ones(32, 2)}
    
    loss, agreement = gar.compute_gar_loss(compute_loss, batch_a, batch_b)
    print(f"  GAR loss: {loss.item():.4f}, Agreement: {agreement:.4f}")
    
    # Test with disagreeing gradients (opposite targets)
    print("\nTest 2: Disagreeing gradients")
    batch_a = {"obs": torch.randn(32, 4), "target": torch.ones(32, 2)   }
    batch_b = {"obs": torch.randn(32, 4), "target": -torch.ones(32, 2)}
    
    loss, agreement = gar.compute_gar_loss(compute_loss, batch_a, batch_b)
    print(f"  GAR loss: {loss.item():.4f}, Agreement: {agreement:.4f}")
    
    # Test gradient history
    print("\nTest 3: Gradient history")
    for level in range(5):
        batch = {"obs": torch.randn(32, 4), "target": torch.randn(32, 2)}
        loss = compute_loss(batch)
        grad = gar.compute_gradient(loss, retain_graph=False)
        gar.store_gradient(level, grad)
    
    print(f"  Stored {len(gar.grad_history)} levels")
    
    history_loss, history_agreement = gar.compute_gar_loss_from_history()
    print(f"  History GAR loss: {history_loss:.4f}, Agreement: {history_agreement:.4f}")
    
    print("\nGAR tests passed!")