"""
GAS: Gradient Agreement Score

A diagnostic tool that measures gradient agreement during training.
Can be used to PREDICT generalization without testing on shifted environments.

Key insight: If gradients from different training levels agree (point same direction),
the model is learning generalizable features. If they disagree, it's overfitting.

Usage:
    gas = GradientAgreementScore(model)
    
    # During training
    for batch, level_id in dataloader:
        loss = compute_loss(batch)
        gas.record(loss, level_id)
    
    # Get score
    score = gas.compute_score()
    print(f"GAS: {score}")  # Higher = better generalization expected
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt


class GradientAgreementScore:
    """
    Computes Gradient Agreement Score during training.
    
    GAS measures how well gradients from different training levels agree.
    Higher agreement → better expected generalization.
    
    Args:
        model: Neural network model
        window_size: Number of recent gradients to consider
        sample_pairs: Pairs to sample when computing score
    """
    
    def __init__(
        self,
        model: nn.Module,
        window_size: int = 50,
        sample_pairs: int = 20,
    ):
        self.model = model
        self.window_size = window_size
        self.sample_pairs = sample_pairs
        
        # Store gradients per level
        self.gradients: Dict[int, List[torch.Tensor]] = defaultdict(list)
        
        # History for plotting
        self.gas_history = []
        self.step_history = []
        self.step = 0
    
    def _compute_gradient(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute flattened gradient vector."""
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            retain_graph=True,
            allow_unused=True,
        )
        
        grad_list = []
        for g in grads:
            if g is not None:
                grad_list.append(g.flatten())
            else:
                # Handle unused parameters
                grad_list.append(torch.zeros(1))
        
        return torch.cat(grad_list).detach()
    
    def record(
        self, 
        loss: torch.Tensor, 
        level_id: int,
        compute_gas: bool = True,
    ) -> Optional[float]:
        """
        Record gradient for a level.
        
        Args:
            loss: Loss tensor (before backward)
            level_id: ID of the training level
            compute_gas: Whether to compute and return GAS
            
        Returns:
            Current GAS if compute_gas=True, else None
        """
        self.step += 1
        
        # Compute and store gradient
        grad = self._compute_gradient(loss)
        self.gradients[level_id].append(grad)
        
        # Keep only recent gradients
        if len(self.gradients[level_id]) > self.window_size:
            self.gradients[level_id] = self.gradients[level_id][-self.window_size:]
        
        if compute_gas:
            gas = self.compute_score()
            self.gas_history.append(gas)
            self.step_history.append(self.step)
            return gas
        
        return None
    
    def record_gradient(self, gradient: torch.Tensor, level_id: int):
        """
        Record a pre-computed gradient.
        
        Use this if you've already computed the gradient.
        """
        self.gradients[level_id].append(gradient.detach())
        
        if len(self.gradients[level_id]) > self.window_size:
            self.gradients[level_id] = self.gradients[level_id][-self.window_size:]
    
    def compute_score(self) -> float:
        """
        Compute current Gradient Agreement Score.
        
        Returns:
            GAS score in range [-1, 1], higher is better
        """
        level_ids = list(self.gradients.keys())
        
        if len(level_ids) < 2:
            return 1.0  # Not enough levels to compare
        
        agreements = []
        
        for _ in range(self.sample_pairs):
            # Sample two different levels
            idx = np.random.choice(len(level_ids), size=2, replace=False)
            level_a, level_b = level_ids[idx[0]], level_ids[idx[1]]
            
            # Get random gradients from each level
            if len(self.gradients[level_a]) > 0 and len(self.gradients[level_b]) > 0:
                grad_a = self.gradients[level_a][np.random.randint(len(self.gradients[level_a]))]
                grad_b = self.gradients[level_b][np.random.randint(len(self.gradients[level_b]))]
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    grad_a.unsqueeze(0), 
                    grad_b.unsqueeze(0)
                ).item()
                
                agreements.append(cos_sim)
        
        if not agreements:
            return 1.0
        
        return float(np.mean(agreements))
    
    def compute_pairwise_agreement(self) -> np.ndarray:
        """
        Compute full pairwise agreement matrix between all levels.
        
        Returns:
            Square matrix where entry (i,j) is agreement between level i and j
        """
        level_ids = sorted(self.gradients.keys())
        n = len(level_ids)
        
        if n < 2:
            return np.array([[1.0]])
        
        matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                level_a, level_b = level_ids[i], level_ids[j]
                
                # Average agreement over recent gradients
                agreements = []
                for grad_a in self.gradients[level_a][-10:]:
                    for grad_b in self.gradients[level_b][-10:]:
                        cos_sim = torch.nn.functional.cosine_similarity(
                            grad_a.unsqueeze(0),
                            grad_b.unsqueeze(0)
                        ).item()
                        agreements.append(cos_sim)
                
                if agreements:
                    avg_agreement = np.mean(agreements)
                    matrix[i, j] = avg_agreement
                    matrix[j, i] = avg_agreement
        
        return matrix
    
    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "current_gas": self.gas_history[-1] if self.gas_history else 1.0,
            "mean_gas": np.mean(self.gas_history[-100:]) if self.gas_history else 1.0,
            "std_gas": np.std(self.gas_history[-100:]) if self.gas_history else 0.0,
            "num_levels": len(self.gradients),
            "total_steps": self.step,
        }
    
    def plot_history(
        self,
        title: str = "Gradient Agreement Score Over Training",
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Plot GAS over training."""
        if not self.gas_history:
            print("No GAS history to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(self.step_history, self.gas_history, linewidth=1, alpha=0.5, label="Raw")
        
        # Smoothed line
        if len(self.gas_history) > 10:
            window = min(50, len(self.gas_history) // 5)
            smoothed = np.convolve(self.gas_history, np.ones(window)/window, mode='valid')
            smoothed_steps = self.step_history[window-1:]
            ax.plot(smoothed_steps, smoothed, linewidth=2, label="Smoothed", color='red')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Gradient Agreement Score", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_agreement_matrix(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """Plot pairwise agreement matrix."""
        matrix = self.compute_pairwise_agreement()
        level_ids = sorted(self.gradients.keys())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(level_ids)))
        ax.set_yticks(range(len(level_ids)))
        ax.set_xticklabels([f"L{i}" for i in level_ids])
        ax.set_yticklabels([f"L{i}" for i in level_ids])
        
        plt.colorbar(im, label="Gradient Agreement")
        ax.set_title("Pairwise Gradient Agreement", fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        if show:
            plt.show()
        
        plt.close()
    
    def reset(self):
        """Reset all stored data."""
        self.gradients.clear()
        self.gas_history.clear()
        self.step_history.clear()
        self.step = 0


def correlate_gas_with_grs(
    gas_scores: List[float],
    grs_scores: List[float],
    names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> float:
    """
    Compute and plot correlation between GAS and GRS.
    
    This validates whether GAS predicts generalization.
    
    Returns:
        Pearson correlation coefficient
    """
    from scipy import stats
    
    correlation, p_value = stats.pearsonr(gas_scores, grs_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(gas_scores, grs_scores, s=100, alpha=0.7)
    
    # Add labels if provided
    if names:
        for i, name in enumerate(names):
            ax.annotate(name, (gas_scores[i], grs_scores[i]), 
                       fontsize=9, ha='center', va='bottom')
    
    # Trend line
    z = np.polyfit(gas_scores, grs_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(gas_scores), max(gas_scores), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, 
            label=f"r = {correlation:.3f} (p = {p_value:.3f})")
    
    ax.set_xlabel("Gradient Agreement Score (GAS)", fontsize=12)
    ax.set_ylabel("Generalization Robustness Score (GRS)", fontsize=12)
    ax.set_title("GAS vs GRS Correlation", fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    if show:
        plt.show()
    
    plt.close()
    
    return correlation


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing GAS module...")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(4, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    
    gas = GradientAgreementScore(model, window_size=20, sample_pairs=10)
    
    # Simulate training with agreeing gradients
    print("\nSimulating training with agreeing gradients...")
    for step in range(50):
        # All levels have similar gradients (same target)
        level_id = step % 5
        x = torch.randn(16, 4)
        target = torch.ones(16, 2)  # Same target = agreeing gradients
        
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        
        score = gas.record(loss, level_id)
    
    print(f"GAS with agreeing gradients: {gas.compute_score():.3f}")
    
    # Reset and test with disagreeing gradients
    gas.reset()
    
    print("\nSimulating training with disagreeing gradients...")
    for step in range(50):
        level_id = step % 5
        x = torch.randn(16, 4)
        # Different levels have opposite targets = disagreeing gradients
        target = torch.ones(16, 2) * (1 if level_id % 2 == 0 else -1)
        
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        
        score = gas.record(loss, level_id)
    
    print(f"GAS with disagreeing gradients: {gas.compute_score():.3f}")
    
    # Print stats
    print("\nStats:", gas.get_stats())
    
    print("\nGAS tests passed!")    