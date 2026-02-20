"""
Shared Neural Network Architectures

These networks are used by DQN, PPO, and ES.
- MLP: For state-based environments (CartPole)
- CNN: For image-based environments (MiniGrid, CoinRun)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network for image observations.
    
    Handles both (C, H, W) and (H, W, C) input formats.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        output_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Handle both (C, H, W) and (H, W, C) formats
        if input_shape[0] in [1, 3, 4]:
            channels, height, width = input_shape
        else:
            height, width, channels = input_shape
        
        self.channels = channels
        self.height = height
        self.width = width
        
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.conv(dummy)
            self.flat_size = conv_out.reshape(1, -1).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.input_shape = input_shape
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] == self.channels:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0
        
        conv_out = self.conv(x)
        flat = conv_out.reshape(conv_out.size(0), -1)
        return self.fc(flat)


class DQNNetwork(nn.Module):
    """
    Q-Network for DQN.
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        use_cnn: bool = False,
    ):
        super().__init__()
        
        self.use_cnn = use_cnn
        self.action_dim = action_dim
        
        if use_cnn:
            if obs_dim[0] in [1, 3, 4]:
                self.channels = obs_dim[0]
            else:
                self.channels = obs_dim[2]
            self.network = CNN(obs_dim, action_dim)
        else:
            self.network = MLP(obs_dim, action_dim, hidden_dims)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
    
    def get_action(self, obs: torch.Tensor, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            if obs.dim() == 1 or (obs.dim() == 3 and self.use_cnn):
                obs = obs.unsqueeze(0)
            q_values = self.forward(obs)
            return q_values.argmax(dim=1).item()


class PPONetwork(nn.Module):
    """
    Actor-Critic Network for PPO.
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        use_cnn: bool = False,
    ):
        super().__init__()
        
        self.use_cnn = use_cnn
        self.action_dim = action_dim
        
        if use_cnn:
            if obs_dim[0] in [1, 3, 4]:
                channels, height, width = obs_dim
            else:
                height, width, channels = obs_dim
            
            self.channels = channels
            self.height = height
            self.width = width
            
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            
            with torch.no_grad():
                dummy = torch.zeros(1, channels, height, width)
                conv_out = self.encoder(dummy)
                feature_dim = conv_out.reshape(1, -1).shape[1]
        else:
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
            )
            feature_dim = hidden_dims[1]
        
        self.policy_head = nn.Linear(feature_dim, action_dim)
        self.value_head = nn.Linear(feature_dim, 1)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_cnn:
            if obs.dim() == 4 and obs.shape[-1] == self.channels:
                obs = obs.permute(0, 3, 1, 2).contiguous()
            if obs.dtype == torch.uint8:
                obs = obs.float() / 255.0
            elif obs.max() > 1.0:
                obs = obs / 255.0
            features = self.encoder(obs)
            features = features.reshape(features.size(0), -1)
        else:
            features = self.encoder(obs)
        
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return action_logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            elif self.use_cnn and obs.dim() == 3:
                obs = obs.unsqueeze(0)
            
            action_logits, value = self.forward(obs)
            probs = F.softmax(action_logits, dim=-1)
            
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            
            log_prob = F.log_softmax(action_logits, dim=-1)
            log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
            
            return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits, values = self.forward(obs)
        
        probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_probs, values.squeeze(-1), entropy


class ESNetwork(nn.Module):
    """
    Simple policy network for Evolution Strategies.
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        use_cnn: bool = False,
    ):
        super().__init__()
        
        self.use_cnn = use_cnn
        self.action_dim = action_dim
        
        if use_cnn:
            if obs_dim[0] in [1, 3, 4]:
                self.channels = obs_dim[0]
            else:
                self.channels = obs_dim[2]
            self.network = CNN(obs_dim, action_dim)
        else:
            self.network = MLP(obs_dim, action_dim, hidden_dims)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> int:
        with torch.no_grad():
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            elif self.use_cnn and obs.dim() == 3:
                obs = obs.unsqueeze(0)
            
            logits = self.forward(obs)
            
            if deterministic:
                return logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                return dist.sample().item()
    
    def get_flat_params(self) -> np.ndarray:
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def set_flat_params(self, flat_params: np.ndarray):
        idx = 0
        for param in self.parameters():
            size = param.numel()
            param.data = torch.from_numpy(
                flat_params[idx:idx+size].reshape(param.shape)
            ).float().to(param.device)
            idx += size


if __name__ == "__main__":
    print("Testing networks...")
    
    # Test MLP
    print("\n1. Testing MLP...")
    mlp = MLP(input_dim=4, output_dim=2)
    x = torch.randn(32, 4)
    out = mlp(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test CNN with (C, H, W)
    print("\n2. Testing CNN (C, H, W)...")
    cnn = CNN(input_shape=(3, 64, 64), output_dim=15)
    x = torch.randn(32, 3, 64, 64)
    out = cnn(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test CNN with (H, W, C) - CoinRun style
    print("\n3. Testing CNN (H, W, C)...")
    cnn2 = CNN(input_shape=(64, 64, 3), output_dim=15)
    x = torch.randn(32, 64, 64, 3)
    out = cnn2(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # Test DQN
    print("\n4. Testing DQN (MLP)...")
    dqn = DQNNetwork(obs_dim=4, action_dim=2)
    obs = torch.randn(4)
    action = dqn.get_action(obs, epsilon=0.0)
    print(f"   Action: {action}")
    
    # Test DQN CNN
    print("\n5. Testing DQN (CNN)...")
    dqn_cnn = DQNNetwork(obs_dim=(64, 64, 3), action_dim=15, use_cnn=True)
    obs = torch.randn(64, 64, 3)
    action = dqn_cnn.get_action(obs, epsilon=0.0)
    print(f"   Action: {action}")
    
    # Test PPO
    print("\n6. Testing PPO (MLP)...")
    ppo = PPONetwork(obs_dim=4, action_dim=2)
    obs = torch.randn(4)
    action, log_prob, value = ppo.get_action(obs)
    print(f"   Action: {action}, LogProb: {log_prob:.3f}, Value: {value:.3f}")
    
    # Test PPO CNN
    print("\n7. Testing PPO (CNN)...")
    ppo_cnn = PPONetwork(obs_dim=(64, 64, 3), action_dim=15, use_cnn=True)
    obs = torch.randn(64, 64, 3)
    action, log_prob, value = ppo_cnn.get_action(obs)
    print(f"   Action: {action}, LogProb: {log_prob:.3f}, Value: {value:.3f}")
    
    # Test ES
    print("\n8. Testing ES...")
    es = ESNetwork(obs_dim=4, action_dim=2)
    obs = torch.randn(4)
    action = es.get_action(obs)
    print(f"   Action: {action}")
    
    flat = es.get_flat_params()
    print(f"   Flat params: {flat.shape}")
    es.set_flat_params(flat)
    
    print("\nAll tests passed!")