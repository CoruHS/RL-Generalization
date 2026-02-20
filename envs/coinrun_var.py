"""
CoinRun (Procgen) with Controllable Variation

This wrapper for OpenAI Procgen's CoinRun allows testing generalization:
- Number of training levels (level diversity)
- Difficulty settings
- Distribution mode (easy/hard)

The variation here is primarily visual + layout since CoinRun uses PCG.

Usage:
    # Training environment (limited levels)
    env = CoinRunVarEnv(variation_level=0.0)
    
    # Test environment with more level diversity  
    env = CoinRunVarEnv(variation_level=0.5)
    
    # Custom settings
    env = CoinRunVarEnv(num_levels=500, start_level=0, distribution_mode="easy")

Requirements:
    pip install procgen

Note: If procgen is not installed, this module provides a MockCoinRunEnv
for testing the experimental framework.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# Try to import procgen, fall back to mock if not available
try:
    import procgen
    PROCGEN_AVAILABLE = True
except ImportError:
    PROCGEN_AVAILABLE = False
    print("Warning: procgen not installed. Using MockCoinRunEnv for testing.")
    print("Install with: pip install procgen")


class MockCoinRunEnv(gym.Env):
    """
    Mock CoinRun environment for testing when procgen is not installed.
    Simulates the interface but uses simple random observations/rewards.
    """
    
    def __init__(self, num_levels: int = 200, start_level: int = 0, **kwargs):
        super().__init__()
        self.num_levels = num_levels
        self.start_level = start_level
        self.current_level = start_level
        
        # CoinRun uses 64x64 RGB observations
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        # 15 discrete actions in CoinRun
        self.action_space = spaces.Discrete(15)
        
        self.step_count = 0
        self.max_steps = 1000
    
    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed)
        if seed is not None:
            self.current_level = self.start_level + (seed % self.num_levels)
        self.step_count = 0
        
        # Random observation simulating visual complexity
        obs = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        info = {"level": self.current_level}
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        
        # Random dynamics for testing
        obs = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        
        # Occasionally succeed
        if np.random.random() < 0.01:
            reward = 10.0
            terminated = True
        else:
            reward = 0.0
            terminated = False
        
        truncated = self.step_count >= self.max_steps
        info = {"level": self.current_level}
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        pass


class CoinRunVarEnv(gym.Env):
    """
    CoinRun environment with controllable variation for generalization research.
    
    Variation is controlled through:
    - num_levels: More levels = more diversity during training
    - distribution_mode: "easy" or "hard" 
    - start_level: Starting seed for level generation
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}
    
    # Variation level mappings
    LEVEL_CONFIGS = {
        0.00: {"num_levels": 200, "distribution_mode": "easy"},
        0.25: {"num_levels": 500, "distribution_mode": "easy"},
        0.50: {"num_levels": 1000, "distribution_mode": "easy"},
        0.75: {"num_levels": 5000, "distribution_mode": "hard"},
        1.00: {"num_levels": 0, "distribution_mode": "hard"},  # 0 = unlimited
    }
    
    def __init__(
        self,
        variation_level: float = 0.0,
        num_levels: Optional[int] = None,
        start_level: int = 0,
        distribution_mode: Optional[str] = None,
        render_mode: Optional[str] = None,
        num_envs: int = 1,
    ):
        """
        Initialize CoinRun with variable settings.
        
        Args:
            variation_level: Float 0-1, controls level diversity and difficulty
            num_levels: Override number of training levels (0 = unlimited)
            start_level: Starting level seed
            distribution_mode: "easy" or "hard"
            render_mode: Render mode for visualization
            num_envs: Number of parallel environments (for vectorized training)
        """
        super().__init__()
        
        self.variation_level = variation_level
        self.render_mode = render_mode
        self.num_envs = num_envs
        
        # Get config for this variation level
        config = self._get_config(variation_level)
        
        self.num_levels = num_levels if num_levels is not None else config["num_levels"]
        self.distribution_mode = distribution_mode if distribution_mode is not None else config["distribution_mode"]
        self.start_level = start_level
        
        # Create underlying procgen environment
        if PROCGEN_AVAILABLE:
            self.env = procgen.ProcgenEnv(
                env_name="coinrun",
                num_envs=num_envs,
                num_levels=self.num_levels,
                start_level=self.start_level,
                distribution_mode=self.distribution_mode,
                render_mode="rgb_array",
            )
            # Procgen returns dict observations, extract rgb
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
            )
            self.action_space = self.env.action_space
        else:
            self.env = MockCoinRunEnv(
                num_levels=self.num_levels if self.num_levels > 0 else 10000,
                start_level=self.start_level,
            )
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
    
    def _get_config(self, level: float) -> Dict[str, Any]:
        """Get configuration for a given variation level."""
        # Find closest defined level
        levels = sorted(self.LEVEL_CONFIGS.keys())
        closest = min(levels, key=lambda x: abs(x - level))
        return self.LEVEL_CONFIGS[closest]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if PROCGEN_AVAILABLE:
            obs = self.env.reset()
            if isinstance(obs, dict):
                obs = obs["rgb"]
            if self.num_envs == 1:
                obs = obs[0]
        else:
            obs, _ = self.env.reset(seed=seed)
        
        info = {
            "variation_level": self.variation_level,
            "num_levels": self.num_levels,
            "distribution_mode": self.distribution_mode,
        }
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        if PROCGEN_AVAILABLE:
            if self.num_envs == 1:
                action = np.array([action])
            obs, reward, done, info = self.env.step(action)
            if isinstance(obs, dict):
                obs = obs["rgb"]
            if self.num_envs == 1:
                obs = obs[0]
                reward = reward[0]
                done = done[0]
                # Procgen returns info as a list, extract first element
                if isinstance(info, list):
                    info = info[0] if len(info) > 0 else {}
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Ensure info is a dict before adding to it
        if not isinstance(info, dict):
            info = {}
        info["variation_level"] = self.variation_level
        
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if PROCGEN_AVAILABLE:
            return self.env.render()
        return None
    
    def close(self):
        """Clean up."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    @classmethod
    def make_shifted(cls, shift_amount: float, **kwargs) -> "CoinRunVarEnv":
        """Factory method to create environment with specific shift."""
        return cls(variation_level=shift_amount, **kwargs)


class CoinRunVarVecEnv:
    """
    Wrapper to provide consistent vectorized interface.
    """
    
    def __init__(
        self,
        num_envs: int,
        variation_level: float = 0.0,
        **kwargs
    ):
        self.num_envs = num_envs
        self.variation_level = variation_level
        
        if PROCGEN_AVAILABLE:
            # Procgen natively supports vectorized envs
            config = CoinRunVarEnv.LEVEL_CONFIGS.get(
                min(CoinRunVarEnv.LEVEL_CONFIGS.keys(), 
                    key=lambda x: abs(x - variation_level))
            )
            self.env = procgen.ProcgenEnv(
                env_name="coinrun",
                num_envs=num_envs,
                num_levels=config["num_levels"],
                start_level=kwargs.get("start_level", 0),
                distribution_mode=config["distribution_mode"],
                render_mode="rgb_array",
            )
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
            )
            self.action_space = self.env.action_space
        else:
            # Create multiple mock environments
            self.envs = [
                MockCoinRunEnv(num_levels=200)
                for _ in range(num_envs)
            ]
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space
    
    def reset(self, seed: Optional[int] = None):
        if PROCGEN_AVAILABLE:
            obs = self.env.reset()
            if isinstance(obs, dict):
                obs = obs["rgb"]
            info = [{"variation_level": self.variation_level} for _ in range(self.num_envs)]
            return obs, info
        else:
            obs_list = []
            info_list = []
            for i, env in enumerate(self.envs):
                env_seed = seed + i if seed is not None else None
                obs, info = env.reset(seed=env_seed)
                obs_list.append(obs)
                info_list.append(info)
            return np.stack(obs_list), info_list
    
    def step(self, actions):
        if PROCGEN_AVAILABLE:
            obs, rewards, dones, infos = self.env.step(actions)
            if isinstance(obs, dict):
                obs = obs["rgb"]
            # Procgen auto-resets, so no terminated/truncated split
            return obs, rewards, dones, np.zeros_like(dones, dtype=bool), infos
        else:
            obs_list, reward_list, term_list, trunc_list, info_list = [], [], [], [], []
            for env, action in zip(self.envs, actions):
                obs, reward, term, trunc, info = env.step(action)
                if term or trunc:
                    obs, _ = env.reset()
                obs_list.append(obs)
                reward_list.append(reward)
                term_list.append(term)
                trunc_list.append(trunc)
                info_list.append(info)
            return (
                np.stack(obs_list),
                np.array(reward_list),
                np.array(term_list),
                np.array(trunc_list),
                info_list
            )
    
    def close(self):
        if PROCGEN_AVAILABLE:
            self.env.close()
        else:
            for env in self.envs:
                env.close()


def make_coinrun_train_test_envs(
    num_train_envs: int = 8,
    test_shift_levels: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    train_levels: int = 200,
    test_levels: int = 0,  # 0 = unlimited (full test distribution)
) -> Tuple[CoinRunVarVecEnv, Dict[float, CoinRunVarEnv]]:
    """
    Create training and test environments.
    
    The key for CoinRun generalization testing:
    - Train on limited levels (e.g., 200)
    - Test on unseen levels
    
    Returns:
        train_env: Vectorized training environment
        test_envs: Dictionary mapping shift level to test environment
    """
    train_env = CoinRunVarVecEnv(
        num_envs=num_train_envs,
        variation_level=0.0,
    )
    
    # For CoinRun, test environments use different start_levels to ensure unseen levels
    test_envs = {}
    for level in test_shift_levels:
        test_envs[level] = CoinRunVarEnv(
            variation_level=level,
            start_level=10000,  # Start from level 10000 to ensure unseen
        )
    
    return train_env, test_envs


# Quick test
if __name__ == "__main__":
    print("Testing CoinRunVarEnv...")
    print(f"Procgen available: {PROCGEN_AVAILABLE}")
    
    # Test default environment
    env = CoinRunVarEnv(variation_level=0.0)
    obs, info = env.reset()
    print(f"Default params: num_levels={info['num_levels']}, mode={info['distribution_mode']}")
    print(f"Observation shape: {obs.shape}")
    
    # Test shifted environment
    env_shifted = CoinRunVarEnv(variation_level=0.5)
    obs, info = env_shifted.reset()
    print(f"\n50% shift params: num_levels={info['num_levels']}, mode={info['distribution_mode']}")
    
    # Test max shift environment
    env_max = CoinRunVarEnv(variation_level=1.0)
    obs, info = env_max.reset()
    print(f"\n100% shift params: num_levels={info['num_levels']}, mode={info['distribution_mode']}")
    
    # Test episode
    env = CoinRunVarEnv(variation_level=0.0)
    obs, _ = env.reset()
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            print(f"Episode ended at step {step}")
            break
    print(f"Episode reward (random policy): {total_reward}")
    
    # Test vectorized env
    vec_env = CoinRunVarVecEnv(num_envs=4, variation_level=0.0)
    obs, infos = vec_env.reset()
    print(f"\nVectorized env obs shape: {obs.shape}")
    
    env.close()
    vec_env.close()
    
    print("\nCoinRunVarEnv tests passed!")