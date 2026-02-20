"""
Starpilot (Procgen) with Controllable Variation

Starpilot is a shooter game from Procgen that learns faster than CoinRun
while still testing generalization through procedural generation.

The agent must shoot enemies and dodge obstacles. Dense rewards make it
more tractable for RL while still requiring generalization.

Usage:
    # Training environment (limited levels)
    env = StarpilotVarEnv(variation_level=0.0)

    # Test environment with more level diversity
    env = StarpilotVarEnv(variation_level=0.5)

Requirements:
    pip install procgen
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
    print("Warning: procgen not installed. Using MockStarpilotEnv for testing.")
    print("Install with: pip install procgen")


class MockStarpilotEnv(gym.Env):
    """
    Mock Starpilot environment for testing when procgen is not installed.
    """

    def __init__(self, num_levels: int = 200, start_level: int = 0, **kwargs):
        super().__init__()
        self.num_levels = num_levels
        self.start_level = start_level
        self.current_level = start_level

        # Starpilot uses 64x64 RGB observations
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
        )
        # 15 discrete actions
        self.action_space = spaces.Discrete(15)

        self.step_count = 0
        self.max_steps = 1000

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed)
        if seed is not None:
            self.current_level = self.start_level + (seed % self.num_levels)
        self.step_count = 0

        obs = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        info = {"level": self.current_level}
        return obs, info

    def step(self, action):
        self.step_count += 1

        obs = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)

        # Starpilot has denser rewards than CoinRun
        # Simulate shooting enemies
        if np.random.random() < 0.1:
            reward = 1.0
        else:
            reward = 0.0

        # Episode termination
        terminated = np.random.random() < 0.005
        truncated = self.step_count >= self.max_steps
        info = {"level": self.current_level}

        return obs, reward, terminated, truncated, info

    def close(self):
        pass


class StarpilotVarEnv(gym.Env):
    """
    Starpilot environment with controllable variation for generalization research.

    Starpilot is a better choice than CoinRun for limited compute:
    - Dense rewards (shooting enemies gives immediate feedback)
    - Learns in 2-5M steps vs 25M+ for CoinRun
    - Still tests generalization via procedural generation
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
        super().__init__()

        self.variation_level = variation_level
        self.render_mode = render_mode
        self.num_envs = num_envs

        config = self._get_config(variation_level)

        self.num_levels = num_levels if num_levels is not None else config["num_levels"]
        self.distribution_mode = distribution_mode if distribution_mode is not None else config["distribution_mode"]
        self.start_level = start_level

        # Track if we've done initial reset (Procgen auto-resets, so we skip subsequent resets)
        self._initialized = False
        self._last_obs = None

        if PROCGEN_AVAILABLE:
            self.env = procgen.ProcgenEnv(
                env_name="starpilot",
                num_envs=num_envs,
                num_levels=self.num_levels,
                start_level=self.start_level,
                distribution_mode=self.distribution_mode,
                render_mode="rgb_array",
            )
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(64, 64, 3), dtype=np.uint8
            )
            self.action_space = self.env.action_space
        else:
            self.env = MockStarpilotEnv(
                num_levels=self.num_levels if self.num_levels > 0 else 10000,
                start_level=self.start_level,
            )
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space

    def _get_config(self, level: float) -> Dict[str, Any]:
        levels = sorted(self.LEVEL_CONFIGS.keys())
        closest = min(levels, key=lambda x: abs(x - level))
        return self.LEVEL_CONFIGS[closest]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if PROCGEN_AVAILABLE:
            # Procgen auto-resets, so after initialization we return the last observation
            # from step() which is already the first obs of the new episode
            if self._initialized and self._last_obs is not None:
                obs = self._last_obs
            else:
                obs = self.env.reset()
                if isinstance(obs, dict):
                    obs = obs["rgb"]
                if self.num_envs == 1:
                    obs = obs[0]
                self._initialized = True
                self._last_obs = obs
        else:
            obs, _ = self.env.reset(seed=seed)

        info = {
            "variation_level": self.variation_level,
            "num_levels": self.num_levels,
            "distribution_mode": self.distribution_mode,
        }

        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
                if isinstance(info, list):
                    info = info[0] if len(info) > 0 else {}
            terminated = done
            truncated = False
            # Store obs for next reset() call (Procgen auto-resets, obs is already new episode)
            self._last_obs = obs
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)

        if not isinstance(info, dict):
            info = {}
        info["variation_level"] = self.variation_level

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if PROCGEN_AVAILABLE:
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    @classmethod
    def make_shifted(cls, shift_amount: float, **kwargs) -> "StarpilotVarEnv":
        return cls(variation_level=shift_amount, **kwargs)


if __name__ == "__main__":
    print("Testing StarpilotVarEnv...")
    print(f"Procgen available: {PROCGEN_AVAILABLE}")

    env = StarpilotVarEnv(variation_level=0.0)
    obs, info = env.reset()
    print(f"Default: num_levels={info['num_levels']}, mode={info['distribution_mode']}")
    print(f"Observation shape: {obs.shape}")

    # Run episode
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            print(f"Episode ended at step {step}")
            break
    print(f"Episode reward (random policy): {total_reward}")

    env.close()
    print("StarpilotVarEnv tests passed!")
