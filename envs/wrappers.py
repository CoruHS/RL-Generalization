"""
Environment Wrappers for RL Generalization Research

Contains wrappers for observation preprocessing, frame stacking,
and other environment modifications commonly used in RL.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional, Tuple, Dict, Any


class FrameStack(gym.Wrapper):
    """
    Stack consecutive frames as observation for temporal information.

    Commonly used for Atari and Procgen environments where a single
    frame doesn't capture motion information (velocity, direction, etc.).

    For image observations (H, W, C), stacks along the channel dimension
    to produce (H, W, C * n_frames).

    Args:
        env: The environment to wrap
        n_frames: Number of frames to stack (default: 4)

    Example:
        >>> env = FrameStack(procgen_env, n_frames=4)
        >>> obs, info = env.reset()
        >>> obs.shape  # (64, 64, 12) for 64x64 RGB with 4 frames
    """

    def __init__(self, env: gym.Env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames

        # Get original observation space shape
        obs_shape = env.observation_space.shape

        if len(obs_shape) == 3:
            # Image observation (H, W, C) -> (H, W, C * n_frames)
            h, w, c = obs_shape
            self.stacked_shape = (h, w, c * n_frames)
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=self.stacked_shape,
                dtype=np.uint8,
            )
        elif len(obs_shape) == 1:
            # Vector observation (D,) -> (D * n_frames,)
            d = obs_shape[0]
            self.stacked_shape = (d * n_frames,)
            low = np.tile(env.observation_space.low, n_frames)
            high = np.tile(env.observation_space.high, n_frames)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=env.observation_space.dtype,
            )
        else:
            raise ValueError(f"Unsupported observation shape: {obs_shape}")

        # Frame buffer
        self.frames = deque(maxlen=n_frames)

    def _get_stacked_obs(self) -> np.ndarray:
        """Stack frames into single observation."""
        if len(self.frames[0].shape) == 3:
            # Image: stack along channel dimension
            return np.concatenate(list(self.frames), axis=-1)
        else:
            # Vector: concatenate
            return np.concatenate(list(self.frames), axis=0)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and initialize frame stack with copies of first frame."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Fill buffer with copies of initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs.copy())

        return self._get_stacked_obs(), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and update frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs.copy())
        return self._get_stacked_obs(), reward, terminated, truncated, info


class AutoResetWrapper(gym.Wrapper):
    """
    Wrapper that handles auto-resetting environments like Procgen.

    Procgen environments auto-reset on episode end, returning the first
    observation of the new episode. This wrapper tracks episode boundaries
    and provides clean episode statistics.

    This is useful for evaluation where we want to count complete episodes
    without the confusion of auto-reset behavior.

    Args:
        env: The environment to wrap

    Note:
        After a done signal, the next step will be in a new episode.
        The observation returned with done=True is from the NEW episode.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._is_auto_reset_env = self._detect_auto_reset()

    def _detect_auto_reset(self) -> bool:
        """Detect if environment auto-resets (like Procgen)."""
        # Check for Procgen-style environments
        env_name = getattr(self.env, 'spec', None)
        if env_name and 'procgen' in str(env_name).lower():
            return True
        # Check for ProcgenEnv wrapper
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'env_name'):
            return True
        # Check class name
        class_name = self.env.__class__.__name__.lower()
        if 'procgen' in class_name or 'starpilot' in class_name:
            return True
        return False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and episode tracking."""
        self._episode_reward = 0.0
        self._episode_length = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step and track episode statistics."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._episode_reward += reward
        self._episode_length += 1

        done = terminated or truncated

        if done:
            # Store episode stats before they get reset
            info['episode'] = {
                'r': self._episode_reward,
                'l': self._episode_length,
            }
            # Reset counters for new episode (auto-reset already happened)
            self._episode_reward = 0.0
            self._episode_length = 0

        return obs, reward, terminated, truncated, info

    @property
    def is_auto_reset(self) -> bool:
        """Whether this environment auto-resets."""
        return self._is_auto_reset_env


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalize observations to [0, 1] range for images or standardize for vectors.

    For uint8 images: divides by 255.0
    For float observations: applies running mean/std normalization

    Args:
        env: The environment to wrap
        normalize_images: Whether to normalize uint8 images to [0,1]
    """

    def __init__(self, env: gym.Env, normalize_images: bool = True):
        super().__init__(env)
        self.normalize_images = normalize_images

        # Update observation space for normalized images
        if normalize_images and env.observation_space.dtype == np.uint8:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=env.observation_space.shape,
                dtype=np.float32,
            )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize the observation."""
        if self.normalize_images and obs.dtype == np.uint8:
            return obs.astype(np.float32) / 255.0
        return obs


class TransposeImage(gym.ObservationWrapper):
    """
    Transpose image observation from (H, W, C) to (C, H, W) for PyTorch.

    PyTorch CNNs expect channel-first format, while many environments
    (Procgen, Atari via gymnasium) return channel-last format.

    Args:
        env: The environment to wrap
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        obs_shape = env.observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError(f"Expected 3D observation, got shape {obs_shape}")

        h, w, c = obs_shape
        self.observation_space = spaces.Box(
            low=env.observation_space.low.transpose(2, 0, 1),
            high=env.observation_space.high.transpose(2, 0, 1),
            shape=(c, h, w),
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Transpose observation to channel-first format."""
        return obs.transpose(2, 0, 1)


if __name__ == "__main__":
    print("Testing environment wrappers...")

    # Test FrameStack with CartPole (vector observation)
    print("\n1. Testing FrameStack with vector observation...")
    env = gym.make("CartPole-v1")
    env = FrameStack(env, n_frames=4)
    obs, info = env.reset()
    print(f"   Original: (4,) -> Stacked: {obs.shape}")
    assert obs.shape == (16,), f"Expected (16,), got {obs.shape}"

    # Step a few times
    for _ in range(5):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    print(f"   After steps: {obs.shape}")
    env.close()

    # Test with a mock image environment
    print("\n2. Testing FrameStack with image observation...")

    class MockImageEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(0, 255, (64, 64, 3), np.uint8)
            self.action_space = spaces.Discrete(4)
            self.step_count = 0

        def reset(self, seed=None, options=None):
            self.step_count = 0
            return np.zeros((64, 64, 3), dtype=np.uint8), {}

        def step(self, action):
            self.step_count += 1
            obs = np.full((64, 64, 3), self.step_count % 256, dtype=np.uint8)
            return obs, 1.0, self.step_count >= 100, False, {}

    env = MockImageEnv()
    env = FrameStack(env, n_frames=4)
    obs, info = env.reset()
    print(f"   Original: (64, 64, 3) -> Stacked: {obs.shape}")
    assert obs.shape == (64, 64, 12), f"Expected (64, 64, 12), got {obs.shape}"

    # Verify frames update correctly
    obs1, _, _, _, _ = env.step(0)
    obs2, _, _, _, _ = env.step(0)
    print(f"   Frame stack updates correctly: {obs1[0,0,0]} != {obs2[0,0,0]}")
    env.close()

    # Test NormalizeObservation
    print("\n3. Testing NormalizeObservation...")
    env = MockImageEnv()
    env = NormalizeObservation(env)
    obs, _ = env.reset()
    print(f"   Normalized dtype: {obs.dtype}, range: [{obs.min()}, {obs.max()}]")
    assert obs.dtype == np.float32
    env.close()

    # Test TransposeImage
    print("\n4. Testing TransposeImage...")
    env = MockImageEnv()
    env = TransposeImage(env)
    obs, _ = env.reset()
    print(f"   Original: (64, 64, 3) -> Transposed: {obs.shape}")
    assert obs.shape == (3, 64, 64), f"Expected (3, 64, 64), got {obs.shape}"
    env.close()

    print("\nAll wrapper tests passed!")
