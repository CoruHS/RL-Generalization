"""
CartPole with Controllable Dynamics Variation

This environment wrapper allows testing generalization to different physics parameters:
- Gravity
- Pole length
- Cart mass
- Pole mass

Usage:
    # Training environment (default physics)
    env = CartPoleVarEnv(variation_level=0.0)
    
    # Test environment with 50% dynamics shift
    env = CartPoleVarEnv(variation_level=0.5)
    
    # Custom physics
    env = CartPoleVarEnv(gravity=12.0, pole_length=0.7)
"""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np
from typing import Optional, Tuple, Dict, Any


class CartPoleVarEnv(gym.Env):
    """
    CartPole environment with controllable dynamics for generalization research.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    # Default (training) parameters
    DEFAULT_GRAVITY = 9.8
    DEFAULT_POLE_LENGTH = 0.5
    DEFAULT_CART_MASS = 1.0
    DEFAULT_POLE_MASS = 0.1
    
    # Maximum shift parameters (100% variation)
    MAX_GRAVITY = 14.7
    MAX_POLE_LENGTH = 1.0
    MAX_CART_MASS = 2.0
    MAX_POLE_MASS = 0.3
    
    def __init__(
        self,
        variation_level: float = 0.0,
        gravity: Optional[float] = None,
        pole_length: Optional[float] = None,
        cart_mass: Optional[float] = None,
        pole_mass: Optional[float] = None,
        render_mode: Optional[str] = None,
        variation_type: str = "all",  # "all", "gravity", "pole_length", "cart_mass", "pole_mass"
    ):
        """
        Initialize CartPole with variable dynamics.
        
        Args:
            variation_level: Float 0-1, where 0 = training distribution, 1 = max shift
            gravity: Override gravity (if None, computed from variation_level)
            pole_length: Override pole length
            cart_mass: Override cart mass
            pole_mass: Override pole mass
            render_mode: Gymnasium render mode
            variation_type: Which parameter(s) to vary
        """
        super().__init__()
        
        self.variation_level = variation_level
        self.variation_type = variation_type
        self.render_mode = render_mode
        
        # Compute parameters based on variation level or use overrides
        self.gravity = gravity if gravity is not None else self._interpolate(
            self.DEFAULT_GRAVITY, self.MAX_GRAVITY, variation_level if variation_type in ["all", "gravity"] else 0
        )
        self.pole_length = pole_length if pole_length is not None else self._interpolate(
            self.DEFAULT_POLE_LENGTH, self.MAX_POLE_LENGTH, variation_level if variation_type in ["all", "pole_length"] else 0
        )
        self.cart_mass = cart_mass if cart_mass is not None else self._interpolate(
            self.DEFAULT_CART_MASS, self.MAX_CART_MASS, variation_level if variation_type in ["all", "cart_mass"] else 0
        )
        self.pole_mass = pole_mass if pole_mass is not None else self._interpolate(
            self.DEFAULT_POLE_MASS, self.MAX_POLE_MASS, variation_level if variation_type in ["all", "pole_mass"] else 0
        )
        
        # Physics computations
        self.total_mass = self.cart_mass + self.pole_mass
        self.polemass_length = self.pole_mass * self.pole_length
        
        # Standard CartPole parameters
        self.force_mag = 10.0
        self.tau = 0.02  # Time step
        self.kinematics_integrator = "euler"
        
        # Thresholds for episode termination
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        
        # Observation and action spaces
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # State
        self.state = None
        self.steps_beyond_terminated = None
        
    def _interpolate(self, default: float, max_val: float, level: float) -> float:
        """Linearly interpolate between default and max value based on level."""
        return default + (max_val - default) * level
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Random initial state (small perturbations)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps_beyond_terminated = None
        
        info = {
            "gravity": self.gravity,
            "pole_length": self.pole_length,
            "cart_mass": self.cart_mass,
            "pole_mass": self.pole_mass,
            "variation_level": self.variation_level,
        }
        
        return self.state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        assert self.action_space.contains(action), f"Invalid action {action}"
        assert self.state is not None, "Call reset() before step()"
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Physics equations
        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_length * (4.0 / 3.0 - self.pole_mass * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass
        
        # Euler integration
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * x_acc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * theta_acc
        else:  # Semi-implicit euler
            x_dot = x_dot + self.tau * x_acc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * theta_acc
            theta = theta + self.tau * theta_dot
        
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        
        # Check termination
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        
        # Reward
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                print("Warning: Calling step() after episode terminated")
            self.steps_beyond_terminated += 1
            reward = 0.0
        
        info = {"variation_level": self.variation_level}
        
        return self.state, reward, terminated, False, info
    
    def render(self):
        """Render the environment."""
        # For simplicity, we'll skip rendering implementation
        # Use gymnasium's CartPole if you need visualization
        pass
    
    def close(self):
        """Clean up resources."""
        pass
    
    @classmethod
    def make_shifted(cls, shift_amount: float, **kwargs) -> "CartPoleVarEnv":
        """Factory method to create environment with specific shift."""
        return cls(variation_level=shift_amount, **kwargs)


class CartPoleVarVecEnv:
    """
    Vectorized environment for faster training with multiple CartPole instances.
    """
    
    def __init__(self, num_envs: int, variation_level: float = 0.0, **kwargs):
        self.num_envs = num_envs
        self.envs = [CartPoleVarEnv(variation_level=variation_level, **kwargs) for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed: Optional[int] = None):
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), info_list
    
    def step(self, actions):
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


def make_cartpole_train_test_envs(
    num_train_envs: int = 8,
    test_shift_levels: list = [0.0, 0.25, 0.5, 0.75, 1.0]
) -> Tuple[CartPoleVarVecEnv, Dict[float, CartPoleVarEnv]]:
    """
    Create training and test environments for experiments.
    
    Returns:
        train_env: Vectorized training environment
        test_envs: Dictionary mapping shift level to test environment
    """
    train_env = CartPoleVarVecEnv(num_envs=num_train_envs, variation_level=0.0)
    test_envs = {level: CartPoleVarEnv(variation_level=level) for level in test_shift_levels}
    return train_env, test_envs


# Quick test
if __name__ == "__main__":
    print("Testing CartPoleVarEnv...")
    
    # Test default environment
    env = CartPoleVarEnv(variation_level=0.0)
    obs, info = env.reset(seed=42)
    print(f"Default params: gravity={info['gravity']:.2f}, pole_length={info['pole_length']:.2f}")
    
    # Test shifted environment
    env_shifted = CartPoleVarEnv(variation_level=0.5)
    obs, info = env_shifted.reset(seed=42)
    print(f"50% shift params: gravity={info['gravity']:.2f}, pole_length={info['pole_length']:.2f}")
    
    # Test episode
    total_reward = 0
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            break
    print(f"Episode reward (random policy): {total_reward}")
    
    # Test vectorized env
    vec_env = CartPoleVarVecEnv(num_envs=4, variation_level=0.0)
    obs, infos = vec_env.reset(seed=42)
    print(f"Vectorized env obs shape: {obs.shape}")
    
    print("\nCartPoleVarEnv tests passed!")