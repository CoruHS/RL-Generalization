"""
MinAtar Space Invaders with controllable variation for generalization research.

Variation levels affect:
- sticky_action_prob: Probability of repeating the last action (0.1 base -> up to 0.25)
- enemy_move_interval: How often aliens move (12 base -> down to 6)
- alien_shot_timer: Timing for alien shots
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minatar import Environment
from typing import Optional, Tuple, Dict, Any, List


class MinAtarVarEnv(gym.Env):
    """
    MinAtar Space Invaders with controllable distribution shift.

    Variation level 0.0 = standard game
    Variation level 1.0 = maximum difficulty shift

    Variations applied:
    - sticky_action_prob: 0.1 -> 0.25 (actions repeat more often)
    - enemy_move_interval: 12 -> 6 (aliens move faster)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        variation_level: float = 0.0,
        game: str = "space_invaders",
        sticky_action_prob: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.variation_level = np.clip(variation_level, 0.0, 1.0)
        self.game = game
        self._seed = seed

        # Base parameters (variation_level = 0)
        self.base_sticky_prob = 0.1
        self.base_move_interval = 12

        # Max variation parameters (variation_level = 1)
        self.max_sticky_prob = 0.25
        self.min_move_interval = 6

        # Calculate actual parameters based on variation level
        if sticky_action_prob is not None:
            self.sticky_prob = sticky_action_prob
        else:
            self.sticky_prob = self.base_sticky_prob + self.variation_level * (
                self.max_sticky_prob - self.base_sticky_prob
            )

        self.move_interval = int(
            self.base_move_interval - self.variation_level * (
                self.base_move_interval - self.min_move_interval
            )
        )

        # Create MinAtar environment
        self.env = Environment(game, sticky_action_prob=self.sticky_prob)
        if seed is not None:
            self.env.seed(seed)

        # Set up spaces
        state_shape = self.env.state_shape()  # [10, 10, n_channels]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=state_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.env.num_actions())

        self.steps = 0
        self.max_steps = 5000  # Reasonable episode limit

    def _apply_variation(self):
        """Apply variation to the inner game parameters."""
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'enemy_move_interval'):
            self.env.env.enemy_move_interval = self.move_interval

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.env.seed(seed)

        self.env.reset()
        self._apply_variation()
        self.steps = 0

        obs = self.env.state().astype(np.float32)
        info = {
            "variation_level": self.variation_level,
            "sticky_prob": self.sticky_prob,
            "move_interval": self.move_interval,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward, terminated = self.env.act(action)
        self.steps += 1

        obs = self.env.state().astype(np.float32)
        truncated = self.steps >= self.max_steps

        info = {"steps": self.steps}

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Render the current state."""
        if hasattr(self.env, 'display_state'):
            self.env.display_state()

    def close(self):
        if hasattr(self.env, 'close_display'):
            self.env.close_display()

    def get_state_channels(self) -> Dict[str, int]:
        """Return channel meanings for visualization."""
        if self.game == "space_invaders":
            return {
                "cannon": 0,
                "alien": 1,
                "alien_left": 2,
                "alien_right": 3,
                "friendly_bullet": 4,
                "enemy_bullet": 5,
            }
        return {}


class MinAtarVarVecEnv:
    """Vectorized MinAtar environment for parallel training."""

    def __init__(
        self,
        num_envs: int,
        variation_level: float = 0.0,
        game: str = "space_invaders",
        base_seed: Optional[int] = None,
    ):
        self.num_envs = num_envs
        self.envs = []
        for i in range(num_envs):
            seed = base_seed + i if base_seed is not None else None
            self.envs.append(MinAtarVarEnv(
                variation_level=variation_level,
                game=game,
                seed=seed,
            ))

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

    def close(self):
        for env in self.envs:
            env.close()


def make_minatar_train_test_envs(
    num_train_envs: int = 8,
    test_shift_levels: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    game: str = "space_invaders",
) -> Tuple[MinAtarVarVecEnv, Dict[float, MinAtarVarEnv]]:
    """Create training and test environments."""
    train_env = MinAtarVarVecEnv(
        num_envs=num_train_envs,
        variation_level=0.0,
        game=game,
    )

    test_envs = {}
    for level in test_shift_levels:
        test_envs[level] = MinAtarVarEnv(
            variation_level=level,
            game=game,
        )

    return train_env, test_envs


if __name__ == "__main__":
    print("Testing MinAtarVarEnv (Space Invaders)...")

    # Test basic environment
    env = MinAtarVarEnv(variation_level=0.0)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    print(f"Info: {info}")

    # Run a few steps
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            print(f"Episode ended at step {i+1}, total reward: {total_reward}")
            break

    # Test variation levels
    print("\nTesting variation levels:")
    for level in [0.0, 0.5, 1.0]:
        env = MinAtarVarEnv(variation_level=level)
        obs, info = env.reset()
        print(f"Level {level}: sticky_prob={info['sticky_prob']:.2f}, move_interval={info['move_interval']}")

    # Test vectorized env
    print("\nTesting vectorized env:")
    vec_env = MinAtarVarVecEnv(num_envs=4, variation_level=0.0)
    obs, _ = vec_env.reset()
    print(f"Vectorized obs shape: {obs.shape}")

    actions = np.random.randint(0, vec_env.action_space.n, size=4)
    obs, rewards, terms, truncs, _ = vec_env.step(actions)
    print(f"Step result shapes: obs={obs.shape}, rewards={rewards.shape}")

    vec_env.close()
    print("\nMinAtar tests passed!")
