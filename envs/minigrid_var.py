"""
MiniGrid with Controllable Layout Variation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from enum import IntEnum


class Actions(IntEnum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    GOAL = 2
    AGENT = 3
    OBSTACLE = 4


class MiniGridVarEnv(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    DEFAULT_GRID_SIZE = 6
    DEFAULT_NUM_OBSTACLES = 2
    DEFAULT_MAX_STEPS_MULTIPLIER = 4
    
    MAX_GRID_SIZE = 12
    MAX_NUM_OBSTACLES = 8
    
    # Direction vectors: (row_delta, col_delta) matching symbols [">", "v", "<", "^"]
    DIR_TO_VEC = [
        np.array([0, 1]),   # Dir 0: ">" = right = same row, +col
        np.array([1, 0]),   # Dir 1: "v" = down = +row, same col
        np.array([0, -1]),  # Dir 2: "<" = left = same row, -col
        np.array([-1, 0]),  # Dir 3: "^" = up = -row, same col
    ]
    
    def __init__(
        self,
        variation_level: float = 0.0,
        grid_size: Optional[int] = None,
        num_obstacles: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        agent_view_size: int = 7,
        fully_observable: bool = True,
        use_compact_obs: bool = True,
        seed: Optional[int] = None,
    ):
        """
        MiniGrid environment with controllable variation.

        Args:
            variation_level: 0.0 = easy (small grid), 1.0 = hard (large grid, more obstacles)
            grid_size: Override grid size (default: interpolated from variation_level)
            num_obstacles: Override obstacle count
            render_mode: "human" for text rendering
            max_steps: Max steps per episode
            agent_view_size: Size of partial observation (only used if fully_observable=False)
            fully_observable: If True, agent sees full grid; if False, agent-centric view
            use_compact_obs: If True, use 10-dim goal-relative observation (easier to learn).
                            If False, use full grid observation (harder, needs CNN).
            seed: Random seed
        """
        super().__init__()

        self.variation_level = variation_level
        self.render_mode = render_mode
        self.fully_observable = fully_observable
        self.use_compact_obs = use_compact_obs
        self.agent_view_size = agent_view_size

        self.grid_size = grid_size if grid_size is not None else int(
            self._interpolate(self.DEFAULT_GRID_SIZE, self.MAX_GRID_SIZE, variation_level)
        )
        self.num_obstacles = num_obstacles if num_obstacles is not None else int(
            self._interpolate(self.DEFAULT_NUM_OBSTACLES, self.MAX_NUM_OBSTACLES, variation_level)
        )

        self.max_steps = max_steps if max_steps is not None else (
            self.grid_size * self.grid_size * self.DEFAULT_MAX_STEPS_MULTIPLIER
        )

        self.action_space = spaces.Discrete(3)

        # Set observation space based on mode
        if use_compact_obs:
            # Compact: [goal_rel(2), direction(4), front_cell(4)] = 10 values
            # Easy to learn with MLP, but "cheats" by giving goal direction
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(10,),
                dtype=np.float32
            )
        else:
            # Full grid: requires spatial reasoning, better for CNN
            if self.fully_observable:
                obs_size = self.MAX_GRID_SIZE * self.MAX_GRID_SIZE + 4  # grid + direction
            else:
                obs_size = self.agent_view_size * self.agent_view_size + 4
            self.observation_space = spaces.Box(
                low=0,
                high=float(max(CellType)),
                shape=(obs_size,),
                dtype=np.float32
            )
        
        self.grid = None
        self.agent_pos = None
        self.agent_dir = None
        self.goal_pos = None
        self.step_count = 0
        
        self._np_random = None
        if seed is not None:
            self.reset(seed=seed)
    
    def _interpolate(self, default: float, max_val: float, level: float) -> float:
        return default + (max_val - default) * level
    
    def _gen_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        
        self.grid[0, :] = CellType.WALL
        self.grid[-1, :] = CellType.WALL
        self.grid[:, 0] = CellType.WALL
        self.grid[:, -1] = CellType.WALL
        
        available = []
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                available.append((i, j))
        
        self._np_random.shuffle(available)
        
        self.agent_pos = np.array(available.pop())
        self.agent_dir = self._np_random.integers(0, 4)
        
        available.sort(key=lambda p: -np.sum(np.abs(np.array(p) - self.agent_pos)))
        self.goal_pos = np.array(available.pop(0))
        self.grid[self.goal_pos[0], self.goal_pos[1]] = CellType.GOAL
        
        available = [p for p in available 
                     if np.sum(np.abs(np.array(p) - self.agent_pos)) > 1
                     and np.sum(np.abs(np.array(p) - self.goal_pos)) > 1]
        
        num_to_place = min(self.num_obstacles, len(available))
        for i in range(num_to_place):
            if available:
                obs_pos = available.pop()
                self.grid[obs_pos[0], obs_pos[1]] = CellType.OBSTACLE
    
    def _get_obs(self) -> np.ndarray:
        """Get agent's observation based on observation mode."""
        if self.use_compact_obs:
            return self._get_compact_obs()
        else:
            return self._get_grid_obs()

    def _get_compact_obs(self) -> np.ndarray:
        """
        Compact 10-dim observation for easy MLP learning.

        Returns: [goal_rel_row, goal_rel_col, direction(4), front_cell(4)]

        This "cheats" by giving the agent goal-relative information directly,
        making spatial reasoning trivial. Good for testing pipeline, not for
        rigorous generalization research.
        """
        # Relative goal position (normalized by grid size)
        goal_row_rel = (self.goal_pos[0] - self.agent_pos[0]) / self.grid_size
        goal_col_rel = (self.goal_pos[1] - self.agent_pos[1]) / self.grid_size

        # Current direction (one-hot)
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[self.agent_dir] = 1.0

        # What's in front of agent (one-hot: empty, wall, goal, obstacle)
        front_vec = self.DIR_TO_VEC[self.agent_dir]
        front_pos = self.agent_pos + front_vec
        front_onehot = np.zeros(4, dtype=np.float32)

        if (0 <= front_pos[0] < self.grid_size and
            0 <= front_pos[1] < self.grid_size):
            cell = self.grid[front_pos[0], front_pos[1]]
            if cell == CellType.EMPTY:
                front_onehot[0] = 1.0
            elif cell == CellType.WALL:
                front_onehot[1] = 1.0
            elif cell == CellType.GOAL:
                front_onehot[2] = 1.0
            elif cell == CellType.OBSTACLE:
                front_onehot[3] = 1.0
        else:
            front_onehot[1] = 1.0  # Out of bounds = wall

        obs = np.array([goal_row_rel, goal_col_rel], dtype=np.float32)
        return np.concatenate([obs, dir_onehot, front_onehot])

    def _get_grid_obs(self) -> np.ndarray:
        """
        Full grid observation for CNN-based learning.

        Returns flattened grid + direction one-hot.
        Requires the agent to learn spatial reasoning from raw grid data.
        Better for rigorous generalization research but harder to learn with MLPs.
        """
        # Direction one-hot (scaled to match grid values for consistent normalization)
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[self.agent_dir] = float(max(CellType))

        if self.fully_observable:
            # Pad to MAX_GRID_SIZE for consistent shape across variation levels
            padded = np.full(
                (self.MAX_GRID_SIZE, self.MAX_GRID_SIZE),
                CellType.WALL,
                dtype=np.uint8
            )
            padded[:self.grid_size, :self.grid_size] = self.grid
            padded[self.agent_pos[0], self.agent_pos[1]] = CellType.AGENT
            grid_flat = padded.flatten().astype(np.float32)
        else:
            # Agent-centric partial view
            obs = np.zeros((self.agent_view_size, self.agent_view_size), dtype=np.uint8)
            half = self.agent_view_size // 2

            for di in range(-half, half + 1):
                for dj in range(-half, half + 1):
                    gi, gj = self.agent_pos[0] + di, self.agent_pos[1] + dj
                    oi, oj = di + half, dj + half

                    if 0 <= gi < self.grid_size and 0 <= gj < self.grid_size:
                        obs[oi, oj] = self.grid[gi, gj]
                    else:
                        obs[oi, oj] = CellType.WALL

            obs[half, half] = CellType.AGENT
            grid_flat = obs.flatten().astype(np.float32)

        return np.concatenate([grid_flat, dir_onehot])
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()
        
        self._gen_grid()
        self.step_count = 0
        
        info = {
            "grid_size": self.grid_size,
            "num_obstacles": self.num_obstacles,
            "variation_level": self.variation_level,
            "agent_pos": self.agent_pos.tolist(),
            "goal_pos": self.goal_pos.tolist(),
        }
        
        return self._get_obs(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        # Track distance for reward shaping
        old_dist = np.sum(np.abs(self.agent_pos - self.goal_pos))

        if action == Actions.LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == Actions.RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == Actions.FORWARD:
            dir_vec = self.DIR_TO_VEC[self.agent_dir]
            new_pos = self.agent_pos + dir_vec

            if (0 <= new_pos[0] < self.grid_size and
                0 <= new_pos[1] < self.grid_size):
                cell = self.grid[new_pos[0], new_pos[1]]
                if cell != CellType.WALL and cell != CellType.OBSTACLE:
                    self.agent_pos = new_pos

                    if cell == CellType.GOAL:
                        reward = 1.0 - 0.9 * (self.step_count / self.max_steps)
                        terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        if not terminated:
            # Simple, balanced reward shaping
            new_dist = np.sum(np.abs(self.agent_pos - self.goal_pos))

            # All actions get a small step penalty to encourage efficiency
            reward = -0.01

            if action == Actions.FORWARD:
                if old_dist > new_dist:
                    reward = 0.2  # Got closer - good!
                elif old_dist < new_dist:
                    reward = -0.1  # Going wrong way
                # else: hit wall, just step penalty
        
        info = {
            "variation_level": self.variation_level,
            "step_count": self.step_count,
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            symbols = {
                CellType.EMPTY: ".",
                CellType.WALL: "#",
                CellType.GOAL: "G",
                CellType.AGENT: "A",
                CellType.OBSTACLE: "X",
            }
            dir_symbols = [">", "v", "<", "^"]
            
            print("\n" + "=" * (self.grid_size * 2 + 1))
            for i in range(self.grid_size):
                row = ""
                for j in range(self.grid_size):
                    if np.array_equal(self.agent_pos, [i, j]):
                        row += dir_symbols[self.agent_dir] + " "
                    else:
                        row += symbols[self.grid[i, j]] + " "
                print(row)
            print("=" * (self.grid_size * 2 + 1))
    
    def close(self):
        pass
    
    @classmethod
    def make_shifted(cls, shift_amount: float, **kwargs) -> "MiniGridVarEnv":
        return cls(variation_level=shift_amount, **kwargs)


class MiniGridVarVecEnv:
    
    def __init__(
        self,
        num_envs: int,
        variation_level: float = 0.0,
        base_seed: Optional[int] = None,
        **kwargs
    ):
        self.num_envs = num_envs
        self.envs = []
        for i in range(num_envs):
            seed = base_seed + i if base_seed is not None else None
            self.envs.append(MiniGridVarEnv(variation_level=variation_level, seed=seed, **kwargs))
        
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


def make_minigrid_train_test_envs(
    num_train_envs: int = 8,
    test_shift_levels: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    fully_observable: bool = True,
) -> Tuple[MiniGridVarVecEnv, Dict[float, MiniGridVarEnv]]:
    train_env = MiniGridVarVecEnv(
        num_envs=num_train_envs,
        variation_level=0.0,
        fully_observable=fully_observable
    )
    test_envs = {
        level: MiniGridVarEnv(variation_level=level, fully_observable=fully_observable)
        for level in test_shift_levels
    }
    return train_env, test_envs


if __name__ == "__main__":
    print("Testing MiniGridVarEnv...")
    
    env = MiniGridVarEnv(variation_level=0.0, fully_observable=True)
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Grid size: {info['grid_size']}")
    
    # Test a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
    
    print(f"After steps - obs shape: {obs.shape}")
    print("MiniGridVarEnv test passed!")