"""
Environment wrappers with controllable variation for generalization research.
"""

from .cartpole_var import CartPoleVarEnv, CartPoleVarVecEnv, make_cartpole_train_test_envs
from .minigrid_var import MiniGridVarEnv, MiniGridVarVecEnv, make_minigrid_train_test_envs
from .coinrun_var import CoinRunVarEnv, CoinRunVarVecEnv, make_coinrun_train_test_envs
from .minatar_var import MinAtarVarEnv, MinAtarVarVecEnv, make_minatar_train_test_envs

__all__ = [
    "CartPoleVarEnv",
    "CartPoleVarVecEnv",
    "make_cartpole_train_test_envs",
    "MiniGridVarEnv",
    "MiniGridVarVecEnv",
    "make_minigrid_train_test_envs",
    "CoinRunVarEnv",
    "CoinRunVarVecEnv",
    "make_coinrun_train_test_envs",
    "MinAtarVarEnv",
    "MinAtarVarVecEnv",
    "make_minatar_train_test_envs",
]