"""Deep Q-Network (DQN) implementation.

This module implements the DQN algorithm from Mnih et al. (2015).

Reference:
    Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).
    Human-level control through deep reinforcement learning.
    Nature, 518(7540), 529-533.
    https://www.nature.com/articles/nature14236
"""

from .agent import select_action, train, train_step, update_target_network
from .model import DQN, DQNConfig
from .replay import ReplayBuffer, Transition

__all__ = [
    "DQN",
    "DQNConfig",
    "ReplayBuffer",
    "Transition",
    "select_action",
    "train",
    "train_step",
    "update_target_network",
]

