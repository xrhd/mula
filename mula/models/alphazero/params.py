import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """AlphaZero model and training configuration for Othello."""

    env_id: str = "othello"
    seed: int = 0
    max_num_iters: int = 400

    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True

    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256

    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001

    # eval params
    eval_interval: int = 5

    # checkpoint/plot params
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"

    def __post_init__(self):
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.selfplay_batch_size <= 0:
            raise ValueError("selfplay_batch_size must be positive")
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if self.max_num_steps <= 0:
            raise ValueError("max_num_steps must be positive")
        if self.training_batch_size <= 0:
            raise ValueError("training_batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
