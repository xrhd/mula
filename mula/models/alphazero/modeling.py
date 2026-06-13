import jax
import jax.numpy as jnp
from flax import nnx


class BlockV1(nnx.Module):
    """ResNet V1 block: Conv -> BN -> ReLU -> Conv -> BN -> Residual + ReLU."""

    def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(
            num_features=num_channels,
            momentum=0.9,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(
            num_features=num_channels,
            momentum=0.9,
            rngs=rngs,
        )

    def __call__(self, x, use_running_average: bool = False):
        i = x
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=use_running_average)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=use_running_average)
        return jax.nn.relu(x + i)


class BlockV2(nnx.Module):
    """ResNet V2 block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> Residual."""

    def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
        self.bn1 = nnx.BatchNorm(
            num_features=num_channels,
            momentum=0.9,
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(
            num_features=num_channels,
            momentum=0.9,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=num_channels,
            out_features=num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x, use_running_average: bool = False):
        i = x
        x = self.bn1(x, use_running_average=use_running_average)
        x = jax.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, use_running_average=use_running_average)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        return x + i


class AZNet(nnx.Module):
    """AlphaZero NN architecture for board games.

    Args:
        num_actions: Number of possible actions (policy head output size).
        num_channels: Number of channels in the ResNet trunk.
        num_blocks: Number of ResNet blocks.
        resnet_v2: Whether to use ResNet V2 blocks.
        rngs: Flax NNX RNG manager.
    """

    def __init__(
        self,
        num_actions: int,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        block_cls = BlockV2 if resnet_v2 else BlockV1

        self.initial_conv = nnx.Conv(
            in_features=2,  # Othello observation has 2 channels
            out_features=num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        if not resnet_v2:
            self.initial_bn = nnx.BatchNorm(
                num_features=num_channels,
                momentum=0.9,
                rngs=rngs,
            )

        self.blocks = nnx.List(
            [block_cls(num_channels, rngs=rngs) for _ in range(num_blocks)]
        )

        if resnet_v2:
            self.final_bn = nnx.BatchNorm(
                num_features=num_channels,
                momentum=0.9,
                rngs=rngs,
            )

        # Policy head
        self.policy_conv = nnx.Conv(
            in_features=num_channels,
            out_features=2,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.policy_bn = nnx.BatchNorm(
            num_features=2,
            momentum=0.9,
            rngs=rngs,
        )
        self.policy_linear = nnx.Linear(
            in_features=2 * 8 * 8,
            out_features=num_actions,
            rngs=rngs,
        )

        # Value head
        self.value_conv = nnx.Conv(
            in_features=num_channels,
            out_features=1,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.value_bn = nnx.BatchNorm(
            num_features=1,
            momentum=0.9,
            rngs=rngs,
        )
        self.value_linear1 = nnx.Linear(
            in_features=1 * 8 * 8,
            out_features=num_channels,
            rngs=rngs,
        )
        self.value_linear2 = nnx.Linear(
            in_features=num_channels,
            out_features=1,
            rngs=rngs,
        )

    def __call__(self, x, use_running_average: bool = False):
        """Forward pass.

        Args:
            x: Input observation of shape (..., 8, 8, 2).
            use_running_average: If True, use stored batch norm stats.

        Returns:
            logits: Policy logits of shape (..., num_actions).
            value: Value estimate of shape (...).
        """
        x = x.astype(jnp.float32)
        x = self.initial_conv(x)

        if not self.resnet_v2:
            x = self.initial_bn(x, use_running_average=use_running_average)
            x = jax.nn.relu(x)

        for block in self.blocks:
            x = block(x, use_running_average=use_running_average)

        if self.resnet_v2:
            x = self.final_bn(x, use_running_average=use_running_average)
            x = jax.nn.relu(x)

        # Policy head
        logits = self.policy_conv(x)
        logits = self.policy_bn(logits, use_running_average=use_running_average)
        logits = jax.nn.relu(logits)
        logits = logits.reshape(logits.shape[:-3] + (-1,))
        logits = self.policy_linear(logits)

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v, use_running_average=use_running_average)
        v = jax.nn.relu(v)
        v = v.reshape(v.shape[:-3] + (-1,))
        v = self.value_linear1(v)
        v = jax.nn.relu(v)
        v = self.value_linear2(v)
        v = jnp.tanh(v)
        v = v.reshape(v.shape[:-1])

        return logits, v
