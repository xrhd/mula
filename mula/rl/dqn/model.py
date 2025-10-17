"""DQN model implementation.

Deep Q-Network from Mnih et al. (2015).

Reference:
    Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).
    Human-level control through deep reinforcement learning.
    Nature, 518(7540), 529-533.
    https://www.nature.com/articles/nature14236
"""

from typing import NamedTuple

import jax.numpy as jnp
from flax import nnx


class DQNConfig(NamedTuple):
    """Configuration for DQN.
    
    Attributes:
        hidden_dims: Hidden layer dimensions for MLP.
        learning_rate: Learning rate for optimizer.
        gamma: Discount factor for future rewards.
        epsilon_start: Initial exploration rate.
        epsilon_end: Final exploration rate.
        epsilon_decay: Decay rate for epsilon.
        target_update_freq: Frequency (in steps) to update target network.
        batch_size: Batch size for training.
        buffer_size: Size of replay buffer.
    """
    hidden_dims: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    batch_size: int = 32
    buffer_size: int = 10000


class DQN(nnx.Module):
    """Deep Q-Network with MLP architecture.
    
    A simple feedforward network that maps states to Q-values for each action.
    The network consists of fully connected layers with ReLU activations.
    
    Reference:
        Mnih et al. (2015) - Human-level control through deep RL
        Paper: https://www.nature.com/articles/nature14236
    """
    
    def __init__(self, state_dim: int, num_actions: int, hidden_dims: tuple[int, ...], rngs: nnx.Rngs):
        """Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space.
            num_actions: Number of possible actions in the environment.
            hidden_dims: Tuple of hidden layer dimensions.
            rngs: Random number generator state.
        """
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims
        
        # Create layers using nnx.List for proper pytree handling
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        
        self.layers = nnx.List(layers)
        
        # Output layer
        self.output_layer = nnx.Linear(in_dim, num_actions, rngs=rngs)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the Q-network.
        
        Args:
            x: State observation, shape (batch, state_dim) or (state_dim,).
            
        Returns:
            Q-values for each action, shape (batch, num_actions) or (num_actions,).
        """
        # Pass through hidden layers with ReLU activation
        for layer in self.layers:
            x = layer(x)
            x = nnx.relu(x)
        
        # Output layer produces Q-value for each action
        q_values = self.output_layer(x)
        return q_values

