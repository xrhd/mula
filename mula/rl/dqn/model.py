"""DQN model implementation.

Deep Q-Network from Mnih et al. (2015).

Reference:
    Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015).
    Human-level control through deep reinforcement learning.
    Nature, 518(7540), 529-533.
    https://www.nature.com/articles/nature14236
"""

from typing import NamedTuple

import jax
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
    
    def __init__(self, state_dim: int, num_actions: int, hidden_dims: tuple[int, ...], key: jax.random.PRNGKey):
        """Initialize DQN network.
        
        Args:
            state_dim: Dimension of state space.
            num_actions: Number of possible actions in the environment.
            hidden_dims: Tuple of hidden layer dimensions.
            key: Random number generator key.
        """
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims
        
        # Force CPU for random operations
        cpu_device = jax.devices('cpu')[0]
        
        # Create individual layers
        self.layer1 = None
        self.layer2 = None
        self.output_layer = None
        
        in_dim = state_dim
        layer_idx = 0
        
        for hidden_dim in hidden_dims:
            key, layer_key = jax.random.split(key)
            layer_key = jax.device_put(layer_key, cpu_device)
            
            if layer_idx == 0:
                self.layer1 = nnx.Linear(in_dim, hidden_dim, rngs=nnx.Rngs(layer_key))
            elif layer_idx == 1:
                self.layer2 = nnx.Linear(in_dim, hidden_dim, rngs=nnx.Rngs(layer_key))
            
            in_dim = hidden_dim
            layer_idx += 1
        
        # Output layer
        key, output_key = jax.random.split(key)
        output_key = jax.device_put(output_key, cpu_device)
        self.output_layer = nnx.Linear(in_dim, num_actions, rngs=nnx.Rngs(output_key))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the Q-network.
        
        Args:
            x: State observation, shape (batch, state_dim) or (state_dim,).
            
        Returns:
            Q-values for each action, shape (batch, num_actions) or (num_actions,).
        """
        # Pass through hidden layers with ReLU activation
        if self.layer1 is not None:
            x = self.layer1(x)
            x = nnx.relu(x)
        
        if self.layer2 is not None:
            x = self.layer2(x)
            x = nnx.relu(x)
        
        # Output layer produces Q-value for each action
        q_values = self.output_layer(x)
        return q_values

