"""Experience replay buffer for DQN.

Implements a simple circular buffer to store and sample transitions.

Reference:
    Mnih et al. (2015) - Section "Experience Replay"
    https://www.nature.com/articles/nature14236
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Transition(NamedTuple):
    """A single transition in the environment.
    
    Attributes:
        state: Current state observation.
        action: Action taken.
        reward: Reward received.
        next_state: Next state observation.
        done: Whether episode ended.
    """
    state: jnp.ndarray
    action: int
    reward: float
    next_state: jnp.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.
    
    Implements experience replay mechanism from DQN paper. Stores transitions
    and allows sampling random minibatches for training.
    
    The buffer uses a circular queue structure that overwrites old experiences
    when full, following the FIFO principle.
    
    Reference:
        Lin, L. J. (1992). Self-improving reactive agents based on reinforcement
        learning, planning and teaching. Machine learning, 8(3-4), 293-321.
    """
    
    def __init__(self, capacity: int):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self.position = 0
    
    def push(
        self,
        state: jnp.ndarray,
        action: int,
        reward: float,
        next_state: jnp.ndarray,
        done: bool,
    ) -> None:
        """Save a transition to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode ended.
        """
        transition = Transition(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Overwrite oldest transition (circular buffer)
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, key: jax.random.PRNGKey) -> Transition:
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            key: JAX random key for sampling.
            
        Returns:
            Batch of transitions with each field stacked into arrays.
        """
        # Sample random indices
        indices = jax.random.choice(key, len(self.buffer), shape=(batch_size,), replace=False)
        
        # Gather transitions
        batch = [self.buffer[int(idx)] for idx in indices]
        
        # Stack into batched arrays
        states = jnp.stack([t.state for t in batch])
        actions = jnp.array([t.action for t in batch])
        rewards = jnp.array([t.reward for t in batch])
        next_states = jnp.stack([t.next_state for t in batch])
        dones = jnp.array([t.done for t in batch])
        
        return Transition(states, actions, rewards, next_states, dones)
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)

