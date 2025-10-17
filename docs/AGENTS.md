# Agent Guide for Mula

This document provides context and guidelines for AI agents working on the Mula codebase.

## Project Overview

**Mula** (Machine Learning's Ultimate Learning Archive) is an educational repository implementing influential machine learning papers from scratch using JAX. The focus is on clarity, educational value, and functional programming principles.

### Core Philosophy

1. **Education First**: Code should be readable and well-documented, prioritizing understanding over performance optimization
2. **Functional Approach**: Leverage JAX's functional programming paradigm (pure functions, no side effects)
3. **Paper Fidelity**: Implementations should match the original papers as closely as possible while being idiomatic JAX
4. **Self-Contained**: Each implementation should be relatively standalone with minimal dependencies

## Project Structure

```
mula/
├── mula/
│   ├── assets/         # Logo and visual assets
│   ├── rl/            # Reinforcement Learning implementations
│   ├── cv/            # Computer Vision implementations
│   ├── nlp/           # Natural Language Processing implementations
│   ├── generative/    # Generative models (VAE, GAN, Diffusion)
│   └── utils/         # Shared utilities (optimizers, losses, etc.)
├── examples/          # Runnable examples and demos
├── tests/             # Unit tests
├── docs/              # Documentation and paper summaries
└── pyproject.toml     # Project dependencies
```

## Technology Stack

- **JAX**: Core framework for automatic differentiation and JIT compilation
- **Flax**: Neural network library (use for model definitions)
- **Optax**: Gradient processing and optimization
- **Gymnasium**: RL environments
- **Chex**: Testing utilities for JAX code
- **uv**: Package manager and dependency management

## Development Guidelines

### 1. JAX Best Practices

**Pure Functions**: All core logic should be pure functions
```python
# Good
def forward(params, x):
    return jnp.dot(x, params['w']) + params['b']

# Bad (stateful)
class Model:
    def __init__(self):
        self.params = {}
    def forward(self, x):
        self.cache = x  # Side effect!
        return x
```

**Use pytrees**: Organize parameters as nested dictionaries/tuples
```python
params = {
    'encoder': {'w': w_enc, 'b': b_enc},
    'decoder': {'w': w_dec, 'b': b_dec}
}
```

**JIT Compilation**: Mark performance-critical functions
```python
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

**Vectorization**: Use `vmap` instead of loops
```python
# Good
batch_predictions = jax.vmap(predict, in_axes=(None, 0))(params, batch)

# Avoid
predictions = [predict(params, x) for x in batch]
```

### 2. Code Organization

**File Structure**: Each implementation should have:
```
mula/rl/dqn/
├── __init__.py       # Public API
├── model.py          # Network architecture
├── agent.py          # Training logic
├── replay.py         # Experience replay buffer
└── config.py         # Hyperparameters
```

**Module Pattern**:
```python
# model.py
from typing import NamedTuple
import jax.numpy as jnp
from flax import linen as nn

class DQNConfig(NamedTuple):
    """Configuration for DQN."""
    hidden_dims: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    gamma: float = 0.99

class DQN(nn.Module):
    """Deep Q-Network implementation.
    
    Reference: Mnih et al. (2015) - Human-level control through deep RL
    Paper: https://www.nature.com/articles/nature14236
    """
    config: DQNConfig
    
    @nn.compact
    def __call__(self, x):
        # Implementation with detailed comments
        pass
```

### 3. Documentation Standards

**Docstrings**: Use Google-style docstrings
```python
def compute_td_error(q_values, actions, rewards, next_q_values, gamma):
    """Compute temporal difference error.
    
    Implements the TD error from Bellman equation:
    δ = r + γ·max(Q(s',a')) - Q(s,a)
    
    Args:
        q_values: Current Q-values, shape (batch, num_actions)
        actions: Selected actions, shape (batch,)
        rewards: Immediate rewards, shape (batch,)
        next_q_values: Next state Q-values, shape (batch, num_actions)
        gamma: Discount factor
        
    Returns:
        TD errors, shape (batch,)
        
    Reference:
        Sutton & Barto (2018), Section 6.1
    """
```

**Paper References**: Always include:
- Paper title and authors
- Publication year
- Link to paper (ArXiv or official)
- Key equations or algorithms being implemented

**Inline Comments**: Explain the "why" not the "what"
```python
# Use double DQN to reduce overestimation bias (van Hasselt et al., 2015)
target_actions = jnp.argmax(online_q_values, axis=-1)
target_q = next_q_values[jnp.arange(batch_size), target_actions]
```

### 4. Testing

**Unit Tests**: Test individual components
```python
# tests/rl/test_dqn.py
import jax
import jax.numpy as jnp
from mula.rl.dqn import DQN, DQNConfig

def test_dqn_forward():
    """Test DQN forward pass."""
    config = DQNConfig(hidden_dims=(32, 32))
    model = DQN(config)
    
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 4))
    params = model.init(key, x)
    
    output = model.apply(params, x)
    assert output.shape == (1, 2)  # Assuming 2 actions
```

**Integration Tests**: Test training loops
```python
def test_dqn_training():
    """Test that DQN can learn on a simple environment."""
    # Test on CartPole with deterministic seed
    # Should achieve > 195 reward within N episodes
```

### 5. Example Scripts

Each implementation should have a runnable example:
```python
# examples/dqn_cartpole.py
"""
Train DQN on CartPole-v1.

Usage:
    uv run python examples/dqn_cartpole.py --episodes 500
"""
import argparse
from mula.rl.dqn import DQN, train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=500)
    args = parser.parse_args()
    
    # Clear training code with progress logging
    train(args.env, num_episodes=args.episodes)

if __name__ == '__main__':
    main()
```

## Implementation Checklist

When implementing a new paper:

- [ ] Create module directory structure
- [ ] Implement core model/algorithm with Flax
- [ ] Add configuration dataclass
- [ ] Write comprehensive docstrings with paper references
- [ ] Add unit tests for key components
- [ ] Create runnable example in `examples/`
- [ ] Document in `docs/` with paper summary
- [ ] Update main README checklist
- [ ] Verify no linter errors
- [ ] Ensure code works with JIT compilation

## Common Patterns

### Random Number Generation
```python
# Always split keys explicitly
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
params = init_fn(subkey, input_shape)
```

### Gradient Computation
```python
# Use value_and_grad for efficiency
loss_fn = lambda p: compute_loss(p, batch)
loss_value, grads = jax.value_and_grad(loss_fn)(params)
```

### Checkpointing
```python
# Use orbax for checkpointing (when needed)
from orbax.checkpoint import PyTreeCheckpointer
checkpointer = PyTreeCheckpointer()
checkpointer.save(path, params)
```

## What NOT to Do

❌ **Don't** use mutable state (class attributes, global variables)
❌ **Don't** use PyTorch/TensorFlow patterns (e.g., `.backward()`, `model.train()`)
❌ **Don't** optimize prematurely - clarity first
❌ **Don't** add unnecessary dependencies
❌ **Don't** skip documentation or paper references
❌ **Don't** commit without running tests
❌ **Don't** use `.numpy()` conversions in hot paths

## What TO Do

✅ **Do** use pure functions and functional transformations
✅ **Do** leverage JAX's `jit`, `vmap`, `grad` effectively
✅ **Do** include paper references and mathematical context
✅ **Do** write clear, educational code with comments
✅ **Do** add type hints for clarity
✅ **Do** test with different random seeds
✅ **Do** provide reproducible results with seed management
✅ **Do** follow the existing code style in the repo

## Debugging Tips

1. **JIT Issues**: Add `@jax.disable_jit()` decorator temporarily to get better error messages
2. **Shape Errors**: Use `chex.assert_shape()` liberally during development
3. **NaN/Inf**: Use `jax.debug.print()` or `jax.debug.callback()` to inspect values
4. **Performance**: Use `jax.profiler` to identify bottlenecks

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [JAX AI Stack](https://github.com/jax-ml/jax-ai-stack)

## Questions?

When implementing new features:
1. Check existing implementations for patterns
2. Review the paper thoroughly
3. Start with a simple, working version
4. Iterate to match paper exactly
5. Add tests and documentation

---

Remember: Mula is about learning and education. Make the code a joy to read and learn from! 🐴💻

