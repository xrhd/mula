# JAX Metal (GPU) Setup for Apple Silicon

This guide shows how to enable GPU acceleration on Apple Silicon Macs using JAX Metal.

## Current Status

Check your current JAX backend:

```bash
uv run python -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"
```

## Enable Metal/MPS GPU Acceleration

### 1. Install JAX Metal

```bash
# Install jax-metal package
uv pip install jax-metal

# Verify installation
uv run python -c "import jax; print('Devices:', jax.devices())"
```

### 2. Set Environment Variables

You can set these in your shell or create a `.env` file:

```bash
# Enable Metal backend
export JAX_PLATFORMS=metal

# Optional: Enable 64-bit precision
export JAX_ENABLE_X64=1

# Optional: Debug JIT compilation
export JAX_LOG_COMPILES=1
```

Or create a `.env` file in the project root:

```bash
# .env
JAX_PLATFORMS=metal
```

Then load it before running:

```bash
# Using direnv (recommended)
direnv allow

# Or manually
export $(cat .env | xargs)
uv run python examples/dqn_cartpole.py
```

### 3. Update pyproject.toml (Optional)

Add to dependencies:

```toml
[project]
dependencies = [
    # ... existing deps
    "jax-metal>=0.1.0; platform_machine == 'arm64'",  # Apple Silicon only
]
```

## Verify Metal is Working

Run this test script:

```python
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Default backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# Test GPU computation
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
print(f"Computed on: {y.device()}")
```

Expected output with Metal:
```
Default backend: METAL
Available devices: [METAL(id=0)]
Computed on: METAL(id=0)
```

## Makefile Commands

We've added convenience commands:

```bash
# Check current JAX configuration
make check-jax

# Install with Metal support
make install-metal

# Run DQN with Metal
make run-dqn
```

## Performance Notes

- **Metal speedup**: 2-5x faster than CPU for neural networks
- **First run**: Slower due to JIT compilation
- **Small models**: May not see speedup (overhead dominates)
- **CartPole**: Small network, speedup will be modest

## Known Issues

### Random Number Generation Limitations
JAX Metal has fundamental limitations with random number generation. If you encounter errors like:
```
UNIMPLEMENTED: default_memory_space is not supported.
```

This occurs when trying to create JAX random keys (`jax.random.PRNGKey`) or perform random operations. This is a known limitation of the experimental JAX Metal plugin that affects:

- `jax.random.PRNGKey()`
- `jax.random.split()`
- `jax.random.uniform()`
- `jax.random.normal()`
- Any Flax NNX operations that require random keys

### Workarounds

**Option 1: Use CPU for Random Operations**
```python
# Force CPU for random operations
cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    key = jax.random.PRNGKey(seed)
    # ... rest of random operations
```

**Option 2: Use CPU for Training**
```bash
# Run with CPU backend
JAX_PLATFORMS=cpu python examples/dqn_cartpole.py
```

**Option 3: Use Flax Linen Instead of NNX**
The older Flax Linen API may have better compatibility with JAX Metal, though this requires significant code changes.

## Troubleshooting

### "No Metal devices found"

- Make sure you're on an Apple Silicon Mac (M1/M2/M3)
- Verify jax-metal is installed: `uv pip list | grep metal`
- Check macOS version (requires macOS 12.3+)

### "Metal compilation failed"

```bash
# Fall back to CPU
export JAX_PLATFORMS=cpu
```

### Memory issues

```bash
# Disable memory preallocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## Alternative: CPU-Only

If you don't need GPU acceleration:

```bash
# Force CPU backend
export JAX_PLATFORMS=cpu
```

This is useful for:
- Debugging (better error messages)
- Consistency across platforms
- When Metal has issues

## References

- [JAX Metal Plugin](https://github.com/google/jax/tree/main/jax_plugins/metal_plugin)
- [JAX Device Management](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices)
- [Apple Metal](https://developer.apple.com/metal/)

