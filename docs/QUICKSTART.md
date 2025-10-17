# Quick Start Guide

## First Time Setup

```bash
# Install the package in editable mode
make install

# Or view all available commands
make help
```

## Running Examples

### DQN on CartPole

```bash
# Quick run (300 episodes)
make run-dqn

# Longer training (500 episodes)
make run-dqn-long

# Watch the agent with rendering (evaluation only)
make run-dqn-render

# Train longer and show results
make run-dqn-demo

# Custom parameters
uv run python examples/dqn_cartpole.py --episodes 500 --seed 42

# With rendering during evaluation
uv run python examples/dqn_cartpole.py --episodes 500 --render

# With rendering during training (slower, for debugging)
uv run python examples/dqn_cartpole.py --episodes 100 --render-train
```

## Development Commands

```bash
# Run tests
make test

# Check code quality
make lint

# Format code
make format

# Clean build artifacts
make clean

# Verify installation
make verify
```

## Quick Start (Install + Run)

```bash
make quickstart
```

This will install the package and immediately run the DQN CartPole example.

## Manual Installation

If you prefer not to use Make:

```bash
# Install package
uv pip install -e .

# Run example
uv run python examples/dqn_cartpole.py
```

## Troubleshooting

### ModuleNotFoundError: No module named 'mula'

Run `make install` first to install the package in editable mode.

### Missing dependencies

Run `uv sync` to ensure all dependencies are installed.

### Want to start fresh?

```bash
make clean    # Remove build artifacts
make install  # Reinstall package
```

## More Resources

- **[Agent Guide](AGENTS.md)** - Development guidelines and best practices
- **[JAX Metal Setup](JAX_METAL_SETUP.md)** - GPU acceleration for Apple Silicon
- **[Makefile Reference](../.makerc)** - Quick command reference

