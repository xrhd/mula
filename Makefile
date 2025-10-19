.PHONY: help install dev clean test lint format run-dqn

# Default target
help:
	@echo "Mula - Machine Learning's Ultimate Learning Archive"
	@echo ""
	@echo "Available commands:"
	@echo "  make install          Install package in editable mode"
	@echo "  make install-metal    Install with Metal/GPU support (Apple Silicon)"
	@echo "  make dev              Install package + dev dependencies"
	@echo "  make clean            Remove build artifacts and cache"
	@echo "  make test             Run tests with pytest"
	@echo "  make lint             Run linter (ruff)"
	@echo "  make format           Format code with ruff"
	@echo "  make verify           Verify installation"
	@echo "  make check-jax        Check JAX backend configuration"
	@echo ""
	@echo "Examples:"
	@echo "  make run-dqn          Run DQN CartPole example (300 episodes)"
	@echo "  make run-dqn-long     Run DQN CartPole example (500 episodes)"
	@echo "  make run-dqn-render   Run DQN with visualization"
	@echo "  make run-dqn-video    Train and save videos"
	@echo "  make run-dqn-demo     Train longer + show results"
	@echo ""
	@echo "Getting started:"
	@echo "  make install          # First time setup"
	@echo "  make check-jax        # Check if GPU is available"
	@echo "  make run-dqn          # Train DQN on CartPole"

# Install package in editable mode
install:
	@echo "📦 Installing mula in editable mode..."
	uv pip install -e .
	@echo "✓ Installation complete!"

# Install with dev dependencies
dev:
	@echo "📦 Installing mula with dev dependencies..."
	uv pip install -e ".[dev]"
	@echo "✓ Development setup complete!"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete!"

# Run tests
test:
	@echo "🧪 Running tests..."
	uv run pytest tests/ -v

# Run linter
lint:
	@echo "🔍 Running linter..."
	uv run ruff check mula/ examples/ tests/

# Format code
format:
	@echo "✨ Formatting code..."
	uv run ruff check --fix mula/ examples/ tests/
	uv run ruff format mula/ examples/ tests/

# Run DQN CartPole example
run-dqn:
	@echo "🎮 Training DQN on CartPole-v1..."
	uv run python examples/dqn_cartpole.py

# Run DQN with custom episodes
run-dqn-long:
	@echo "🎮 Training DQN on CartPole-v1 (500 episodes)..."
	uv run python examples/dqn_cartpole.py --episodes 500

# Run DQN with rendering
run-dqn-render:
	@echo "🎬 Training DQN on CartPole-v1 with rendering..."
	uv run python examples/dqn_cartpole.py --episodes 300 --render

# Train longer and render evaluation
run-dqn-demo:
	@echo "🎬 Training DQN (500 episodes) and showing results..."
	uv run python examples/dqn_cartpole.py --episodes 500 --render

# Train and save videos (uses CPU for compatibility)
run-dqn-video:
	@echo "📹 Training DQN and saving videos..."
	uv run python examples/dqn_cartpole.py --episodes 120 --save-video

quickstart: install run-dqn

# Verify installation
verify:
	@echo "🔍 Verifying installation..."
	@uv run python -c "import mula; print('✓ mula package found')"
	@uv run python -c "import jax; print('✓ jax found')"
	@uv run python -c "import flax; print('✓ flax found')"
	@uv run python -c "import gymnasium; print('✓ gymnasium found')"
	@echo "✓ All core dependencies verified!"

# Check JAX configuration
check-jax:
	@echo "🔍 Checking JAX configuration..."
	@uv run python -c "import jax; print('JAX version:', jax.__version__); print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"

# Install with Metal support (Apple Silicon)
install-metal:
	@echo "📦 Installing mula with Metal (GPU) support..."
	uv pip install -e ".[metal]"
	@echo "✓ Metal support installed!"
	@echo ""
	@echo "🚀 To enable Metal GPU acceleration:"
	@echo "   export JAX_PLATFORMS=metal"
	@echo ""
	@echo "📖 See docs/JAX_METAL_SETUP.md for more details"
	@echo ""
	@make check-jax

