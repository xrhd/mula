<div align="center">
  <img src="docs/assets/mula.png" alt="Mula Logo" width="400"/>
</div>

# Mula 🚀

**M**achine Learning's **U**ltimate **L**earning **A**rchive

A clean, educational implementation of Machine Learning's most influential works using [JAX](https://github.com/google/jax).

## 🎯 Vision

Mula is a curated collection of seminal machine learning papers and algorithms, reimplemented from scratch in JAX. The goal is to provide:

- **Clear, readable implementations** that prioritize understanding over performance
- **Educational focus** with detailed comments and documentation
- **Modern tooling** leveraging JAX's functional approach and automatic differentiation
- **Reproducible results** matching original papers where possible

## 📚 Implementations

### Reinforcement Learning
- [x] Deep Q-Network (DQN) - *Mnih et al., 2015* ✨
- [ ] Proximal Policy Optimization (PPO) - *Schulman et al., 2017*
- [ ] Deep Deterministic Policy Gradient (DDPG) - *Lillicrap et al., 2015*
- [ ] Soft Actor-Critic (SAC) - *Haarnoja et al., 2018*

### Computer Vision
- [ ] Convolutional Neural Networks - *LeCun et al., 1989*
- [ ] ResNet - *He et al., 2015*
- [ ] Vision Transformer (ViT) - *Dosovitskiy et al., 2020*
- [ ] Diffusion Models - *Ho et al., 2020*

### Natural Language Processing
- [ ] Attention is All You Need (Transformer) - *Vaswani et al., 2017*
- [ ] BERT - *Devlin et al., 2018*
- [ ] GPT - *Radford et al., 2018*

### Foundational Works
- [ ] Backpropagation - *Rumelhart et al., 1986*
- [ ] Adam Optimizer - *Kingma & Ba, 2014*
- [ ] Batch Normalization - *Ioffe & Szegedy, 2015*
- [ ] Dropout - *Srivastava et al., 2014*

### Generative Models
- [ ] Variational Autoencoders (VAE) - *Kingma & Welling, 2013*
- [ ] Generative Adversarial Networks (GAN) - *Goodfellow et al., 2014*

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mula.git
cd mula

# Install dependencies
make install

# Or with GPU support (Apple Silicon)
make install-metal
```

See [Quick Start Guide](docs/QUICKSTART.md) for detailed setup instructions.

### Running Examples

```bash
# Train DQN on CartPole
make run-dqn

# Watch the trained agent
make run-dqn-render

# Check available commands
make help
```

### Example Code

```python
import gymnasium as gym
from mula.rl.dqn import DQNConfig, train

# Create environment
env = gym.make("CartPole-v1")

# Configure and train DQN
config = DQNConfig()
model = train(env, config, num_episodes=500)
```

## 🏗️ Project Structure

```
mula/
├── docs/              # Documentation and guides
│   ├── QUICKSTART.md      # Quick start guide
│   ├── AGENTS.md          # Guide for AI agents/contributors
│   └── JAX_METAL_SETUP.md # GPU acceleration setup
├── mula/
│   ├── rl/           # Reinforcement Learning implementations
│   │   └── dqn/      # Deep Q-Network
│   ├── cv/           # Computer Vision implementations
│   ├── nlp/          # NLP implementations
│   ├── generative/   # Generative models
│   └── utils/        # Shared utilities and helpers
├── examples/         # Example scripts and notebooks
│   └── dqn_cartpole.py
├── tests/            # Unit tests
├── Makefile          # Convenient commands
└── pyproject.toml    # Dependencies
```

## 📖 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Detailed installation and usage
- **[Agent Guide](docs/AGENTS.md)** - Development guidelines and best practices
- **[JAX Metal Setup](docs/JAX_METAL_SETUP.md)** - GPU acceleration for Apple Silicon
- **[Makefile Reference](.makerc)** - Quick command reference

## 🧪 Running Examples

```bash
# Using Makefile (recommended)
make run-dqn          # Train DQN on CartPole
make run-dqn-render   # Train and visualize
make help             # See all commands

# Direct execution
uv run python examples/dqn_cartpole.py --episodes 500 --render
```

## 🤝 Contributing

Contributions are welcome! Whether it's:

- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ New implementations
- 🧪 Additional tests

Please feel free to open an issue or submit a pull request.

### Guidelines

See **[Agent Guide](docs/AGENTS.md)** for comprehensive development guidelines.

Quick summary:
1. **Code Style**: Follow JAX idioms (pure functions, no side effects)
2. **Documentation**: Include paper references and clear docstrings
3. **Tests**: Add tests for new implementations
4. **Examples**: Provide a working example in `examples/`

## 🎓 Learning Resources

Each implementation includes:

- Links to the original paper
- Mathematical background and intuition
- Code walkthrough with inline comments
- Training tips and hyperparameters

## 📖 Why JAX?

- **Functional Programming**: Clean, composable code
- **Auto-differentiation**: Native gradient computation
- **JIT Compilation**: Performance when needed
- **Hardware Acceleration**: Seamless GPU/TPU support
- **Numerical Stability**: Built for scientific computing

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

This project stands on the shoulders of giants. We acknowledge all the researchers whose groundbreaking work made modern machine learning possible.

## 🔗 Links

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax](https://flax.readthedocs.io/en/stable/)
- [Bonsai](https://github.com/jax-ml/bonsai/tree/main)
- [JAX AI Stack](https://github.com/jax-ml/jax-ai-stack?tab=readme-ov-file#jax-ai-stack)


---

*Built with ❤️ and JAX*