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
- [ ] Deep Q-Network (DQN) - *Mnih et al., 2015*
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

Using uv (recommended):
```bash
# Clone the repository
git clone https://github.com/yourusername/mula.git
cd mula

# Install dependencies
uv sync
```

Using pip:
```bash
pip install -e .
```

### Quick Start

```python
import jax
import jax.numpy as jnp
from mula import DQN

# Initialize your model
model = DQN(state_dim=8, action_dim=4)

# Train on your environment
# ... (implementation specific)
```

## 🏗️ Project Structure

```
mula/
├── docs/              # Documentation and paper summaries
├── mula/
│   ├── rl/           # Reinforcement Learning implementations
│   ├── cv/           # Computer Vision implementations
│   ├── nlp/          # NLP implementations
│   ├── generative/   # Generative models
│   └── utils/        # Shared utilities and helpers
├── examples/         # Example scripts and notebooks
├── tests/            # Unit tests
└── main.py          # Entry point
```

## 🧪 Running Examples

```bash
# Run a specific implementation
uv run python -m mula.rl.dqn --env CartPole-v1

# Or use the main entry point
uv run python main.py
```

## 🤝 Contributing

Contributions are welcome! Whether it's:

- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ New implementations
- 🧪 Additional tests

Please feel free to open an issue or submit a pull request.

### Guidelines

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