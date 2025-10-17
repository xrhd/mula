"""Train DQN on CartPole-v1.

This example demonstrates training a DQN agent on the CartPole environment.
CartPole is a simple control task where the agent must balance a pole on a cart.

Usage:
    # Train without rendering
    uv run python examples/dqn_cartpole.py --episodes 500
    
    # Train and render evaluation
    uv run python examples/dqn_cartpole.py --episodes 500 --render
    
    # Render during training (slower)
    uv run python examples/dqn_cartpole.py --episodes 100 --render-train
    
    # Custom seed
    uv run python examples/dqn_cartpole.py --episodes 500 --seed 42 --render

Reference:
    Mnih et al. (2015) - Human-level control through deep RL
    https://www.nature.com/articles/nature14236
"""

import argparse

import gymnasium as gym
import jax

from mula.rl.dqn import DQNConfig, train


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train DQN on CartPole-v1")
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="Number of episodes to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress every N episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation",
    )
    parser.add_argument(
        "--render-train",
        action="store_true",
        help="Render the environment during training (slow)",
    )
    args = parser.parse_args()
    
    # Print JAX backend info
    print("=" * 60)
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"JAX Devices: {jax.devices()}")
    print("=" * 60)
    
    # Create environment
    print(f"\nCreating environment: {args.env}")
    render_mode = "human" if args.render_train else None
    env = gym.make(args.env, render_mode=render_mode)
    
    # Create DQN config
    # Using simple hyperparameters optimized for CartPole
    config = DQNConfig(
        hidden_dims=(64, 64),
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=100,
        batch_size=32,
        buffer_size=10000,
    )
    
    print("\nDQN Configuration:")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Epsilon: {config.epsilon_start} -> {config.epsilon_end}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Buffer size: {config.buffer_size}")
    print(f"\nTraining for {args.episodes} episodes...\n")
    
    # Train agent
    model = train(
        env=env,
        config=config,
        num_episodes=args.episodes,
        seed=args.seed,
        log_every=args.log_every,
    )
    
    print("\nTraining complete!")
    env.close()
    
    # Evaluate trained agent
    print("\nEvaluating trained agent...")
    
    # Create evaluation environment with optional rendering
    eval_render_mode = "human" if args.render else None
    eval_env = gym.make(args.env, render_mode=eval_render_mode)
    
    if args.render:
        print("🎬 Rendering enabled - watch the agent play!\n")
    
    eval_episodes = 10
    total_reward = 0.0
    
    for i in range(eval_episodes):
        obs, _ = eval_env.reset(seed=args.seed + 1000 + i)
        episode_reward = 0.0
        done = False
        
        while not done:
            # Use greedy policy (epsilon=0)
            q_values = model(obs[None, ...])[0]
            action = int(q_values.argmax())
            
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
        print(f"  Episode {i + 1}: {episode_reward:.2f}")
    
    avg_reward = total_reward / eval_episodes
    print(f"\nAverage reward over {eval_episodes} episodes: {avg_reward:.2f}")
    
    # CartPole is considered solved when avg reward > 195
    if avg_reward >= 195:
        print("✓ CartPole solved!")
    else:
        print("✗ Not quite solved yet (need 195+)")
    
    eval_env.close()


if __name__ == "__main__":
    main()

