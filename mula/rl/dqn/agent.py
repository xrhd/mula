"""DQN agent with training logic.

Implements the DQN algorithm including action selection, training step,
and target network updates.

Reference:
    Mnih et al. (2015) - Algorithm 1: Deep Q-learning with Experience Replay
    https://www.nature.com/articles/nature14236
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .model import DQN, DQNConfig
from .replay import ReplayBuffer, Transition


def select_action(
    model: DQN,
    observation: jnp.ndarray,
    epsilon: float,
    key: jax.random.PRNGKey,
) -> int:
    """Select action using epsilon-greedy policy.
    
    With probability epsilon, select a random action (exploration).
    Otherwise, select the action with highest Q-value (exploitation).
    
    Args:
        model: DQN model.
        observation: Current state observation.
        epsilon: Exploration rate.
        key: JAX random key.
        
    Returns:
        Selected action index.
    """
    key_explore, key_action = jax.random.split(key)
    
    # Compute Q-values for current state
    q_values = model(observation[None, ...])[0]
    
    # Epsilon-greedy action selection
    num_actions = q_values.shape[0]
    explore = jax.random.uniform(key_explore) < epsilon
    
    # Random action for exploration
    random_action = jax.random.choice(key_action, num_actions)
    
    # Greedy action for exploitation
    greedy_action = jnp.argmax(q_values)
    
    return int(jax.lax.select(explore, random_action, greedy_action))


@nnx.jit
def train_step(
    model: DQN,
    target_model: DQN,
    optimizer: nnx.Optimizer,
    batch: Transition,
    gamma: float,
) -> float:
    """Perform a single training step.
    
    Computes the temporal difference error and updates the Q-network
    using gradient descent.
    
    The loss is the mean squared Bellman error:
        L = E[(r + γ·max Q_target(s', a') - Q(s, a))²]
    
    Args:
        model: Online Q-network.
        target_model: Target Q-network.
        optimizer: Optimizer for updating model parameters.
        batch: Batch of transitions from replay buffer.
        gamma: Discount factor.
        
    Returns:
        Loss value.
        
    Reference:
        Mnih et al. (2015) - Equation 1
    """
    def loss_fn(model: DQN):
        # Compute Q-values for current states
        q_values = model(batch.state)
        
        # Get Q-values for actions taken (gather along action dimension)
        batch_indices = jnp.arange(q_values.shape[0])
        q_action = q_values[batch_indices, batch.action]
        
        # Compute target Q-values using target network
        # For terminal states, target is just the reward
        next_q_values = target_model(batch.next_state)
        next_q_max = jnp.max(next_q_values, axis=1)
        
        # Bellman target: r + γ·max Q_target(s', a') if not done, else r
        target = batch.reward + gamma * next_q_max * (1 - batch.done)
        
        # Mean squared TD error
        td_error = q_action - target
        loss = jnp.mean(td_error ** 2)
        
        return loss
    
    # Compute loss and gradients, then update
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    
    return loss


def update_target_network(model: DQN, target_model: DQN) -> None:
    """Update target network with current online network parameters.
    
    Args:
        model: Online Q-network.
        target_model: Target Q-network to update.
    """
    # Copy parameters from online network to target network
    nnx.update(target_model, nnx.state(model))


def train(
    env,
    config: DQNConfig,
    num_episodes: int,
    seed: int = 0,
    log_every: int = 10,
) -> DQN:
    """Train DQN agent on an environment.
    
    Implements the full DQN training loop with experience replay and
    target network updates.
    
    Args:
        env: Gymnasium environment.
        config: DQN configuration.
        num_episodes: Number of episodes to train.
        seed: Random seed for reproducibility.
        log_every: Frequency to log progress.
        
    Returns:
        Trained DQN model.
        
    Reference:
        Mnih et al. (2015) - Algorithm 1
    """
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # Initialize random number generator
    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)
    key, target_key = jax.random.split(key)
    
    # Create online and target networks
    model = DQN(state_dim, num_actions, config.hidden_dims, key=model_key)
    target_model = DQN(state_dim, num_actions, config.hidden_dims, key=target_key)
    
    # JAX will automatically use Metal for computations when available
    # No need to manually move the model objects
    
    # Copy online network parameters to target network
    nnx.update(target_model, nnx.state(model))
    
    # Create optimizer
    optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate))
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(config.buffer_size)
    
    # Training loop
    key = jax.random.PRNGKey(seed)
    epsilon = config.epsilon_start
    step_count = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False
        
        while not done:
            # Select action
            key, action_key = jax.random.split(key)
            action = select_action(model, obs, epsilon, action_key)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition
            replay_buffer.push(obs, action, reward, next_obs, done)
            
            # Train if enough samples
            if len(replay_buffer) >= config.batch_size:
                key, sample_key = jax.random.split(key)
                batch = replay_buffer.sample(config.batch_size, sample_key)
                train_step(model, target_model, optimizer, batch, config.gamma)
                
                step_count += 1
                
                # Update target network periodically
                if step_count % config.target_update_freq == 0:
                    update_target_network(model, target_model)
            
            obs = next_obs
        
        # Decay epsilon
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        
        # Log progress
        if (episode + 1) % log_every == 0:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}, "
                  f"Buffer: {len(replay_buffer)}")
    
    return model

