import gymnasium as gym  # Note: 'gymnasium' not 'gym'

def render():
    # Environment creation with render mode specified upfront
    env = gym.make("LunarLander-v3", render_mode="human")

    # Reset with seed parameter
    observation, info = env.reset(seed=123, options={})

    print(observation.shape)

    # Training loop with terminated/truncated distinction
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation)

        # Episode ends if either terminated OR truncated
        done = terminated or truncated

    env.close()

def main():
    print("Hello from mula!")
    render()


if __name__ == "__main__":
    main()
