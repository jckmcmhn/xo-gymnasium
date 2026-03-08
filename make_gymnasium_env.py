import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

import gymnasium as gym
from xo import XO, XOAgent
from hyperparameters import learning_rate, n_episodes, start_epsilon, epsilon_decay, final_epsilon

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level="INFO")


def test_agent(agent, env, num_episodes=100):
    """Test agent performance without learning or exploration."""
    # Based on https://gymnasium.farama.org/introduction/train_agent/#testing-your-trained-agent
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_draw_rate = np.mean(np.array(total_rewards) > 0)
    win_rate = np.mean(np.array(total_rewards) == 2)
    draw_rate = np.mean(np.array(total_rewards) == 1)
    loss_rate = np.mean(np.array(total_rewards) < 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win or Draw Rate: {win_draw_rate:.1%}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Draw Rate: {draw_rate:.1%}")
    print(f"Bad play Rate: {loss_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")
    print("\n")
    print(f"Learning rate was {learning_rate}")
    print(f"N episodes was {n_episodes}")
    print(f"Start epsilon was {start_epsilon}")
    print(f"Epsilon decay was {epsilon_decay}")
    print(f"Final epsilon was {final_epsilon}")

    
# Register the environment so we can create it with gym.make()
gym.register(
    id="gymnasium_env/xo-v0",
    entry_point=XO,
    max_episode_steps=9,  # Prevent infinite episodes
)

env = gym.make("gymnasium_env/xo-v0", opponent_logic = "random") # opponent_logic is passed through as **kwargs https://gymnasium.farama.org/api/registry/#gymnasium.make

agent = XOAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


for episode in tqdm(range(n_episodes)):
    # Start a new hand
    logging.debug("Resetting")
    obs, info = env.reset()
    done = False

    # Play one complete game
    while not done:

        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Learn from this experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()


# Test your agent
print("\nTest the agent in the environment in which it was trained")
test_agent(agent, env, 500)

# Create a new version of the environment where the opponent plays competitively (does not miss winning/blocking moves)
print("\nTest the agent against a more difficult environment")
env = gym.make("gymnasium_env/xo-v0", opponent_logic = "competitive")

test_agent(agent, env, 2000)

df = pd.DataFrame(agent.q_values).transpose()
print(len(df))
df.to_csv("outfile.csv")