from __future__ import annotations

import pickle

from tqdm import tqdm

import gymnasium as gym


# Let's start by creating the blackjack environment.
# Note: We are going to follow the rules from Sutton & Barto.
# Other versions of the game can be found below for you to experiment.
from BlackJackAgent import BlackjackAgent
from FiniteDeck.FiniteDeckBlackjackEnv import FiniteDeckBlackjackEnv
from Visualize.Visualize_Grid import visualize_grid
from Visualize.Visualize_Rate import visualize_rate

train = True
if train:
    env = FiniteDeckBlackjackEnv(number_of_decks=2)

    # reset the environment to get the first observation
    done = False
    observation = env.reset()



    # hyperparameters
    learning_rate = 0.0005
    n_episodes = 2_000_000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1

    agent = BlackjackAgent(
        env= env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
    with open("environment_agent_finite_deck.pkl", "wb") as f:
        pickle.dump(env, f)
        pickle.dump(agent, f)
else:
    with open("environment_agent_finite_deck.pkl", "rb") as f:
        env = pickle.load(f)
        agent = pickle.load(f)



visualize_rate(env = env, agent = agent, rolling_length=10000)
visualize_grid(agent)