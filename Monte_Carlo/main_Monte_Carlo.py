from __future__ import annotations
import pickle
from tqdm import tqdm
from BlackJackAgent import BlackjackAgent
from typing import Tuple
import numpy as np
from collections import deque
from Visualize.Visualize_Rate import visualize_rate
from Visualize.Visualize_Grid import visualize_grid
import gymnasium as gym
from BlackJackAgentMonteCarlo import BlackjackAgent
train = True

if train:

    learning_rate = 0.001
    n_episodes = 1_000_00
    start_epsilon = 1
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    env = gym.make("Blackjack-v1", sab=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)  # Move the wrapper here
    done = False
    observation, info = env.reset()



    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes)):
        episode_data = []
        obs, _ = agent.env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = agent.env.step(action)
            episode_data.append((obs, action, reward, terminated))
            obs = next_obs
            done = terminated

        agent.update_q_values_monte_carlo(episode_data)

        agent.learning_rate_decay(episode)
        agent.decay_epsilon()

    with open("environment_agent_monte_carlo.pkl", "wb") as f:
        pickle.dump(env, f)
        pickle.dump(agent, f)
else:
    with open("environment_agent_monte_carlo.pkl", "rb") as f:
        env = pickle.load(f)
        agent = pickle.load(f)

visualize_rate(learning_rate, env=env, agent=agent, rolling_length=5000)
visualize_grid(agent)

