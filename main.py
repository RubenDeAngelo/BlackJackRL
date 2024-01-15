from __future__ import annotations

import pickle
from typing import List, Tuple

from tqdm import tqdm

import gymnasium as gym
import numpy as np
from Visualize.Visualize_Grid import visualize_grid
from Visualize.Visualize_Rate import visualize_rate
from BlackJackAgent import BlackjackAgent  # Assuming you have this module

def main(
    train: bool = False,
    learning_rate: float = 0.001,
    n_episodes: int = 1_000_00,
    start_epsilon: float = 1,
    final_epsilon: float = 0.1,
    tracked_states: List[Tuple[int, int, bool]] = None
):
    epsilon_decay: float = start_epsilon / (n_episodes / 2)
    q_values_history = []  # List to store Q-values for tracked states

    if train:
        env = gym.make("Blackjack-v1", sab=True)

        # reset the environment to get the first observation
        done = False
        observation, info = env.reset()

        agent = BlackjackAgent(
            env=env,
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

                # Update tracked Q-values
                if tracked_states and obs in tracked_states:
                    q_values_history.append(agent.q_values.get(obs, np.zeros(env.action_space.n)))

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs

            agent.learning_rate_decay(learning_rate, episode)
            agent.decay_epsilon()

        with open("environment_agent.pkl", "wb") as f:
            pickle.dump(env, f)
            pickle.dump(agent, f)

    else:
        with open("environment_agent.pkl","rb") as f:
            env = pickle.load(f)
            agent = pickle.load(f)
    if tracked_states:
        return learning_rate, env, agent, n_episodes, q_values_history
    else:
        return learning_rate, env, agent, n_episodes

learning_rate, env, agent, n_episodes = main()
visualize_rate(learning_rate, env=env, agent=agent, rolling_length=10000)
visualize_grid(agent,episode_number= n_episodes)