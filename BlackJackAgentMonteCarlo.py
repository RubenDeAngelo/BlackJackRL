from typing import List, Tuple
from collections import deque
import numpy as np

from BlackJackAgent import learning_rate_decay_function


class BlackjackAgent:
    def __init__(
            self,
            env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = {}

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values.get(obs, np.zeros(self.env.action_space.n))))

    def update_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def learning_rate_decay(self, episode):
        decayed_learning_rate = learning_rate_decay_function(self.learning_rate, episode)
        min_learning_rate = 1e-5
        decayed_learning_rate = max(min_learning_rate, decayed_learning_rate)
        self.update_learning_rate(decayed_learning_rate)

    def update_q_values_monte_carlo(self, episode_data: List[Tuple[tuple[int, int, bool], int, float, bool]]):
        returns = 0
        for obs, action, reward, terminated in reversed(episode_data):
            returns = reward + self.discount_factor * returns

        # After the episode is complete, update Q-values using the accumulated returns
        for obs, action, _, _ in episode_data:
            self.q_values.setdefault(obs, np.zeros(self.env.action_space.n))
            self.q_values[obs][action] += self.learning_rate * (returns - self.q_values[obs][action])
            self.training_error.append(abs(returns - self.q_values[obs][action]))


