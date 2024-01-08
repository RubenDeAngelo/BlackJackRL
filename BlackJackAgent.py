import numpy as np


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
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = {}  # Regular dictionary instead of defaultdict

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values.get(obs, np.zeros(self.env.action_space.n))))

    def update(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values.get(next_obs, np.zeros(self.env.action_space.n)))
        temporal_difference = (
                reward + self.discount_factor * future_q_value -
                self.q_values.get(obs, np.zeros(self.env.action_space.n))[action]
        )

        self.q_values.setdefault(obs, np.zeros(self.env.action_space.n))
        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def update_learning_rate(self, new_learning_rate):
        self.learning_rate = new_learning_rate

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def learning_rate_decay(self, learning_rate, episode):
        # Calculate the decayed learning rate for this episode
        decayed_learning_rate = learning_reate_decay_function(learning_rate, episode)

        # Set the minimum learning rate (e.g., 1e-5 or 0.00001)
        min_learning_rate = 1e-5

        # Apply the minimum threshold to the decayed learning rate
        decayed_learning_rate = max(min_learning_rate, decayed_learning_rate)

        # Set the agent's learning rate for this episode
        self.update_learning_rate(decayed_learning_rate)


def learning_reate_decay_function(learning_rate, episode):
    return learning_rate / (1 + episode * 0.001)
