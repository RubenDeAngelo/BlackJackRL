class BlackjackAgent_history:
    def __init__(
        self,
        initial_alpha: float,
        final_alpha:float,
        alpha_decay: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 1.0,
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
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.alpha = initial_alpha
        self.initial_alpha = initial_alpha
        self.alpha_decay = alpha_decay
        self.final_alpha = final_alpha

        self.training_error = []
        self.rewards = []
        self.observations = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.alpha * temporal_difference
        )
        self.training_error.append(temporal_difference)
        self.rewards.append(reward)
        self.observations.append(self.q_values[(16, 10, 0)][1] - self.q_values[(16, 10, 0)][0])

    def decay_epsilon(self, episode: int):
        #self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)
        self.epsilon = max(self.final_epsilon, self.initial_epsilon * np.exp(-self.epsilon_decay * episode))
        
    def learning_rate_decay(self, episode: int):
        #self.alpha = max(self.final_alpha, self.alpha - alpha_decay)
        self.alpha = max(self.final_alpha, self.initial_alpha * np.exp(-self.alpha_decay * episode))