class tester:
    def get_action(self, obs: tuple[int, int, bool], q_values) -> int:
        return int(np.argmax(q_values[obs]))


tester = tester()

# env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
env = gym.make("Blackjack-v1", sab=True)
performance = []
performance1 = []

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # test the agents every n steps

    if episode % 5_000 == 0:
        rewards = []
        rewards1 = []

        for episode in tqdm(range(test_episodes)):
            obs, info = env.reset()
            done = False

            while not done:
                action = tester.get_action(obs, agent.q_values)
                next_obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)

                done = terminated or truncated
                obs = next_obs

        for episode in tqdm(range(test_episodes)):
            obs, info = env.reset()
            done = False

            while not done:
                action = tester.get_action(obs, agent1.q_values)
                next_obs, reward, terminated, truncated, info = env.step(action)
                rewards1.append(reward)

                done = terminated or truncated
                obs = next_obs

        training_reward = np.sum(rewards) / test_episodes
        performance.append(training_reward)

        training_reward1 = np.sum(rewards1) / test_episodes
        performance1.append(training_reward1)

    # play one episode

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

    obs, info = env.reset()
    done = False

    while not done:
        action = agent1.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent1.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon(episode)
    agent.learning_rate_decay(episode, 1)
    agent1.decay_epsilon(episode)
    agent1.learning_rate_decay(episode, 1)