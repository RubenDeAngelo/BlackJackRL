#create a tester to evaluate the agents during training
class tester:
    def get_action(self, obs: tuple[int, int, bool], q_values) -> int:
        return int(np.argmax(q_values[obs]))


tester = tester()

#create arrays to store the agents performance

env = gym.make("Blackjack-v1", sab=True)
performance = []
performance1 = []

#train the agents

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    episode_data = []

    # test the agents every 100000 steps

    if episode % 100_000 == 0:
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

    # play one episode with 1st agent 

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_data.append((obs, action, reward, terminated))

        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

    obs, info = env.reset()
    done = False

    # play one episode with 2nd agent 

    while not done:
        action = agent1.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_data.append((obs, action, reward, terminated))

        agent1.update_q_values_monte_carlo(episode_data)

        done = terminated or truncated
        obs = next_obs

    #decay parameters
    
    agent.decay_epsilon(episode)
    agent.learning_rate_decay(episode, 1)
    agent1.decay_epsilon()
    agent1.learning_rate_decay(episode)
