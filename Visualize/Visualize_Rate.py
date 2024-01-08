import numpy as np
from matplotlib import pyplot as plt

from BlackJackAgent import learning_reate_decay_function


def visualize_rate(env=None, agent=None, rolling_length=500):
    fig, axs = plt.subplots(ncols=4, figsize=(16, 5))

    axs[0].set_title("Episode rewards")
    reward_moving_average = (
        np.convolve(np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid")
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)

    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)

    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

    axs[3].set_title("Learning Rate")
    decayed_learning_rate = [
        learning_reate_decay_function(agent.learning_rate, episode) for episode in range(len(env.return_queue))
    ]
    axs[3].plot(range(len(decayed_learning_rate)), decayed_learning_rate)

    plt.tight_layout()
    plt.show()
