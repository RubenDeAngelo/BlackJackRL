import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
"""
Visualization of the rewards, erorrs and episode length
"""

def visualize_rate(env=None , agent=None, rolling_length=500):
    fig, axs = plt.subplots(ncols=3, figsize=(16, 5))
    fontsize = 20
    labelsize = 10.5

    axs[0].set_title("Episode rewards", fontsize=fontsize)
    reward_moving_average = (
            np.convolve(np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid")
            / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].tick_params(axis='both', which='major', labelsize=labelsize - 1)

    axs[1].set_title("Episode lengths", fontsize=fontsize)
    length_moving_average = (
                                    np.convolve(np.array(env.length_queue).flatten(), np.ones(rolling_length),
                                                mode="same")
                                    / rolling_length
                            )[0:-10000]

    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].tick_params(axis='both', which='major', labelsize=labelsize - 1)

    axs[2].set_title("Training Error", fontsize=fontsize)
    training_error_moving_average = (
            np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
            / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].tick_params(axis='both', which='major', labelsize=labelsize - 1)

    # Ensure integer ticks on x-axes
    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    plt.tight_layout()
    plt.show()
