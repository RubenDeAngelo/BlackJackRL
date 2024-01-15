import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

from main import main
from main_Monte_Carlo import main_monte_carlo


def compare_q_values(q_values_history_list, tracked_states):
    """
    Compare Q-values for a given state across different training runs.

    Parameters:
        q_values_history_list (list): List of Q-values history for each training run.
        tracked_states (list): List of states for which Q-values will be compared.

    Returns:
        None (displays a plot).
    """
    num_runs = len(q_values_history_list)
    num_states = len(tracked_states)

    # Ensure axs is a list, even if there is only one tracked state
    if num_states == 1:
        axs = [plt.subplot(1, 1, 1)]
    else:
        # Create a subplot for each tracked state
        fig, axs = plt.subplots(num_states, 1, figsize=(10, 3 * num_states), sharex=True)

    for state_index, state in enumerate(tracked_states):
        state_q_values = np.zeros((num_runs, 2))  # Assuming two actions (hit, stand)

        # Extract Q-values for the tracked state across different training runs
        for run_index, q_values_history in enumerate(q_values_history_list):
            state_q_values[run_index, :] = q_values_history[state_index]

        # Plot the Q-values for the tracked state
        axs[state_index].plot(state_q_values[:, 0], label='Hit')
        axs[state_index].plot(state_q_values[:, 1], label='Stand')
        axs[state_index].set_title(f'Tracked State: {state}')
        axs[state_index].set_xlabel('Training Run')
        axs[state_index].set_ylabel('Q-Value')
        axs[state_index].legend()

    plt.tight_layout()
    plt.show()

# Function to run main and get agent and q_values
# Function to get Q-values history using both TD and Monte Carlo methods
def get_q_history(
    learning_rate: float = 0.001,
    n_episodes: int = 1_000_000,
    start_epsilon: float = 1,
    final_epsilon: float = 1,
    tracked_states: List[Tuple[int, int, bool]] = None
):
    # Get Q-values history for TD method
    td_q_history_list = get_main_agent_and_q_values(
        learning_rate, n_episodes, start_epsilon, final_epsilon, tracked_states
    )

    # Get Q-values history for Monte Carlo method
    mc_q_history_list = get_main_monte_carlo_agent_and_q_values(
        learning_rate, n_episodes, start_epsilon, final_epsilon, tracked_states
    )

    return td_q_history_list, mc_q_history_list

# Function to run main and get agent and q_values
def get_main_agent_and_q_values(learning_rate, n_episodes, start_epsilon, final_epsilon, tracked_states):
    _, _, _, _, TD_q_history = main(
        train=True,
        learning_rate=learning_rate,
        n_episodes=n_episodes,
        start_epsilon=start_epsilon,
        final_epsilon=final_epsilon,
        tracked_states=tracked_states
    )
    return [np.copy(q_values) for q_values in TD_q_history]

# Function to run main_monte_carlo and get agent and q_values
def get_main_monte_carlo_agent_and_q_values(learning_rate, n_episodes, start_epsilon, final_epsilon, tracked_states):
    _, _, _, _, MC_q_history = main_monte_carlo(
        train=True,
        learning_rate=learning_rate,
        n_episodes=n_episodes,
        start_epsilon=start_epsilon,
        final_epsilon=final_epsilon,
        tracked_states=tracked_states
    )
    return [np.copy(q_values) for q_values in MC_q_history]

# Example usage:
# Define tracked states for comparison (modify based on your needs)
tracked_states = [(12, 4, False)]

# Get Q-values history for both TD and Monte Carlo
td_q_history, mc_q_history = get_q_history(
    learning_rate=0.001,
    n_episodes=1_000_00,
    start_epsilon=1,
    final_epsilon=0.1,
    tracked_states=tracked_states
)

# Compare Q-values for the tracked states across different training runs
compare_q_values([td_q_history, mc_q_history], tracked_states)
