from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from Visualize.optimal_policy import optimal_policy_matrix

"""
Visualization of the state values and policy
"""

def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(action_values[0]-action_values[1])
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, diff_indices, title: str, episode_number: int, fontsize=15, labelsize=10.5):
    player_count, dealer_count, value = value_grid

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the state values as a heatmap with numeric annotations
    ax1 = axs[0]
    sns.heatmap(value, annot=True, fmt=".2f", cmap='viridis', cbar=True, ax=ax1,
                annot_kws={"size": fontsize - 5})  # Adjust size of the values in the heatmap cells
    ax1.set_xticklabels(range(12, 22)[::1], fontsize=fontsize)
    ax1.set_yticklabels(["A"] + list(range(2, 11))[::1], fontsize=fontsize)
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")

    # Plot the policy with highlighting differences
    ax2 = axs[1]
    sns.heatmap(policy_grid, linewidth=0.5, annot=True, cmap="Accent_r", cbar=False, ax=ax2,
                annot_kws={"size": fontsize})
    for idx in diff_indices:
        ax2.add_patch(plt.Rectangle((idx[1], idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))
    ax2.set_title(f"Policy: {title}")
    ax2.set_yticklabels(["A"] + list(range(2, 11))[::1], fontsize=fontsize)
    ax2.set_xticklabels(list(range(12, 22))[::1], fontsize=fontsize)

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.18, 1))

    plt.tight_layout()
    plt.show()

def visualize_grid(agent=None, episode_number=0):
    usable_ace_matrix = optimal_policy_matrix(usable_ace=True)

    value_grid, policy_grid = create_grids(agent, usable_ace=True)
    diff_indices = find_policy_differences(policy_grid, usable_ace_matrix)
    create_plots(value_grid, policy_grid, diff_indices, title=f"With usable ace, Episode {episode_number}", episode_number=episode_number)

    value_grid, policy_grid = create_grids(agent, usable_ace=False)
    no_usable_ace_matrix = optimal_policy_matrix(usable_ace=False)
    diff_indices = find_policy_differences(policy_grid, no_usable_ace_matrix)
    create_plots(value_grid, policy_grid, diff_indices, title=f"Without usable ace, Episode {episode_number}", episode_number=episode_number)


def find_policy_differences(custom_policy_grid, optimal_policy_grid):
    differences = (custom_policy_grid != optimal_policy_grid)
    diff_indices = np.argwhere(differences)
    return diff_indices