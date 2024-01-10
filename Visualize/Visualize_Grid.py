from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

from optimal_policy import optimal_policy_matrix


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


def create_plots(value_grid, policy_grid, diff_indices, title: str):
    """Creates a plot using a value and policy grid with highlighted differences."""
    player_count, dealer_count, value = value_grid

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the state values as a heatmap with numeric annotations
    ax1 = axs[0]
    heatmap = ax1.imshow(
        value,
        cmap='viridis',
        origin='lower',  # Ensure correct orientation
        extent=[11.5, 21.5, 0.5, 10.5],  # Adjust extent to match the other grid
        aspect='auto'  # Adjust aspect to fit the grid cells properly
    )
    fig.colorbar(heatmap, ax=ax1)  # Add color bar for reference
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.set_xticks(np.arange(12, 22))
    ax1.set_yticks(np.arange(1, 11))
    ax1.set_xticklabels(np.arange(12, 22))
    ax1.set_yticklabels(np.arange(1, 11))  # Update y-tick labels

    # Annotate each cell with the corresponding state value at the center
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            ax1.text(j + 12, i + 1, f'{value[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)

    # Plot the policy with highlighting differences
    ax2 = axs[1]
    sns.heatmap(policy_grid, linewidth=0.5, annot=True, cmap="Accent_r", cbar=False, ax=ax2)
    for idx in diff_indices:
        ax2.add_patch(plt.Rectangle((idx[1], idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # Add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stand"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    plt.tight_layout()
    plt.show()







# state values & policy with usable ace (ace counts as 11)
def visualize_grid(agent = None):
    usable_ace_matrix = optimal_policy_matrix(usable_ace=True)


    value_grid, policy_grid = create_grids(agent, usable_ace=True)
    diff_indicies = find_policy_differences(policy_grid, usable_ace_matrix)
    create_plots(value_grid, policy_grid,diff_indicies, title="With usable ace")


    # state values & policy without usable ace (ace counts as 1)
    value_grid, policy_grid = create_grids(agent, usable_ace=False)
    no_usable_ace_matrix = optimal_policy_matrix(usable_ace=False)
    diff_indicies = find_policy_differences(policy_grid, no_usable_ace_matrix)
    create_plots(value_grid, policy_grid,diff_indicies, title="Without usable ace")


def find_policy_differences(custom_policy_grid, optimal_policy_grid):
    """Find indices where the policy differs between two policy grids."""
    differences = (custom_policy_grid != optimal_policy_grid)
    diff_indices = np.argwhere(differences)
    return diff_indices