import numpy as np


def optimal_policy_matrix(usable_ace=False):
    rows, cols = 10, 10  # Dealer card sum: 1-10, Player card sum: 12-21
    policy_matrix = np.zeros((rows, cols), dtype=int)

    for dealer_sum in range(1, 11):
        for player_sum in range(12, 22):
            # Usable Ace case
            if player_sum == 19:
                break

            if usable_ace:
                if player_sum <= 17:
                    policy_matrix[
                        dealer_sum - 1, player_sum - 12] = 1  # Hit if player's sum is less than or equal to 17
                else:
                    if 1 < dealer_sum < 9:
                        policy_matrix[dealer_sum - 1, player_sum - 12] = 0  # Stand if player's sum is 18 or more
                    else:
                        policy_matrix[dealer_sum - 1, player_sum - 12] = 1

            # No usable Ace case
            else:
                if player_sum == 17:
                    break
                if player_sum <= 16 and (dealer_sum == 1 or dealer_sum > 6):
                    policy_matrix[
                        dealer_sum - 1, player_sum - 12] = 1  # Stand if player's sum is 11 or less, or 21 or more
                if player_sum == 12 and dealer_sum in [2, 3]:
                    policy_matrix[dealer_sum - 1, player_sum - 12] = 1  # Hit if dealer's sum is 7 or more


    return policy_matrix


# Construct matrices for scenarios with and without a usable Ace
usable_ace_matrix = optimal_policy_matrix(usable_ace=True)
no_usable_ace_matrix = optimal_policy_matrix(usable_ace=False)

"""
# Display the matrices
print("Optimal Policy Matrix with Usable Ace:")
print(usable_ace_matrix)

print("\nOptimal Policy Matrix without Usable Ace:")
print(no_usable_ace_matrix)
"""