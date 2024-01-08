def identify_and_visualize_critical_states(agent):
    critical_states = [
        # Define the critical states here, for example:
        (12, 2),  # Player's hand value 12 and dealer's upcard 2
        (15, 7),  # Player's hand value 15 and dealer's upcard 7
        # Add more critical states as needed
    ]

    for state in critical_states:
        player_hand_value, dealer_upcard = state

        # Generate the state in a format suitable for your agent
        # For example, in gym Blackjack-v0, it might be (player_hand_value, dealer_upcard, usable_ace)
        formatted_state = (player_hand_value, dealer_upcard, False)  # Assuming no usable Ace initially

        # Get Q-values for the current state
        q_values = agent.get_q_values(formatted_state)

        # Plot the Q-values for the current state
        plt.figure(figsize=(6, 4))
        actions = [0, 1]  # Assuming two possible actions (0: Stand, 1: Hit), adjust as needed
        plt.bar(actions, q_values)
        plt.xlabel('Actions')
        plt.ylabel('Q-values')
        plt.title(f'Q-values for state: {formatted_state}')
        plt.xticks(actions, ['Stand', 'Hit'])  # Labeling the x-axis
        plt.show()