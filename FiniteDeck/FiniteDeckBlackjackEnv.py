import numpy as np
from gymnasium.envs.toy_text.blackjack import BlackjackEnv

class FiniteDeckBlackjackEnv(BlackjackEnv):
    def __init__(self, number_of_decks=2):
        super().__init__()  # Call the parent class constructor
        self.num_decks = number_of_decks  # Set the number of decks for the game
        self.cards = self._get_cards()  # Generate the deck of cards

    def _get_cards(self):
        # Create an array representing the deck with values from 1 to 11 (11 represents an Ace)
        return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 11] * 4 * self.num_decks)

    def _get_obs(self):
        player_tuple = tuple(self.player)  # Convert self.player list to a tuple
        usable_ace = 1 in self.player and sum(self.player) + 10 <= 21
        return (player_tuple, self.dealer[0], usable_ace)

    def reset(self):
        # Reset the environment to the initial state
        self.reset_deck()  # Reshuffle the deck and deal cards
        return super().reset()  # Return the initial observation

    def reset_deck(self):
        # Reset the deck by shuffling cards and dealing them to the dealer and player
        self.cards = self._get_cards()  # Generate a new deck of cards
        np.random.shuffle(self.cards)  # Shuffle the deck
        self.dealer = self.cards[:2]  # Deal two cards to the dealer
        self.player = self.cards[2:4]  # Deal two cards to the player
        return self._get_obs()  # Return the initial observation

    def step(self, action):
        step_return = super().step(action)  # Obtain the values from the superclass
        if len(self.cards) < 15:  # Reshuffle the deck if remaining cards fall below a threshold
            self.reset_deck()  # Reshuffle the deck
        return step_return  # Return the values obtained from the superclass