import numpy as np
import game
import random

class RandomAgent:

    # def __init__(self):
    #

    def choose_action(self, game):
        actions = game.gameState.allowedActions
        return random.choice(actions)
