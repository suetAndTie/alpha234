import numpy as np
from game import Game
from random_agent import RandomAgent
from copy import deepcopy


class SinglePlayerGameState():
    def __init__(self, adversary):
        self.adversary = adversary
        self.game = Game()

    def getPossibleActions(self):
        return self.game.gameState.allowedActions

    def takeAction(self, action):
        self_copy = deepcopy(self)
        next_state, value, done, info = self_copy.game.step(action)
        return self_copy

    def isTerminal(self):
        return self.game.gameState._checkForEndGame()

    def getReward(self):
        return self.game.currentPlayer
