from mcts import mcts
from random_agent import RandomAgent
from SinglePlayerGameState import SinglePlayerGameState

class MCTSAgent():
    def __init__(self, adversary=RandomAgent(), timeLimit=1000):
        self.adversary = adversary
        self.state = SinglePlayerGameState(adversary)
        self.m = mcts(timeLimit=timeLimit)

    def choose_action(self, game):
        self.state.game = game
        return self.m.search(initialState=self.state)
