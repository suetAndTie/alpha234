import numpy as np
from MCTS import MCTS


class NNetPlayer():
    """
    Wrapper for neural network + MCTS player. Used for multiprocessing since
    we can't pickle lambda functions
    Params:
        game
        nnet: neural net to use
        args: config
    """
    def __init__(self, game, nnet, args):
        self.mcts = MCTS(game, nnet, args)

    def play(self, x, temp=0):
        return np.argmax(self.mcts.getActionProb(x, temp=temp))
