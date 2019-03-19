"""
MCTS Player
uses a dummy neural network of all zeros
"""
from NeuralNet import NeuralNet
from PytorchNNet import NNetWrapper
from MCTS import MCTS
import numpy as np

class DummyNNet(NeuralNet):
        """
        Dummy Neural Network: returns 0 for everything
        """
        def __init__(self, game, args):
            self.game = game
            self.args = args
            self.player_num = 1

            self.action_size = self.game.getActionSize()

        def predict(self, board):
            pi = np.zeros(self.action_size)

            v = self.game.getGameEnded(board, self.player_num)

            valid_moves = self.game.getValidMoves(board, self.player_num)
            for move, valid in enumerate(valid_moves):
                if not valid: continue
                if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                    pi[move] = 1
                elif -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                    pi[move] = -1

            return pi, v

class MCTSPlayer():
    """
    Wrapper for neural network + MCTS player. Used for multiprocessing since
    we can't pickle lambda functions
    Params:
        game
        nnet: neural net to use
        args: config
    """
    def __init__(self, game, args):
        dummynnet = NNetWrapper(game, args)
        self.mcts = MCTS(game, dummynnet, args)

    def play(self, x, temp=0):
        return np.argmax(self.mcts.getActionProb(x, temp=temp))
