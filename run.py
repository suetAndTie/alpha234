"""
run.py

Testing ground, not actual code (use main.py instead)
"""


from Arena import Arena, ArenaMP
# from games.tictactoe.TicTacToeGame import TicTacToeGame, display
# from games.tictactoe.TicTacToePlayers import *

from games.connect4.Connect4Game import Connect4Game, display
from games.connect4.Connect4Players import HumanConnect4Player
from games.connect4.Connect4NNet import Connect4NNet
from players.OneStepLookaheadPlayer import OneStepLookaheadPlayer
from players.RandomPlayer import RandomPlayer
from PytorchNNet import NNetWrapper as NNet
from MCTS import MCTS
from utils.util import *
import torch
from connect4_config import Config
from players.NeuralNetPlayer import NNetPlayer

def main():
    game = Connect4Game()

    rp1 = RandomPlayer(game).play
    osp1 = OneStepLookaheadPlayer(game).play
    osp2 = OneStepLookaheadPlayer(game).play
    hp = HumanConnect4Player(game).play
    n1 = NNet(game, Config)
    n1.load_checkpoint('./games/connect4/pretrained_models/','connect4_checkpoint_26.pth.tar')
    mcts_args = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    n1p = NNetPlayer(game, n1, mcts_args).play

    # mcts1 = MCTS(game, n1, mcts_args)
    # n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    arena = Arena(n1p, rp1, game, display=display)
    # arena = ArenaMP(n1p, rp2, game, display=display)

    out = arena.playGames(10, verbose=True)
    print(out)

if __name__ == '__main__':
    main()
