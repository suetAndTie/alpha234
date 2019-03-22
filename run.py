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
from games.othello.OthelloGame import OthelloGame
from players.OneStepLookaheadPlayer import OneStepLookaheadPlayer
from players.MCTSPlayer import MCTSPlayer
from players.NNetPlayer import NNetPlayer
from players.RandomPlayer import RandomPlayer
from PytorchNNet import NNetWrapper

from config import Config
from trained.connect4.connect4_config import Config as C4Config
from trained.connect4_resnet.connect4_resnet_config import Config as ResConfig

def main():
    game = Connect4Game()
    config = Config()

    rp = RandomPlayer(game).play
    oslp = OneStepLookaheadPlayer(game).play
    hp = HumanConnect4Player(game).play
    mctsp = MCTSPlayer(game, config).play

    c4config = C4Config()
    nn = NNetWrapper(game, c4config)
    ckpt = ('./trained/connect4','connect4_best_34.pth.tar')
    nn.load_checkpoint(ckpt[0], ckpt[1])
    nnp = NNetPlayer(game, nn, c4config).play

    nn2 = NNetWrapper(game, c4config)
    ckpt2 = ('./trained/connect4','connect4_checkpoint_26.pth.tar')
    nn2.load_checkpoint(ckpt2[0], ckpt2[1])
    nnp2 = NNetPlayer(game, nn2, c4config).play


    arena = Arena(hp, nnp2, game, display=display)
    # arena = ArenaMP(nnp, nnp2, game, display=display)
    # arena.playGame(verbose=True)
    out = arena.playGames(50, verbose=False)
    print(out)


if __name__ == '__main__':
    main()
