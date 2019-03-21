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

    resconfig = ResConfig()
    resnet = NNetWrapper(game, resconfig)
    resnetckpt = ('./trained/connect4_resnet','checkpoint_29.pth.tar')
    resnet.load_checkpoint(resnetckpt[0], resnetckpt[1])
    resnetp = NNetPlayer(game, resnet, resconfig).play

    name_list = ['rp', 'oslp', 'mctsp', 'nnp']
    opponent_list = [rp, oslp, mctsp,nnp]

    # arena = Arena(hp, rp, game, display=display)
    # arena = ArenaMP(rp, rp2, game, display=display)

    for opponent, name in zip(opponent_list, name_list):
        arena = ArenaMP(resnetp, opponent, game, display=display)
        out = arena.playGames(500, verbose=False)
        print(name)
        print(out)

if __name__ == '__main__':
    main()
