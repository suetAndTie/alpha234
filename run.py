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
from connect4_config import Config

def main():
    game = OthelloGame()
    config = Config()

    rp = RandomPlayer(game).play
    oslp = OneStepLookaheadPlayer(game).play
    hp = HumanConnect4Player(game).play
    mctsp = MCTSPlayer(game, config).play
    # nn = NNetWrapper(game, config)
    # ckpt = ('./weights/alpha234_connect4','connect4_best_34.pth.tar')
    # nn.load_checkpoint(ckpt[0], ckpt[1])
    # nnp = NNetPlayer(game, nn, config).play

    arena = Arena(hp, rp, game, display=display)
    # arena = ArenaMP(rp, rp2, game, display=display)

    out = arena.playGames(10, verbose=True)
    print(out)

if __name__ == '__main__':
    main()
