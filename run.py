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
from players.NNetPlayer import NNetPlayer
from PytorchNNet import NNetWrapper
from config import Config

def main():
    game = Connect4Game()
    config = Config()

    # rp = OneStepLookaheadPlayer(game).play
    # rp2 = OneStepLookaheadPlayer(game).play
    hp = HumanConnect4Player(game).play
    nn = NNetWrapper(game, config)
    ckpt = ('./weights/','connect4_checkpoint_26.pth.tar')
    nn.load_checkpoint(ckpt[0], ckpt[1])
    nnp = NNetPlayer(game, nn, config)

    arena = Arena(hp, nnp, game, display=display)
    # arena = ArenaMP(rp, rp2, game, display=display)

    out = arena.playGames(1, verbose=True)
    print(out)

if __name__ == '__main__':
    main()
