"""
run.py

Testing ground, not actual code (use main.py instead)
"""


from Arena import Arena, ArenaMP
# from games.tictactoe.TicTacToeGame import TicTacToeGame, display
# from games.tictactoe.TicTacToePlayers import *

from games.connect4.Connect4Game import Connect4Game, display
from games.connect4.Connect4Players import HumanConnect4Player
from players.OneStepLookaheadPlayer import OneStepLookaheadPlayer

def main():
    game = Connect4Game()

    rp = OneStepLookaheadPlayer(game).play
    rp2 = OneStepLookaheadPlayer(game).play
    hp = HumanConnect4Player(game).play
    # arena = Arena(hp, rp, game, display=display)
    arena = ArenaMP(rp, rp2, game, display=display)

    out = arena.playGames(10, verbose=True)
    print(out)

if __name__ == '__main__':
    main()
