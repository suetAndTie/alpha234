from Arena import Arena
# from games.tictactoe.TicTacToeGame import TicTacToeGame, display
# from games.tictactoe.TicTacToePlayers import *

from games.connect4.Connect4Game import Connect4Game, display
from games.connect4.Connect4Players import *

def main():
    game = Connect4Game()

    rp = RandomPlayer(game).play
    hp = HumanConnect4Player(game).play

    arena = Arena(hp, rp, game, display=display)

    arena.playGames(2, verbose=True)

if __name__ == '__main__':
    main()
