"""
main.py
Method to train the neural network
"""

from Coach import Coach, CoachMP
# from games.othello.OthelloGame import OthelloGame as Game
# from games.othello.pytorch.NNet import NNetWrapper as nn
# from games.tictactoe.TicTacToeGame import TicTacToeGame as Game
from games.connect4.Connect4Game import Connect4Game as Game
from games.connect4.NNet import NNetWrapper as nn
from config import Config


def main(config):
    g = Game()
    nnet = nn(g, config, tensorboard=config.tensorboardX)

    if config.load_model:
        nnet.load_checkpoint(config.load_folder_file[0], config.load_folder_file[1])

    c = Coach(g, nnet, config)
    if config.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()

if __name__=="__main__":
    config = Config()
    main(config)
