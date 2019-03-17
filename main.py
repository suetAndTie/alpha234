"""
main.py
Method to train the neural network
"""

from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from config import Config


def main(config):
    g = Game(6)
    nnet = nn(g)

    if config.load_model:
        nnet.load_checkpoint(config.load_folder_file[0], config.load_folder_file[1])

    c = Coach(g, nnet, config)
    if config.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()

if __name__=="__main__":
    config = Config()
    main(confiug)
