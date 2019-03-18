"""
main.py
Method to train the neural network
"""

from Coach import Coach, CoachMP
from PytorchNNet import NNetWrapper
from config import Config


def main(config):
    game = config.game()

    # Set up model
    nn = config.nnet(game, **config.nnet_kwargs)
    nnet = NNetWrapper(game, config, nnet=nn, tensorboard=config.tensorboardX)

    # load model from checkpoint
    if config.load_model:
        nnet.load_checkpoint(folder=config.load_model_file[0], filename=config.load_model_file[1])

    if config.use_multiprocessing: coach = CoachMP(game, nnet, config)
    else: coach = Coach(game, nnet, config)

    # load training examples
    if config.load_train_examples:
        print("Load trainExamples from file")
        coach.loadTrainExamples()

    coach.learn()

if __name__=="__main__":
    config = Config()
    main(config)
