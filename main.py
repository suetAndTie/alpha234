"""
main.py
Method to train the neural network
"""

from Coach import Coach, CoachMP
from PytorchNNet import NNetWrapper
from config import Config
import torch.multiprocessing as mp


def main(config):
    game = config.game()

    # Set up model
    nnet = NNetWrapper(game, config, tensorboard=config.tensorboardX)

    # load model from checkpoint
    if config.load_model:
        nnet.load_checkpoint(folder=config.load_model_file[0], filename=config.load_model_file[1])

    if config.use_multiprocessing:
        # Required for multiprocessing
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        # Disable semaphore warning
        import warnings
        warnings.filterwarnings("ignore", message="There appear to be 1 leaked semaphores to clean up at shutdown")

        coach = CoachMP(game, nnet, config)
    else:
        coach = Coach(game, nnet, config)

    # load training examples
    if config.load_train_examples:
        print("Load trainExamples from file")
        coach.loadTrainExamples()

    coach.learn()

if __name__=="__main__":
    config = Config()
    main(config)
