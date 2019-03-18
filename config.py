"""
config.py
"""
import os
from multiprocessing import cpu_count
import torch
from functools import partial

# from games.connect4.Connect4Game import Connect4Game
# from games.connect4.Connect4NNet import Connect4NNet
from games.tictactoe.TicTacToeGame import TicTacToeGame
from games.tictactoe.TicTacToeNNet import TicTacToeNNet
from players.RandomPlayer import RandomPlayer
from players.OneStepLookaheadPlayer import OneStepLookaheadPlayer


class Config():
    # Overall setting
    name = 'alpha234_tictactoe'
    game = TicTacToeGame
    nnet = TicTacToeNNet
    use_multiprocessing = True


    # RL Training
    numIters = 1000
    numEps = 2
    tempThreshold = 15
    updateThreshold = 0.6
    maxlenOfQueue = 200000
    numMCTSSims = 25
    arenaCompare = 40 # number of games of self play to choose previous or current nnet
    cpuct = 1
    numItersForTrainExamplesHistory = 20


    # Hardware
    # num_workers = cpu_count()
    num_workers = 2
    cuda = torch.cuda.is_available() # use cuda if available


    # Model Architecture
    dropout = 0.3
    num_channels = 512
    nnet_kwargs = {'num_channels':num_channels, 'dropout':dropout}


    # Model Training
    epochs = 1 # number of epochs of train model given a single iteration
    batch_size = 64
    lr = 0.001
    optimizer = torch.optim.Adam
    optimizer_kwargs = {'betas': (0.9, 0.999)}
    lr_scheduler = torch.optim.lr_scheduler.StepLR
    lr_scheduler_kwargs = {'step_size':1, 'gamma':1}


    # Metrics
    metric_opponents = [RandomPlayer, OneStepLookaheadPlayer]
    metricArenaCompare = 20 # number of games to play against metric opponent


    # Model Loading
    checkpoint = os.path.join('saved/', name)
    load_model = False # load model
    load_model_file = (checkpoint, 'best.pth.tar')
    load_train_examples = False # load training examples
    load_folder_file = (checkpoint,'best.pth.tar') # file to training examples


    # Logging
    log_dir = 'saved/runs'
    tensorboardX = False
