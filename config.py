"""
config.py
"""
import os
from multiprocessing import cpu_count
import torch
from functools import partial

# from games.connect4.Connect4Game import Connect4Game
# from games.connect4.Connect4NNet import Connect4NNet, Connect4ResNet
from games.connect4.AlphaZeroNNet import AlphaZeroNNet
from games.othello.OthelloGame import OthelloGame
# from games.tictactoe.TicTacToeGame import TicTacToeGame
# from games.tictactoe.TicTacToeNNet import TicTacToeNNet
from players.RandomPlayer import RandomPlayer
from players.OneStepLookaheadPlayer import OneStepLookaheadPlayer


class Config():
    # Overall setting
    name = 'alpha234_othello_alphazero'
    game = OthelloGame
    nnet = AlphaZeroNNet
    use_multiprocessing = True


    # RL Training
    numIters = 1000
    numEps = 100
    tempThreshold = 15 # 30 in paper
    updateThreshold = 0.6 # 0.55 in AlphaGoZero, not in AlphaZero (used continuous updates, no selection)
    maxlenOfQueue = 200000
    numMCTSSims = 25 # 1600 is AlphaGoZero, 800 in AlphaZero
    arenaCompare = 40 # number of games of self play to choose previous or current nnet (400 in paper)
    cpuct = 1
    numItersForTrainExamplesHistory = 20


    # Hardware
    num_workers = cpu_count()
    cuda = torch.cuda.is_available() # use cuda if available


    # Model Architecture
    dropout = 0.3
    num_channels = 512
    # nnet_kwargs = {'num_channels':num_channels, 'dropout':dropout}
    nnet_kwargs = {}


    # Model Training
    epochs = 10 # num of epochs for single train iteration (not in AlphaZero, use continuous training)
    batch_size = 64 # 4096 in paper
    lr = 0.001
    optimizer = torch.optim.Adam
    optimizer_kwargs = {'betas': (0.9, 0.999), 'weight_decay':0.001}
    lr_scheduler = torch.optim.lr_scheduler.StepLR
    lr_scheduler_kwargs = {'step_size':1, 'gamma':0.967}


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
    tensorboardX = True
