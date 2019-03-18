"""
config.py
"""
from multiprocessing import cpu_count
import torch
from players.RandomPlayer import RandomPlayer

class Config():
    name = 'alpha234'

    # RL Training
    numIters = 1000
    numEps = 100
    tempThreshold = 15
    updateThreshold = 0.6
    maxlenOfQueue = 200000
    numMCTSSims = 25
    arenaCompare = 10 # number of games of self play to choose previous or current nnet
    cpuct = 1
    numItersForTrainExamplesHistory = 20

    # Hardware
    num_workers = cpu_count()

    # Model
    lr = 0.001
    betas = (0.9, 0.999)
    dropout = 0.3
    epochs = 10
    batch_size = 64
    cuda = torch.cuda.is_available()
    num_channels = 512

    # Metrics
    metric_opponents = [RandomPlayer]
    metricArenaCompare = 1 # number of games to play against metric opponent


    # Model Loading
    checkpoint = './saved/'
    load_model = False
    load_folder_file = ('/dev/models/8x100x50','best.pth.tar')

    # Logging
    log_dir = 'saved/runs'
    tensorboardX = True
