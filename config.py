"""
config.py
"""
from multiprocessing import cpu_count

class Config():
    name = 'alpha234'

    # Training
    numIters = 1000
    numEps = 100
    tempThreshold = 15
    updateThreshold = 0.6
    maxlenOfQueue = 200000
    numMCTSSims = 25
    arenaCompare = 40
    cpuct = 1

    # Hardware
    num_workers = cpu_count()

    # Models
    checkpoint = './saved/'
    load_model = False
    load_folder_file = ('/dev/models/8x100x50','best.pth.tar')
    numItersForTrainExamplesHistory = 20

    # Logging
    log_dir = 'saved/runs'
    tensorboardX = True
