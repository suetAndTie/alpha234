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
    numItersForTrainExamplesHistory = 20

    # Hardware
    num_workers = cpu_count()

    # Model
    lr = 0.001
    dropout = 0.3
    epochs = 10
    batch_size = 64
    cuda = torch.cuda.is_available()
    num_channels = 512


    # Model Loading
    checkpoint = './saved/'
    load_model = False
    load_folder_file = ('/dev/models/8x100x50','best.pth.tar')

    # Logging
    log_dir = 'saved/runs'
    tensorboardX = True
