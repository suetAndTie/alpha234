"""
Based on
https://github.com/suragnair/alpha-zero-general
https://github.com/victoresque/pytorch-template/blob/master/base/base_trainer.py
"""

import os
import sys
import datetime
from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from tqdm import tqdm
from pickle import Pickler, Unpickler
from random import shuffle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from utils.visualization import WriterTensorboardX
from metric import elo


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.args = args
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, self.args)  # the competitor network
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()

        self.elo = 0 # elo score of the current model

        self.logger = logging.getLogger(self.__class__.__name__)
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        # setup visualization writer instance
        writer_dir = os.path.join(self.args.log_dir, self.args.name, self.__class__.__name__, start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, self.args.tensorboardX)

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in tqdm(range(1, self.args.numIters+1), desc='Iteration'):
            self.writer.set_step(i, "learning")
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for eps in tqdm(range(self.args.numEps), desc='mcts.Episode'):
                    iterationTrainExamples += self.executeEpisode()


                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i-1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            print("PITTING AGAINST METRIC COMPONENTS")
            for metric_opponent in self.args.metric_opponents:
                arena = Arena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                              metric_opponent(self.game).play, self.game)
                nwins, owins, draws = arena.playGames(self.args.metricArenaCompare)
                self.writer.add_scalar('{}_win'.format(metric_opponent.__name__),
                                       float(nwins) / self.args.metricArenaCompare)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            self.writer.add_scalar('self_win', float(nwins) / self.args.arenaCompare)

            # Calculate elo score for self play
            results = [-x for x in arena.get_results()] # flip to be next neural network wins
            nelo, pelo = elo(self.elo, self.elo, results)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.elo = pelo
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.elo = nelo
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            self.writer.add_scalar('self_elo', self.elo)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile+".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

"""
Multiprocessing
https://stackoverflow.com/questions/44185770/call-multiprocessing-in-class-method-python
"""
from multiprocessing import Pool
def parallel_call(params):  # a helper for calling 'remote' instances
    cls = getattr(sys.modules[__name__], params[0])  # get our class type
    instance = cls.__new__(cls)  # create a new instance without invoking __init__
    instance.__dict__ = params[1]  # apply the passed state to the new instance
    method = getattr(instance, params[2])  # get the requested method
    args = params[3] if isinstance(params[3], (list, tuple)) else [params[3]]
    return method(*args)  # expand arguments, call our method and return the result

class CoachMP(Coach):
    def prepare_call(self, name, args):  # creates a 'remote call' package for each argument
        for arg in args:
            yield [self.__class__.__name__, self.__dict__, name, arg]

    def executeEpisode(self, lock=None):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Params:
            lock=None: optional multiprocessing lock to add
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        self.mcts = MCTS(self.game, self.nnet, self.args, lock=lock)   # reset search tree
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in tqdm(range(1, self.args.numIters+1), desc="Iteration"):
            self.writer.set_step(i-1, "learning")
            # Global lock for multiprocessing
            manager = multiprocessing.Manager()
            lock = manager.Lock()

            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                bar = tqdm(desc='mcts.Episode', total=self.args.numEps)

                with Pool(processes=self.args.num_workers) as pool:
                    # pool.
                    futures = []
                    for eps in range(self.args.numEps):
                        # iterationTrainExamples += self.executeEpisode()
                        # futures.append(executor.submit(self.executeEpisode, lock))
                        futures.append(executor.submit(parallel_call, self.prepare_call("executeEpisode", lock)))

                    for future in as_completed(futures):
                        # iterationTrainExamples += future.result()
                        print(future.result())
                        # bookkeeping + plot progress
                        bar.update()

                bar.close()
                raise

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i-1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args, lock=lock)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args, lock=lock)

            print("PITTING AGAINST METRIC COMPONENTS")
            for metric_opponent in self.args.metric_opponents:
                arena = ArenaMP(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                              metric_opponent(self.game).play, self.game)
                nwins, owins, draws = arena.playGames(self.args.metricArenaCompare)
                self.writer.add_scalar('{}_win'.format(metric_opponent.__name__),
                                       float(nwins) / self.args.metricArenaCompare)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = ArenaMP(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game, lock=lock)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, num_workers=self.args.num_workers)
            self.writer.add_scalar('self_win', float(nwins) / self.args.arenaCompare)

            # Calculate elo score for self play
            results = [-x for x in arena.get_results()] # flip to be next neural network wins
            nelo, pelo = elo(self.elo, self.elo, results)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            self.writer.add_scalar('self_elo', self.elo)
