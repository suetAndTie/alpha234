"""
Based on
https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/NNet.py
"""


import argparse
import os
import shutil
import datetime
import random
import numpy as np
import math
import sys
import logging
from NeuralNet import NeuralNet

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.visualization import WriterTensorboardX
from .Connect4NNet import Connect4NNet as c4net

class NNetWrapper(NeuralNet):
    def __init__(self, game, args, tensorboard=False):
        self.args = args
        self.nnet = c4net(game, num_channels=self.args.num_channels, dropout=self.args.dropout)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.train_iteration = 0

        if self.args.cuda:
            self.nnet.cuda()

        self.logger = logging.getLogger(self.__class__.__name__)
        start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        # setup visualization writer instance
        writer_dir = os.path.join(self.args.log_dir, self.args.name, self.__class__.__name__, start_time)
        self.writer = WriterTensorboardX(writer_dir, self.logger, tensorboard)


    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr, betas=self.args.betas)

        for epoch in tqdm(range(self.args.epochs), desc="Training Epoch"):
            self.nnet.train()

            num_batches = int(len(examples)/self.args.batch_size)
            bar = tqdm(desc='Batch', total=num_batches)
            batch_idx = 0

            while batch_idx < num_batches:
                self.writer.set_step((self.train_iteration * self.args.epochs * num_batches) + (epoch * num_batches) + batch_idx)

                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                self.writer.add_scalar('pi_loss', l_pi.item())
                self.writer.add_scalar('v_loss', l_v.item())
                self.writer.add_scalar('loss', total_loss.item())

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_idx += 1

                # plot progress
                bar.set_postfix(
                    lpi=l_pi.item(),
                    lv=l_v.item(),
                    loss=total_loss.item()
                )

                bar.update()
            bar.close()
        self.train_iteration += 1

    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
