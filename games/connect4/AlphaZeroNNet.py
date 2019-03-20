"""
Based on
AlphaGo Zero
https://www.nature.com/articles/nature24270

RESNET
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AlphaZeroNNet(nn.Module):
    """
    Resnet Model
    AlphaZero used
        1 conv block
        19 or 39 residual blocks
    """

    def __init__(self, game, block=BasicBlock, layers=19, zero_init_residual=False):
        super().__init__()

        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.inplanes = 256
        # Conv Block
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        # Residual Blocks
        self.res_block = self._make_layer(block, 256, layers)
        # policy head
        self.conv_pi = nn.Conv2d(256, 2, kernel_size=1, stride=1, bias=False)
        self.bn_pi = nn.BatchNorm2d(2)
        self.relu_pi = nn.ReLU(inplace=True)
        self.fc_pi = nn.Linear(2 * self.board_x * self.board_y, self.action_size)
        # value head
        self.conv_v = nn.Conv2d(256, 1, kernel_size=1, stride=1, bias=False)
        self.bn_v = nn.BatchNorm2d(1)
        self.relu_v = nn.ReLU(inplace=True)
        self.fc_v = nn.Linear(self.board_x * self.board_y, 256)
        self.relu_v2 = nn.ReLU(inplace=True)
        self.fc_v2 = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 1, self.board_x, self.board_y)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_block(x)

        pi = self.conv_pi(x)
        pi = self.bn_pi(pi)
        pi = self.relu_pi(pi)
        pi = pi.view(pi.size(0), -1)
        pi = self.fc_pi(pi)

        v = self.conv_v(x)
        v = self.bn_v(v)
        v = self.relu_v(v)
        v = v.view(v.size(0), -1)
        v = self.fc_v(v)
        v = self.relu_v2(v)
        v = self.fc_v2(v)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
