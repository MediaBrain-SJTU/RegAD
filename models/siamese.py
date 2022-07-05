import torch
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models import resnet18

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv1x1(in_planes=256, out_planes=256)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.conv2 = conv1x1(in_planes=256, out_planes=256)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        self.conv3 = conv1x1(in_planes=256, out_planes=256)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        out = self.conv3(x)
        # out = self.relu3(x)

        return out



class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.conv1 = conv1x1(in_planes=256, out_planes=256)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.conv2 = conv1x1(in_planes=256, out_planes=256)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        out = self.conv2(x)
        # out = self.relu2(x)

        return out



