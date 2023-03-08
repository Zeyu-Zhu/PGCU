import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, kernel_num):
        super(ResidualBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.Conv2 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)

    def forward(self, x):
        y = F.relu(self.Conv1(x), False)
        y = self.Conv2(y)
        return x + y