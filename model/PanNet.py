import torch
import torch.nn as nn
import torch.nn.functional as fun
from .BasicBlock import *
from .PGCU import PGCU


class PanNet_PGCU(nn.Module):

    def __init__(self, channel, veclen, kernel_size=(3,3), kernel_num=32):
        super(PanNet_PGCU, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.PGCU = PGCU(4, veclen)
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)

    def forward(self, pan, ms, hpan):
        up_ms = self.PGCU(pan, ms)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + up_ms, up_ms
    
        
class PanNet(nn.Module):

    def __init__(self, channel, kernel_size=(3,3), kernel_num=32):
        super(PanNet, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                                        nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3,3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel+1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=channel, padding=1, kernel_size=kernel_size)


    def forward(self, pan, ms, hms, hpan):
        x_ms = fun.interpolate(ms, scale_factor=(4,4), mode='bicubic')
        up_ms = self.ConvTrans(hms)
        x = torch.cat([hpan, up_ms], dim=1)
        y = fun.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y + x_ms