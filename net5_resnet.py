import numpy as np
import torch
import torch.nn.functional as F
# 创建网络
# 最简单的网络
from torch import nn


class ResidualBlock(torch.nn.Module):
    def __init__(self, inchannels, outchannels, use_1x1conv=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = torch.nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.b1 = torch.nn.BatchNorm2d(outchannels)
        self.b2 = torch.nn.BatchNorm2d(outchannels)

    def forward(self, x):
        y = F.relu(self.b1(self.conv1(x)))
        y = self.b2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(x+y)


class net_8(torch.nn.Module):
    def __init__(self):
        super(net_8, self).__init__()
        self.flaten = nn.Flatten()

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),

            nn.MaxPool2d(2, 2)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),

            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(64, 10),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flaten(x)
        #print(len(x[0]))
        x = self.fc(x)
        return x