import torch
import numpy as np
import torch.nn.functional as F
# 创建网络
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


class Net_vgg16(nn.Module):
    def __init__(self):
        super(Net_vgg16, self).__init__()
        self.layer1 = nn.Sequential(
            ResidualBlock(3, 64, use_1x1conv=True),
            ResidualBlock(64, 64),

            # inplace-选择是否进行覆盖运算
            # 意思是是否将计算得到的值覆盖之前的值，比如
            # 意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            # 这样能够节省运算内存，不用多存储其他变量
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, use_1x1conv=True),

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, use_1x1conv=True),

            ResidualBlock(256, 256),

            ResidualBlock(256, 256),

            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, use_1x1conv=True),

            ResidualBlock(512, 512),

            ResidualBlock(512, 512),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            ResidualBlock(512, 512),

            ResidualBlock(512, 512),

            ResidualBlock(512, 512),

            nn.MaxPool2d(2, 2)
        )
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
            # self.layer5
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 10),
            nn.ReLU(inplace=True)
                    )

    def forward(self, x):
        x = self.conv(x)
        # print(x)
        x = x.view(-1, 2048)
        # print((len(x),len(x[0])))
        x = self.fc(x)
        return x