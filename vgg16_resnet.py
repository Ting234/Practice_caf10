import torch
import numpy as np
import torch.nn.functional as F
# 创建网络
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.Mish(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.mish(out)

        return out

# 残差模块的网络框架
# class ResidualBlock(torch.nn.Module):
#     def __init__(self, inchannels, outchannels, use_1x1conv=False, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = torch.nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, stride=stride)
#         self.conv2 = torch.nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride)
#         else:
#             self.conv3 = None
#         self.b1 = torch.nn.BatchNorm2d(outchannels)
#         self.b2 = torch.nn.BatchNorm2d(outchannels)
#
#
#     def forward(self, x):
#         y = F.relu(self.b1(self.conv1(x)))
#         y = self.b2(self.conv2(y))
#         if self.conv3:
#             x = self.conv3(x)
#         return F.relu(x+y)



class vgg16_resNet(nn.Module):
    def __init__(self):
        super(vgg16_resNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(inplace=True),
            ResidualBlock(64, 64),
            # inplace-选择是否进行覆盖运算
            # 意思是是否将计算得到的值覆盖之前的值，比如
            # 意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            # 这样能够节省运算内存，不用多存储其他变量
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Mish(inplace=True),
            ResidualBlock(64, 64),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Mish(inplace=True),
            ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Mish(inplace=True),
            ResidualBlock(128, 128),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Mish(inplace=True),
            ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Mish(inplace=True),
            ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Mish(inplace=True),
            ResidualBlock(256, 256),

            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),
            ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),
            ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),
            ResidualBlock(512, 512),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),
            ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),
            ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Mish(inplace=True),
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
            nn.Mish(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.Mish(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.Mish(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 10),
            nn.Mish(inplace=True)
                    )

    def forward(self, x):
        x = self.conv(x)
        # print(x)
        x = x.view(-1, 2048)
        # print((len(x),len(x[0])))
        x = self.fc(x)
        return x