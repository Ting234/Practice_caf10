import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 输入图片大小为：3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64,
                                 3)  # 64 * 222 * 222                               (224 - 3 + 2*0)/1 + 1 = 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222               (222 - 3 + 2*1)/1 + 1 = 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112     (222 - 2 + 2*1)/2 + 1 = 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110                            (112 - 3 + 2*0)/1 + 1 =110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110           (110 - 3 + 2*1)/1 + 1 =110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56      (110 - 2 + 2*1)/2 + 1 = 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54                             (56 - 3 + 2*0)/1 + 1 = 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54             (54 - 3 + 2*1)/1 + 1 = 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54             (54 - 3 + 2*1)/1 + 1 = 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28      (54 - 2 + 2*1)/2 + 1 = 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26                             (28 - 3 + 2*0)/1 + 1 = 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26             (26 - 3 + 2*1)/1 + 1 = 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26             (26 - 3 + 2*1)/1 + 1 = 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14      (26 - 2 + 2*1)/2 + 1 = 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12                             (14 - 3 + 2*0)/1 + 1 = 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12             (12 - 3 + 2*1)/1 + 1 = 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12             (12 - 3 + 2*1)/1 + 1 = 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7        (12 - 2 + 2*1)/2 + 1 =7

        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)  # 512 * 7 * 7 = 25088 ————> 4096
        self.fc2 = nn.Linear(4096, 4096)  # 4096 ————> 4096
        self.fc3 = nn.Linear(4096, 1000)  # 4096 ————> 1000
        # softmax 1 * 1 * 1000

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)

        return out
