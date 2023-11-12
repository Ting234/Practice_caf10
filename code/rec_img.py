import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from net_vgg16 import Net_vgg16 as Net
from vgg16_resnet import Net_vgg16_renet as Net_res
from net_8 import net_8 as Net8
import torch
import cv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#windows上为0，mac上为1
transform1 = transforms.Compose(
    [
     transforms.Resize([32,32]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img = cv2.imread('../img_test/apic24745.jpg')
img=np.transpose(img, (2,0,1))
input=torch.tensor(img, dtype=torch.float)
resize = transforms.Resize([32, 32])
# 转换图片格式

# cv2.imshow('img_window',img)  #显示图片,[图片窗口名字，图片]
# cv2.waitKey(0)  # 无限期显示窗口
# resized_up = cv2.resize(img, (32, 32), interpolation= cv2.INTER_LINEAR)
input = resize(input)
print(input.shape)
print(type(input))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#windows上为0，mac上为1
model = torch.load('Net8_mish.pt')
model.eval()

input = torch.unsqueeze(input, dim=0)
input = input.to(device)
output = model(input)
predicted = output.argmax(dim=1)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

torch.set_printoptions(sci_mode=False)
print(output)
if predicted[0]==0:
    print('该图片的预测为：  '+str(classes[0]))
else:
    print("oters")
