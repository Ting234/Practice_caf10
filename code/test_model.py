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
lr = 0.001

batch_size = 100
transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform1 = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_Dataset = datasets.CIFAR10(root='../dataset/cifar10/',
                                 train=True,
                                 download=True,
                                 transform=transform
                                 )
test_Dataset = datasets.CIFAR10(root='../dataset/cifar10/',
                                train=False,
                                download=True,
                                transform=transform1
                                )
train_Loder = DataLoader(train_Dataset,
                         shuffle=True,
                         batch_size=batch_size,
                         )
test_Loder = DataLoader(test_Dataset,
                        shuffle=False,
                        batch_size=50)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#选择gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#windows上为0，mac上为1
model = torch.load('Vgg16.pt')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, lable in test_Loder:
            image, target = inputs.to(device), lable.to(device)
            outputs = model(image)
            predicted = outputs.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print("acc       :    %d   %%"%(100*correct/total))
if __name__ == '__main__':
    for epoch in range(1):
        test()