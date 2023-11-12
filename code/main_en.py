import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from net_vgg16 import Net_vgg16 as Net
from vgg16_resnet import vgg16_resNet as Net_res
from net_8 import net_8 as Net8
from vgg16_RES_NET import Net_vgg16 as vgg16
from net5_resnet import net_8 as net5_res
import torch

#读出数据
lr = 0.01

batch_size = 32
transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(), # 基于概率的水平翻转
     transforms.RandomGrayscale(), #基于概率对图片进行灰度话
     # transforms.RandomRotation(90),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

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
# print(device)
#创建网络，网络位于net_vgg16中
model = Net().to(device)
# print(model)
#创建优化器

criterion = torch.nn.CrossEntropyLoss()#交叉墒损失函数

optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer_Rms = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
schedule = optim.lr_scheduler.StepLR(optimizer, step_size=batch_size, gamma=0.5, last_epoch=-1)

loss_list = [] #为了后续画出损失图
trainAcc_txt = "../plot_data/temp.txt"
test_acc = "../plot_data/test.txt"
def train(epoch):
    runing_loss = 0.0
    total = 0.0
    correct = 0.0
    for batch_index, data in enumerate(train_Loder, 0):
        #print("训练  ："+str(batch_index))
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()#进行梯度清零

        outputs = model(inputs)

        predicted = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        runing_loss += loss.item()


        if ((batch_index+1) % 100) == 0:
            loss_list.append(runing_loss / 100)

            with open(trainAcc_txt, "a+") as f:
                f.write(str(runing_loss / 100) + ',' + str(100 * correct / float(total)) + '\n')
                f.close
            # print(loss_list)
            print('[%d epoch,%d]  loss:%.6f' % (epoch + 1, batch_index + 1, runing_loss / 100))
            runing_loss = 0.0
    print("train____acc       :    %f   %%" % (100 * correct / float(total)))

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
    with open(test_acc, "a+") as f:
        f.write(str(100 * correct / total) + '\n')
        f.close


if __name__ == '__main__':
    for epoch in range(60):
        train(epoch)
        test()
    # 保存网络训练模型
    # torch.save(model, './Net_res.pt')

    # X = np.linspace(0, 150)
    # Y = np.linspace(0, 5, 100)



