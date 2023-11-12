
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot

train_loss = np.loadtxt("../plot_data/res_net5.txt",delimiter=',',dtype=np.float32)
test_acc = np.loadtxt('../plot_data/res_net5_test.txt', dtype=np.float32)
itr = np.arange(0, len(train_loss), 1)
itr1 = np.arange(0, len(test_acc), 1)
# plt.plot(itr, train_loss[:,1],label='Fist line',linewidth=0.05,color='r',marker='o',
#              markerfacecolor='blue',markersize=3)
# plt.show()
# fig, a = plt.figure(figsize=(8, 8)).subplots(2, 2)
plt.rcParams['font.family'] = 'SimHei'
fig = plt.figure(figsize=(8, 6),facecolor='lightgrey')
a =fig.subplots(2, 2)

a[0][0].plot(itr, train_loss[:,1],label='Fist line',linewidth=1,color='r',marker='o',
             markerfacecolor='blue',markersize=1)
a[0][0].set_title('net5-trainAcc')
a[0][1].plot(itr, train_loss[:,0],label='second line',linewidth=1,color='b',marker='o',
             markerfacecolor='blue',markersize=1)
a[0][1].set_title('net5-trainloss')
a[1][0].plot(itr1, test_acc,label='second line',linewidth=1,color='b',marker='o',
             markerfacecolor='blue',markersize=1)
a[1][0].set_title('net5-teatAcc')
plt.suptitle('net_5_res_relu',fontsize = 20, color = 'red',backgroundcolor='yellow')
plt.savefig('../result/net_5_res_relu.jpg')
# plt.tight_layout(rect=(0,0,1,0.9))#使子图标题和全局标题与坐标轴不重叠
plt.show()