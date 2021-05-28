# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,10)
    
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
        x4 = self.relu(x3)
        x5 = self.fc3(x4)
        
        return x5


download_root = './MNIST_data'

dataset1 = datasets.MNIST(root = download_root,
                         train=True,
                         transform = transforms.ToTensor(),
                         download=True)

dataset2 = datasets.MNIST(root=download_root,
                         train=False,
                         transform = transforms.ToTensor(),
                         download=True)

batch_s = 100
# dataset1_loader의 len은 600
# dataset2_loader의 len은 100
dataset1_loader = DataLoader(dataset1, batch_size = batch_s)
dataset2_loader = DataLoader(dataset2, batch_size = batch_s)

model_dict = {}
loss_dict = {}
accuracy_dict = {}
# optimizer에 따른 학습 정도를 살펴볼 4가지 테스트 케이스
optimizer_case = ['SGD','Adam','AdaGrad','RMSprop']
for key in optimizer_case:
    model_dict[key] = Net()
    loss_dict[key] = []
    accuracy_dict[key] = []

# 4가지 테스트케이스에 대한 optimizer 정의
optimizer_dict = {}
optimizer_dict['SGD'] = optim.SGD(model_dict['SGD'].parameters(),lr = 0.005 )
optimizer_dict['Adam'] = optim.Adam(model_dict['Adam'].parameters(),lr= 0.005)
optimizer_dict['AdaGrad'] = optim.Adagrad(model_dict['AdaGrad'].parameters(), lr=0.005)
optimizer_dict['RMSprop'] = optim.RMSprop(model_dict['RMSprop'].parameters(),lr=0.005)

# loss_function, total batch size, epoch 정의
loss_function = nn.CrossEntropyLoss()
total_batch = len(dataset1_loader) # 600 (60000 / 100) => (train dataset / batch_size)
epochs = np.arange(1,16)

# 총 4가지 optimizer를 사용하여 학습
for optimizer_name, optimizer in optimizer_dict.items():
    print(optimizer_name)
    for epoch in epochs:
        cost=0
        for images, labels in dataset1_loader: #dataloader는 image와 label로 구성
            # 하나의 Tensor에 데이터 784(28x28)개가 담긴 리스트가 100개 들어있음
            # 그리고 그것이 총 dataset1_loader의 len인 600개 존재
            images = images.reshape(100,784) 
           
            model_dict[optimizer_name].zero_grad()
            
            # feed forward
            predict = model_dict[optimizer_name].forward(images)
            
            # loss 값 구하기
            loss = loss_function(predict,labels) # 예측된 것과 label이 얼마나 차이가 나는지
            
            # back propagation
            loss.backward()
            
            # optimizer update
            optimizer.step()
            
            cost += loss # 총 600번의 loss를 더한다.
            
        with torch.no_grad(): # 미분하지 않겠다
            total = 0
            correct = 0
            for images, labels in dataset2_loader:
                images = images.reshape(100,784)
                
                outputs = model_dict[optimizer_name].forward(images)
                
                # torch.max에서 두 번째 인자는 dim을 의미
                # 1로 지정했다는 것은 하나의 행에서 가장 큰 값을 찾겠다는 것
                # dim을 지정하면 인덱스에 있는 값과 인덱스를 반환
                _,predict = torch.max(outputs, 1) 
                
                total += labels.size(0)
                correct += (predict == labels).sum() # 예측한 것과 labels이 얼마나 일치하는지
                
            avg_cost = cost / total_batch # loss 값 600개의 평균
            accuracy = 100 * (correct/total) 
            
            loss_dict[optimizer_name].append(avg_cost.detach().numpy())
            accuracy_dict[optimizer_name].append(accuracy)
            
            print("epoch : {} | loss : {:.6f}" .format(epoch, avg_cost))
            print("Accuracy : {:.2f}".format(100*correct/total))


# +
markers = {'SGD' : 'o', 'Adam' : 'x','AdaGrad' : 's', 'RMSprop' : 'D' }
plt.figure(figsize = (10,5))
plt.subplot(1,2,1)

for key in optimizer_case:
    plt.plot(epochs,loss_dict[key], marker = markers[key], markevery=100, label = key)
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.subplot(1,2,2)
for key in optimizer_case:
    plt.plot(epochs, accuracy_dict[key],marker = markers[key], markevery=100, label=key)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
# -


