# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch.nn as nn
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


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


download_root = 'MNIST_data/'

dataset1 = datasets.MNIST(root=download_root,
                         train=True,
                         transform = transforms.ToTensor(),
                         download=True)

dataset2 = datasets.MNIST(root=download_root,
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# DataLoader를 이용해서 batch size 정하기
batch_s = 100
dataset1_loader = DataLoader(dataset1, batch_size=batch_s, shuffle=True)
dataset2_loader = DataLoader(dataset2, batch_size=batch_s, shuffle=True)

model = Net()
model.zero_grad()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

total_batch = len(dataset1_loader)
epochs = np.arange(1,11)
print(epochs)
print(len(dataset1_loader)) #60000개의 data를 batch_size를 100으로 했기 때문에 600이 나오는 것. 

loss_list = []
accuracy_list = []
for epoch in epochs:
    cost=0
    for images, labels in dataset1_loader: #총 600번 반복
        
        images = images.reshape(100,784)
        
        optimizer.zero_grad() # 변화도 매개변수 0
        
        #forward
        pred = model.forward(images)
        loss = loss_function(pred, labels)
        
        #backward
        loss.backward()
        
        #Update
        optimizer.step()
        
        cost += loss # 600번 반복해서 나온 loss를 다 더해주는 것
        
    with torch.no_grad(): #미분하지 않겠다는 것
        total = 0
        correct=0
        for images, labels in dataset2_loader:
            images = images.reshape(100,784)

            outputs = model(images)
            _,predict = torch.max(outputs.data, 1)
            
            
            total += labels.size(0) # label 열 사이즈 같음 총 6만이 나오게 된다.
            correct += (predict==labels).sum() # 예측한 값과 일치한 값의 합
            
    avg_cost = cost / total_batch # 600번 반복해서 나온 loss / 600(batch size)
    accuracy = 100*correct/total 
    
    loss_list.append(avg_cost.detach().numpy())
    accuracy_list.append(accuracy)
    
    print("epoch : {} | loss : {:.6f}" .format(epoch, avg_cost))
    print("Accuracy : {:.2f}".format(100*correct/total))
    print("------")
    

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epochs,loss_list)
plt.subplot(1,2,2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(epochs, accuracy_list)
plt.show()



