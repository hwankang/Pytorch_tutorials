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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,n_layers):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim,n_layers,dropout=0.7, batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)
        # batch_first - 처음 들어오는 배치의 크기에 맞추는 것
    
    def forward(self,x,hidden):
        output,hidden = self.rnn(x,hidden)
        output = output[-1]
        output = self.fc(output)
       
        return output,hidden


def sin_data(x, T=100):
    return np.sin(2.0*np.pi*x/T) #들어온 list에 각각 sin을 적용


def toy_problem(T=100, amp = 0.05):
    x = np.arange(0,2*T+1) 
    return sin_data(x,T)


T = 1000 
f = toy_problem(T)
seq_length=25
data=[]
target=[]

for i in range(0,T-seq_length): # T가 100일 때 0~75까지
    data.append(f[i:i+seq_length]) # data에는 25개씩 묶인 300개의 sin된 값이 들어가 있음 
    target.append(f[i+seq_length]) # 26번째 값이 target이 됨. 총 300개

print(len(data))
print(len(target))
print(target[274])

data = np.array(data)
target = np.array(target)

data = data.reshape(T-seq_length,25,1)
target = target.reshape(T-seq_length,1)

print(data.shape)
print(target.shape)

input_dim = 1
output_dim = 1
n_layers = 1
hidden_dim = 10

rnn = RNN(input_dim,hidden_dim,output_dim,n_layers)
print(rnn)

loss_function = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr = 0.01)

steps = T - seq_length
graph = 50
hidden = None
prediction_target = []
x_data = np.array([], dtype=np.float64)
for idx, step in enumerate(range(steps)):
    data_index = data[idx] # numpy 형태의 data 0번째 index부터 가져옴 크기 25
    
    if idx ==0: # 처음 25개의 데이터 넣기
        x_data = data_index[:] # data[idx]에서 가져온 데이터
        save_x_data = data_index[:]

    else: # 2번째부터 예측된 prediction 데이터와 그 전의 24개 값 입력으로 사용
        save_x_data = np.append(save_x_data,prediction_target[idx-1])
        x_data = np.append(save_x_data[idx:idx+seq_length-1],prediction_target[idx-1]).reshape(25,1)

    y_data = np.append(data_index[1:],target[idx]).reshape(25,1) # y 데이터 : data[idx]에서 가져온 데이터의 두 번째 ~ 본인꺼 하나 +
    
    optimizer.zero_grad()

    x_Tensor = torch.Tensor(x_data).unsqueeze(0) # 학습시킬 데이터 -> (1,25,1) : 텐서
    y_Tensor = torch.Tensor(y_data) # 예측되는 데이터 -> (25,1) : 텐서
    
    prediction,hidden= rnn(x_Tensor,hidden)
    hidden = hidden.data
    
    loss = loss_function(prediction, y_Tensor)
    
    loss.backward()
    optimizer.step()
    
    prediction_target.append(prediction[seq_length-1][0].data.numpy().flatten())

    if idx % graph == 0:
        print('Loss : ', loss.item())
        plt.plot(y_data,'r.')
        plt.plot(prediction.data.numpy().flatten(),'b.')
        plt.show()

plt.plot(target,'r',label = 'target')
plt.plot(prediction_target,'b',label ='predict')
plt.legend()
plt.show()
print('300번째 예측된 값 : ',prediction_target[274]) # 300번째 예측된 값
print('실제 sin 그래프 300번째 값', target[274])




