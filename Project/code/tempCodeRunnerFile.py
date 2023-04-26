from aenet import Autoencoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import wandb
import os


#Swandb.init(project="AE_MNIST")


batch_size =16
learning_rate = 0.001
total_epoch = 100
train_step = 0
test_step = 0

net = Autoencoder()
loss_fn = nn.MSELoss()
opt = optim.SGD(net.parameters(), lr = learning_rate)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
loss_fn.to(device)


#data

file_name = '../newcode/data4D.npy'
folder_path = 'E:\GitProject\magnet\MagneticFields\Project\code'

# 使用 os.path.join 函数构建跨平台的文件路径
file_path = os.path.join(folder_path, file_name)


if not os.path.exists(file_path):
    print("File not found:", file_path)
else:
    # 读取数据集
    data = np.load(file_path)


data = torch.tensor(data)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index]
    
dataset = MyDataset(data)

dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
test_size = int(dataset_size * 0.2)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

data_train = DataLoader(train_dataset, batch_size, shuffle = True)
data_test = DataLoader(test_dataset, batch_size, shuffle = True)

print(data_train)