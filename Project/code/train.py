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


wandb.init(project="AE_MNIST")


batch_size =1
learning_rate = 0.001
total_epoch = 100
train_step = 0
test_step = 0

net = Autoencoder()
loss_fn = nn.MSELoss()
opt = optim.SGD(net.parameters(), lr = learning_rate)


"""device = torch.device('cuda')
net.to(device)
loss_fn.to(device)"""


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

#data = torch.tensor(data).to(device)
data = torch.tensor(data)

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load data
data = np.load(file_path)
data = torch.tensor(data, dtype=torch.float32)

# Create dataset and dataloader
dataset = MyDataset(data, transform=transform)
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
data_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
data_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



train_loss = []
for epoch in range(total_epoch):
    running_loss = 0.0
    for data in data_train:
        img = data.squeeze(dim = 0)
        print(img.shape)
        #img.to(device)
            
        #forward
        """output  = net(img).to(device)
        loss = loss_fn(output, img).to(device)"""
        output  = net(img)
        loss = loss_fn(output, img)
            
        #backward
        opt.zero_grad()
        loss.backward()
        opt.step()
            
        running_loss += loss.item()
            
        if train_step % 100 == 0:
            wandb.log({'train_loss': loss})
            print(f"Train {epoch + 1}\{total_epoch}  {train_step} loss:{loss.item():.3f}")
        train_step += 1
            
        #b. evaluate
    net.eval()
    with torch.no_grad():
        for data in data_test:
            img = data.squeeze(dim = 0)
            #img.to(device)
            
            # forward
            outputs = net(img)
            loss = loss_fn(outputs, img)

            if test_step % 10 == 0:
                wandb.log({'test_loss': loss})
                print(f"Test {epoch + 1}\{total_epoch}  {test_step} loss:{loss.item():.3f}")
            test_step += 1
                
        
    #c. save
    if epoch % 2 == 0:
        torch.save(net, f'M.\Project\code\output\models\m_{epoch}.pkl')
            
torch.save(net, f'./EXE/AE/output/models/m.pkl') 
wandb.finish()
