from aenet import Autoencoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Autoencoder()
loss_fn = nn.MSELoss()
opt = optim.SGD(net.parameters(), s = 0.001)


#train

def train(net,dataload):