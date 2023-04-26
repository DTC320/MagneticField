import numpy as np
import wandb
import yaml
import shutil
from yaml.loader import SafeLoader
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributions as td
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from net import AE

def loss_function(recon_x, x, q_z, variational):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    if variational:
        p_z = td.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
        KL = td.kl_divergence(q_z, p_z).sum()
    else:
        KL = torch.zeros(size=BCE.size())

    return BCE, KL

if __name__ == "__mian__":
    config = yaml.load(
        open()
    )