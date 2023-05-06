import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.unet = UNet()

    def forward(self, x, noise_schedule, noise_level):
        timesteps = len(noise_schedule)
        for i in range(timesteps - 1, -1, -1):
            eta = noise_schedule[i]
            x = x + self.unet(x) * (eta - noise_level)
        return x

data = np.load("data4D.npy", allow_pickle=True)
magnetic_data = torch.from_numpy(data)


class MagneticDataset(Dataset):
    def __init__(self, magnetic_data):
        self.magnetic_data = magnetic_data

    def __len__(self):
        return len(self.magnetic_data)

    def __getitem__(self, idx):
        return self.magnetic_data[idx]


dataset = MagneticDataset(magnetic_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

timesteps = 1000
noise_schedule = np.linspace(0, 1, timesteps)

model = DiffusionModel()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100

for epoch in range(num_epochs):
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()

        noise_level = noise_schedule[-1]
        noisy_data = data + torch.randn_like(data) * np.sqrt(noise_level)

        recon_data = model(noisy_data, noise_schedule, noise_level)

        loss = nn.MSELoss()(recon_data, data)

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
