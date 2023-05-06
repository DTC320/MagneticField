import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class CNNAE(nn.Module):
    def __init__(self):
        super(CNNAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, 3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv3d(32, 64, 7),
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 7),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose3d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # %%
    learning_rate = 0.0005
    batch_size = 64
    num_epochs = 150

    #DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化网络
    autoencoder = CNNAE().to(device)
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # 训练数据加载器

    train_data = np.load("Megnat_3D.npy", allow_pickle=True)
    train_data = torch.from_numpy(train_data).to(device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    wandb.init(
        # set the wandb project where this run will be logged
        project="MagAE_3D",

        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "CNN",
            "dataset": "3DMag",
            "epochs": num_epochs,
            "batch_size": batch_size,
        })

    for epoch in range(num_epochs):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()

        wandb.log({"Epochs": epoch + 1, "loss": loss.item()})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    torch.save(autoencoder,"AE.pkl")
    wandb.finish()
