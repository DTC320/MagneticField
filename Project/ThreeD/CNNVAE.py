import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class VAE(nn.Module):
    def __init__(self, img_channels, latent_dim):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 获取编码器的输出大小
        self.encoder_output_size = self.encoder(torch.zeros(1, img_channels, 32, 32, 32)).shape[1]

        # 计算均值和对数方差
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)

        # 解码器
        self.fc_decode = nn.Linear(latent_dim, self.encoder_output_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4, 4)),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(self.fc_decode(z))
        return x_recon, mu, logvar

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = np.load("Megnat_3D.npy", allow_pickle=True)

    # Trainset/testset
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 将NumPy数组转换为张量
    train_data_tensor = torch.from_numpy(train_data)
    test_data_tensor = torch.from_numpy(test_data)

    # 创建DataLoader
    batch_size = 64
    train_dataset = TensorDataset(train_data_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_data_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    def vae_loss(x, x_recon, mu, logvar):
        recon_loss = torch.sum((x_recon - x) ** 2) / x.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_div


    img_channels = 3
    latent_dim = 128
    learning_rate = 0.0007

    model = VAE(img_channels, latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 100
    wandb.init(project="MagVAE_3D",
               config={
                   "Learning_rate": learning_rate,
                   "Architecture": "VAE",
                   "Dataset": "3D",
                   "Num_epochs": num_epochs,
                   "Batch_size": batch_size,
               })

    # TRAIN
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.float().to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(data, recon_batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        wandb.log({"Epoch": epoch + 1, "Training loss": train_loss})
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch + 1, train_loss))

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for i, (data,) in enumerate(test_loader):
            data = data.float().to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss(data, recon_batch, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    wandb.log({"Testing loss": test_loss})
    print('====> Test set loss: {:.4f}'.format(test_loss))
    wandb.finish()