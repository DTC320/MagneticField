import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class VAE(nn.Module):
    def __init__(self, img_channels, latent_dim):
        super(VAE, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )

        # 获取编码器的输出大小
        self.encoder_output_size = self.encoder(torch.zeros(1, img_channels, 32, 32)).shape[1]

        # 计算均值和对数方差
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)

        # 解码器
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            #nn.Sigmoid()
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

    data = np.load("data4D.npy", allow_pickle= True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trainset/testset
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 将NumPy数组转换为张量
    train_data_tensor = torch.from_numpy(train_data).to(device)
    test_data_tensor = torch.from_numpy(test_data).to(device)

    train_data_tensor = torch.from_numpy(train_data)
    test_data_tensor = torch.from_numpy(test_data)

    # 创建DataLoader
    batch_size = 64
    train_dataset = TensorDataset(train_data_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_data_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    def vae_loss(x, x_recon, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_div, recon_loss,kl_div


    img_channels = 3
    latent_dim = 256
    learning_rate = 0.0002
    model = VAE(img_channels, latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 400
    wandb.init(project="MagVAE_2D",
               config={
                   "Learning_rate": learning_rate,
                   "Architecture": "VAE",
                   "Dataset": "2D",
                   "Num_epochs": num_epochs,
                   "Batch_size": batch_size,
               })

    # TRAIN
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        reconed_loss = 0.0
        KLed_loss = 0.0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss,recon_loss,kl_loss = vae_loss(data, recon_batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            reconed_loss += recon_loss.item()
            KLed_loss += kl_loss.item()
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        reconed_loss /= len(train_loader.dataset)
        KLed_loss /= len(train_loader.dataset)
        wandb.log({"Epocch": epoch + 1, "Training loss": train_loss,  "T_Recon_loss": reconed_loss, "T_KLed_loss": KLed_loss}  )
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch + 1, train_loss))
        print('====>           Recon loss: {:.4f} KL loss:{:.4f}'.format(reconed_loss,KLed_loss))


    model.eval()
    test_loss = 0.0
    test_reconed = 0.0
    test_KL = 0.0

    with torch.no_grad():
        for i, (data,) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(data, recon_batch, mu, logvar)
            test_loss += loss.item()
            test_reconed += recon_loss.item()
            test_KL += kl_loss.item()


    test_loss /= len(test_loader.dataset)
    test_reconed /= len(test_loader.dataset)
    test_KL /= len(test_loader.dataset)
    wandb.log({"Testing loss": test_loss, "Test_Recon": test_reconed, "Test_KL": test_KL})
    print('====> Test loss: {:.4f} Test recon loss: {:.4f} Test KL loss: {:.4f}'.format(test_loss, test_reconed, test_KL))