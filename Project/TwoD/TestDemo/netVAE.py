import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch import distributions


class CNN_Encoder(nn.Module):
    def __init__(self, in_channel, latent, hidden_dims, variational=False):
        super(CNN_Encoder, self).__init__()
        self.variational = variational
        layers = []
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, dim, 3),
                    nn.BatchNorm2d(dim),
                    nn.PReLU(),
                    nn.Conv2d(dim, dim, 3, stride=2),
                    nn.BatchNorm2d(dim),
                    nn.PReLU(),
                )
            )
            in_channel = dim

        self.encoder = nn.Sequential(*layers)

        if self.variational:
            self.lin_mu = nn.Linear(16 * in_channel, latent)
            self.lin_logvar = nn.Linear(16 * in_channel, latent)
        else:
            self.lin_latent = nn.Linear(16 * in_channel, latent)

    def forward(self, x):
        """
        编码器
        :param imgs: [N, in_features]
        :return: [N, out_features]
        """
        b, c, h, w = x.shape
        x = self.encoder(x)
        x = x.view(b, -1)

        if self.variational:
            z_mu = self.lin_mu(x)
            z_sig = self.lin_logvar(x)
            z = (z_mu, z_sig)
        else:
            z = self.lin_latent(x)

        return z


class CNN_Decoder(nn.Module):
    def __init__(self, out_channel, latent, hidden_dims):
        super(CNN_Decoder, self).__init__()
        self.dec_lin = nn.Sequential(
            nn.Linear(latent, 16 * hidden_dims[-1]),
            nn.ReLU()
        )
        layers = []
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 5, 2),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.PReLU(),
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-2], 5, 2, output_padding=1),
                nn.BatchNorm2d(hidden_dims[-2]),
                nn.PReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-2], out_channel, 3),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        编码器
        :param imgs: [N, in_features]
        :return: [N, out_features]
        """
        b, _ = x.shape
        x = self.dec_lin(x)
        x = self.decoder(x.view(b, -1, 4, 4))
        return x


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = CNN_Encoder
        self.decoder = CNN_Decoder

    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()
        return self.decoder(z), q_z


transform = transforms.Compose(
    [transforms.ToTensor(),
     # Normalize the images to be -0.5, 0.5
     transforms.Normalize(0.5, 1)]
    )
mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)


