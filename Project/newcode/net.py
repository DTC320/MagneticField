import torch
import torch.nn as nn
import torch.distributions as td


class Encoder(nn.Module):
    def __init__(self, in_channel, latent, hidden_dims, variational = False):
        super(Encoder,self).__init__()
        self.variational = variational
        layers = []
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, dim, 3),
                    nn.BatchNorm2d(dim),
                    nn.PReLU(),
                    nn.Conv2d(dim,dim,3, stride=2),
                    nn.BatchNorm2d(dim),
                    nn.PReLU(),
                )
            )
            in_channel = dim

        self.encoder = nn.Sequential(*layers)

        if self.variational:
            self.lin_mu = nn.Linear(16*in_channel, latent)
            self.ln_logvar = nn.Linear(16* in_channel, latent)
        else:
            self.linlatent = nn.Linear(16*in_channel, latent)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.encoder(x)
        