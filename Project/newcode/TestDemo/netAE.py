import torch
import torch.nn as nn
import torch.distributions as td


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


class AEEncoder(nn.Module):
    def __init__(self, img_shape, latent, hidden_dims, variational=False):
        super(AEEncoder, self).__init__()
        in_channel, in_height, in_width = img_shape
        self.in_features = in_channel * in_height * in_width
        self.variational = variational

        in_features = self.in_features
        layers = []
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features, dim),
                    nn.ReLU()
                )
            )
            in_features = dim

        self.encode = nn.Sequential(*layers)

        if self.variational:
            self.lin_mu = nn.Linear(in_features, latent)
            self.lin_logvar = nn.Linear(in_features, latent)
        else:
            self.lin_latent = nn.Linear(in_features, latent)

    def forward(self, x):
        """
        编码器
        :param imgs: [N, in_features]
        :return: [N, out_features]
        """
        x = x.view(-1, self.in_features)
        x = self.encode(x)

        if self.variational:
            z_mu = self.lin_mu(x)
            z_sig = self.lin_logvar(x)
            z = (z_mu, z_sig)
        else:
            z = self.lin_latent(x)

        return z


class AEDecoder(nn.Module):
    def __init__(self, img_shape, latent, hidden_dims):
        super(AEDecoder, self).__init__()
        self.latent = latent
        out_channel, out_height, out_width = img_shape
        out_features = out_channel * out_height * out_width
        in_channel = latent

        layers = []
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channel, dim),
                    nn.ReLU()
                )
            )
            in_channel = dim

        layers.append(
            nn.Sequential(
                nn.Linear(dim, out_features),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, imgs):
        """
        解码器
        :param imgs: [N, in_features]
        :return: [N, out_features]
        """
        imgs = imgs.view(-1, self.latent)
        return self.decoder(imgs)


class AE(nn.Module):
    def __init__(self,
                 img_shape,
                 hidden_dims,
                 latent,
                 variational=False,
                 cnn=False,
                 ):
        super(AE, self).__init__()
        self.img_channel, self.img_height, self.img_width = img_shape
        self.variational = variational

        if cnn:
            self.encoder = CNN_Encoder(
                self.img_channel,
                latent,
                hidden_dims['cnn'],
                variational
            )
            self.decoder = CNN_Decoder(
                self.img_channel,
                latent,
                hidden_dims['cnn']
            )
        else:
            self.encoder = AEEncoder(
                img_shape,
                latent,
                hidden_dims['fully'],
                variational
            )

            self.decoder = AEDecoder(
                img_shape,
                latent,
                hidden_dims['fully']
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, z=None, eval=False):
        q_z = None

        if z is None:
            z = self.encoder(x)

            if self.variational:
                mu, logvar = z
                # z = self.reparameterize(mu, logvar)
                # q_z = mu, logvar
                std = logvar.exp().pow(0.5)
                q_z = td.normal.Normal(mu, std)
                z = q_z.rsample()

                # Sampled latent code for evaluation
                if eval: q_z = z

        x = self.decoder(z)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)
        return x, q_z