import torch
import torch.nn as nn


class AEEncoder(nn.Module):
    def __init__(self, in_features, out_features, hidden_dims=None):
        super(AEEncoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.in_features = in_features
        layers = []
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features, dim),
                    nn.BatchNorm1d(dim),
                    nn.PReLU()
                )
            )
            in_features = dim
        layers.append(nn.Linear(in_features, out_features))
        self.encoder = nn.Sequential(*layers)

    def forward(self, imgs):
        """
        编码器
        :param imgs: [N, in_features]
        :return: [N, out_features]
        """
        imgs = imgs.view(-1, self.in_features)
        return self.encoder(imgs)


class AEDecoder(nn.Module):
    def __init__(self, in_features, out_features, hidden_dims=None):
        super(AEDecoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        self.in_features = in_features
        layers = []
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features, dim),
                    nn.BatchNorm1d(dim),
                    nn.PReLU()
                )
            )
            in_features = dim
        layers.append(nn.Linear(in_features, out_features))
        self.decoder = nn.Sequential(*layers)

    def forward(self, imgs):
        """
        解码器
        :param imgs: [N, in_features]
        :return: [N, out_features]
        """
        imgs = imgs.view(-1, self.in_features)
        return self.decoder(imgs)


class AE(nn.Module):
    def __init__(self, img_channel, img_height, img_width):
        super(AE, self).__init__()
        self.img_channel = img_channel
        self.img_height = img_height
        self.img_width = img_width
        num_features = self.img_channel * self.img_height * self.img_width
        self.encoder = AEEncoder(num_features, 32)
        self.decoder = AEDecoder(32, num_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)
        return x

if __name__ == '__main__':
    net = AE(1,28,28)
    print(net)