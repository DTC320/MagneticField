import numpy as np
import wandb
import yaml
import shutil
from yaml.loader import SafeLoader
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

parser = ArgumentParser()
parser.add_argument(
    '--config', type=str,
    default='./EXE/AE/config.yaml',
    help="training configuration"
)


class CNN_Encoder(nn.Module):
    def __init__(self, in_channel, hidden_dims):
        super(CNN_Encoder, self).__init__()
        layers = []
        hidden_dims = [16, 32]

        # self.conv1 = nn.Conv2d(in_channel, dim, kernel_size=5, stride=2)
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, dim, kernel_size=5, stride=2),
                    nn.BatchNorm2d(dim),
                    nn.PReLU(),
                )
            )
            in_channel = dim
        layers.append(nn.MaxPool2d(4))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        编码器
        :param imgs: [N, in_features]
        :return: [N, out_features]
        """
        b, c, h, w = x.shape
        x = self.encoder(x)
        return x.view(b, -1)
        

class CNN_Decoder(nn.Module):
    def __init__(self, in_channel, hidden_dims):
        super(CNN_Decoder, self).__init__()
        self.in_channel = in_channel
        layers = []
        hidden_dims = [32, 16]

        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channel, dim, 5, stride=2),
                    nn.ReLU(),
                )
            )
            in_channel = dim
        
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channel, 1, 5, stride=2, padding=1, output_padding=1),
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
        x = x.view(-1, self.in_channel, 1, 1)
        return self.decoder(x)
        

class AEEncoder(nn.Module):
    def __init__(self, in_channel, in_height, in_width, out_features, hidden_dims):
        super(AEEncoder, self).__init__()
        self.in_features = in_channel * in_height * in_width
        in_features = in_channel * in_height * in_width
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
    def __init__(self, in_features, out_channel, out_height, out_width, hidden_dims):
        super(AEDecoder, self).__init__()
        self.in_features = in_features
        out_features = out_channel * out_height * out_width
        layers = []
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features, dim),
                    nn.BatchNorm1d(dim),
                    nn.PReLU()
                )
            )
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
        imgs = imgs.view(-1, self.in_features)
        return self.decoder(imgs)


class AE(nn.Module):
    def __init__(self, img_channel, img_height, img_width, hidden_dims, latent, cnn=False):
        super(AE, self).__init__()
        self.img_channel = img_channel
        self.img_height = img_height
        self.img_width = img_width

        if cnn:
            self.encoder = CNN_Encoder(self.img_channel, latent)
            self.decoder = CNN_Decoder(latent, self.img_channel)
        else:
            self.encoder = AEEncoder(
                self.img_channel,
                self.img_height,
                self.img_width,
                latent,
                hidden_dims
            )

            self.decoder = AEDecoder(
                latent,
                self.img_channel,
                self.img_height,
                self.img_width,
                hidden_dims
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)
        return x

if __name__ == '__main__':
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=SafeLoader)
    if config['wandb']: wandb.init(project="AE_MNIST", config=config)

    # prepare output folders
    datapath = Path(__file__).parent.absolute() / '..' / '..' / 'output' / 'AE'
    if not datapath.exists(): datapath.mkdir(parents=True)

    # load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(28, 28), scale=(0.85, 1.0))
    ])

    trainset = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)
    testset = datasets.MNIST(root="./data/", transform=transform, train=False, download=True)

    trainloader = DataLoader(trainset,
                             batch_size=config['batch_size'],
                             shuffle=True,
                             num_workers=config['num_workers'],
                             prefetch_factor=config['batch_size']*2
                            )
    testloader = DataLoader(testset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                           )

    #2 Net
    net = AE(
        img_channel=1,
        img_height=28,
        img_width=28,
        hidden_dims=config['hidden_dims'],
        latent=config['latent'],
        cnn=config['CNN']
    )

    if config['loss'] == 'BCE':
        loss_fn = nn.BCELoss()
    elif config['loss'] == 'MSE':
        loss_fn = nn.MSELoss()
    else:
        raise NotImplementedError()

    if config['optimizer'] == 'adam':
        opt = optim.Adam(net.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'sgd':
        opt = optim.SGD(net.parameters(), lr=config['lr'])
    else:
        raise NotImplementedError()

    if config['scheduler']:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.97**epoch)

    # GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net.to(device)
    loss_fn.to(device)
    
    #3.train
    total_train_samples = len(trainset)
    test_step = 0
    for epoch in range(config['epochs']):
        #a. train
        net.train(True)
        train_loss = []
        for data in testloader:
            images, _ = data

            # forward
            _output = net(images)
            _loss = loss_fn(_output, images)
            #backward
            opt.zero_grad()
            _loss.backward()
            opt.step()
            train_loss.append(_loss.item())

        if config['scheduler']: scheduler.step()

        #b. evaluate
        net.eval()
        test_loss = []
        with torch.no_grad():
            for data in testloader:
                images, _ = data

                # forward
                _outputs = net(images)
                _loss = loss_fn(_outputs, images)
                test_loss.append(_loss.item())

        if config['wandb']:
            wandb.log({
                'train_loss': sum(train_loss) / len(train_loss),
                'test_loss': sum(test_loss) / len(test_loss)
            })
        print(f"Loss of {epoch+1:02d}/{config['epochs']} | Train: {sum(train_loss) / len(train_loss):.3f} | Test: {sum(test_loss) / len(test_loss):.3f}")

    #c. save
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    torch.save(net, Path(datapath, f'{timestamp}_m.pkl'))
    shutil.copy(args.config, Path(datapath, f'{timestamp}_cfg.yaml'))


    #d. random prediction
    net.eval()
    with torch.no_grad():
        idx = np.random.randint(0, len(testset))
        img, label = testset[idx]
        img = img[None, ...]
        img0 = net(img)
        img = torch.cat([img, img0], dim=0)
        torchvision.utils.save_image(img, Path(datapath, f'{timestamp}_{label}.png'))

    wandb.finish()