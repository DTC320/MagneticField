import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import wandb


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
    wandb.init(project="AE_MNIST")

    _batch_size = 16

    # load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size = (28, 28), scale = (0.85, 1.0))
    ])

    trainset = datasets.MNIST(root="./data/",
                                transform=transform,
                                train=True,
                                download=True)

    testset = datasets.MNIST(root="./data/",
                               transform=transform,
                               train=False,
                               download=True)

    trainloader = DataLoader(trainset,
                             batch_size= _batch_size,
                             shuffle= True,
                             num_workers=2,
                             prefetch_factor= _batch_size *2
                             )
    testloader = DataLoader(testset,
                             batch_size=_batch_size,
                             shuffle=False,
                             num_workers=0,
                             )

    #2 Net
    net = AE(1,28,28)
    loss_fn = nn.MSELoss()
    opt = optim.SGD(net.parameters(), lr = 0.001)

    # GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net.to(device)
    loss_fn.to(device)
    

    #3.train
    total_train_samples = len(trainset)
    total_epoch = 100
    train_step = 0
    test_step = 0
    for epoch in range(total_epoch):
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

            if train_step % 100 == 0:
                wandb.log({'train_loss': _loss})
                print(f"Train {epoch + 1}/{total_epoch}  {train_step} loss:{_loss.item():.3f}")
            train_step += 1

        #b. evaluate
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images, _ = data

                # forward
                _outputs = net(images)
                _loss = loss_fn(_outputs, images)

                if test_step % 10 == 0:
                    wandb.log({'test_loss': _loss})
                    print(f"Test {epoch + 1}/{total_epoch}  {test_step} loss:{_loss.item():.3f}")
                test_step += 1


        #c. save
        if epoch % 2 == 0:
            torch.save(net, f'./EXE/AE/output/models./m_{epoch}.pkl')


        #d. random prediction
        net.eval()
        with torch.no_grad():
            idx = np.random.randint(0, len(testset))
            img, label = testset[idx]
            img = img[None, ...]
            img0 = net(img)
            img = torch.cat([img, img0], dim=0)
            torchvision.utils.save_image(img, f'./EXE/AE/output/images/{epoch}_{label}.png')
    torch.save(net, f'./EXE/AE/output/models/m.pkl')  
    wandb.finish()