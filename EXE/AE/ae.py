'''
Inspired from: https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
'''
import numpy as np
import wandb
import yaml
import shutil
from yaml.loader import SafeLoader
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import torch.distributions as td
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from networks import AE


def loss_function(recon_x, x, q_z, variational):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # mu, logvar = q_z
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if variational:
        p_z = td.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
        KLD = td.kl_divergence(q_z, p_z).sum()
    else:
        KLD = torch.zeros(size=BCE.size())

    return BCE, KLD


if __name__ == '__main__':
    config = yaml.load(
        open(Path(Path(__file__).parent.resolve(), 'config.yaml'), 'r'),
        Loader=SafeLoader
    )
    if config['wandb']: wandb.init(project="AE_MNIST", config=config)

    # prepare output folders
    datapath = Path(__file__).parent.absolute() / '..' / '..' / 'output' / 'AE'
    if not datapath.exists(): datapath.mkdir(parents=True)

    timestamp = datetime.now().strftime("%y%m%d%H%M")
    shutil.copy(Path(Path(__file__).parent.resolve(), 'config.yaml'), 
                Path(datapath, f'{timestamp}_cfg.yaml'))

    trainset = datasets.MNIST(root=Path(__file__).parent.resolve() / '..'/ '..' / 'data',
                              transform=transforms.ToTensor(), train=True, download=config['download_data'])
    testset = datasets.MNIST(root=Path(__file__).parent.resolve() / '..'/ '..' / 'data',
                             transform=transforms.ToTensor(), train=False, download=config['download_data'])
    trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True,
                             num_workers=config['num_workers'], prefetch_factor=config['batch_size']*2)
    testloader = DataLoader(testset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'],)

    #2 Net
    net = AE(
        img_shape=[1,28,28],
        hidden_dims=config['hidden_dims'],
        latent=config['latent'],
        variational=config['variational'],
        cnn=config['CNN']
    )
    
    if config['optimizer'] == 'adam':
        opt = optim.Adam(net.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'rmsprop':
        opt = optim.RMSprop(net.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'sgd':
        opt = optim.SGD(net.parameters(), lr=config['lr'])
    else:
        raise NotImplementedError()

    if config['scheduler']:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.95**epoch)

    # GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net.to(device)
    
    #3.train
    for epoch in range(config['epochs']):
        #a. train
        net.train(True)
        train_loss = []
        rec_loss = []
        kl_loss = []

        for data in trainloader:
            images, _ = data

            # forward
            _outputs, q_z = net(images)
            _rec_loss, _kl_loss = loss_function(_outputs, images, q_z, config['variational'])
            _loss = _rec_loss + _kl_loss

            #backward
            opt.zero_grad()
            _loss.backward()
            opt.step()
            train_loss.append(_loss.item())
            kl_loss.append(_kl_loss.item())
            rec_loss.append(_rec_loss.item())

        if config['scheduler']: scheduler.step()

        #b. evaluate
        net.eval()
        test_loss = []
        with torch.no_grad():
            for data in testloader:
                images, _ = data

                # forward
                _outputs, q_z = net(images)
                _rec_loss, _kl_loss = loss_function(_outputs, images, q_z, config['variational'])
                _loss = _rec_loss + _kl_loss
                test_loss.append(_loss.item())

        if config['wandb']:
            wandb.log({
                'train_loss': sum(train_loss) / len(trainloader.dataset),
                'test_loss': sum(test_loss) / len(testloader.dataset),
            })
            if config['variational']:
                wandb.log({
                    'rec_loss': sum(rec_loss) / len(trainloader.dataset),
                    'kl_loss': sum(kl_loss) / len(trainloader.dataset),
                })

        console = f"Loss of {epoch+1:02d}/{config['epochs']}"
        console += f" | Train: {sum(train_loss) / len(trainloader.dataset):.3f}"
        console += f" | Test: {sum(test_loss) / len(testloader.dataset):.3f}"
        if config['variational']:
            console += f" | Recon: {sum(rec_loss) / len(trainloader.dataset):.3f}"
            console += f" | KL: {sum(kl_loss) / len(trainloader.dataset):.3f}"
        print(console)

    #c. save
    torch.save(net, Path(datapath, f'{timestamp}_m.pkl'))

    #d. random prediction
    net.eval()
    with torch.no_grad():
        idx = np.random.randint(0, len(testset))
        img, label = testset[idx]
        img = img[None, ...]
        img0, _ = net(img)
        img = torch.cat([img, img0], dim=0)
        torchvision.utils.save_image(img, Path(datapath, f'{timestamp}_{label}.png'))

    wandb.finish()
