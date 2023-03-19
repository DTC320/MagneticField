#%%
import numpy as np
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
from scipy.stats import norm

import torch
import torchvision
from torchvision import datasets, transforms

from ae import AE, AEEncoder, AEDecoder, CNN_Encoder, CNN_Decoder

def eval_ae(timestamp):
    output_path = Path(__file__).parent.resolve() / '..'/ '..' / 'output' / 'AE'
    config = yaml.load(open(Path(output_path, f'{timestamp}_cfg.yaml'), 'r'), Loader=SafeLoader)

    net = AE(
        img_shape=[1,28,28],
        hidden_dims=config['hidden_dims'],
        latent=config['latent'],
        variational=config['variational'],
        cnn=config['CNN']
    )
    net = torch.load(Path(output_path, f'{timestamp}_m.pkl'))


    # Define the transform to normalize the data
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])

    transform = transforms.Compose([transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(28, 28), scale=(0.85, 1.0))])

    # Load the MNIST test dataset
    testset = datasets.MNIST(root=Path(__file__).parent.resolve() / '..'/ '..' / 'data',
                             train=False, download=False, transform=transform)
    loss_fn = torch.nn.MSELoss()

    # Extract a random sample from the test dataset
    eval_loss = []
    for i in range(3):
        # index = np.random.randint(0, len(testset))
        img, label = testset[i]
        img = img[None, ...]
        output, _ = net(img)
        loss = loss_fn(output, img)
        eval_loss.append(loss.item())

        img = torch.cat([img, output], dim=0)
        torchvision.utils.save_image(img, Path(output_path, f'test_{i}_{label}.png'))

    print(f'Reconstruction error: {sum(eval_loss) / len(eval_loss):.3f}')

    if config['variational']:
        img = torch.zeros(size=[config['batch_size'], *testset[0][0].size()])
        label = torch.zeros(size=[config['batch_size']])
        for i in range(config['batch_size']):
            img[i], label[i] = testset[i]
        _, z = net(img, eval=True)
        z_np = z.detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.scatter(z_np[:100, 0], z_np[:100, 1], c=label[:100], cmap='brg')
        plt.colorbar()
        plt.show()

        # Display a 2D manifold of the digits | figure with 20x20 digits
        n = 20
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        # Construct grid of latent variable values
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        # decode for each square in the grid
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float32)
                z_sample = torch.tile(z_sample, (config['batch_size'], config['latent'] // 2))
                img_decoded, _ = net(img, z=z_sample)
                digit = img_decoded[0].detach().numpy().reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='gnuplot2')
        plt.show()
# %%

if __name__ == '__main__':
    eval_ae('2303182248')