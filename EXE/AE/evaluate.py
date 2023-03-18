import yaml
from yaml.loader import SafeLoader
from pathlib import Path
from argparse import ArgumentParser
from torchvision import datasets, transforms

import torch
import torchvision
from ae import AE, AEEncoder, AEDecoder, CNN_Encoder, CNN_Decoder


timestamp = '2303180922'
output_path = Path(__file__).parent.resolve() / '..'/ '..' / 'output' / 'AE'

parser = ArgumentParser()
parser.add_argument(
    '--config', type=str,
    default=Path(output_path, f'{timestamp}_cfg.yaml'),
    help="test configuration"
)

args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=SafeLoader)

# Load the model
net = AE(
    img_channel=1,
    img_height=28,
    img_width=28,
    hidden_dims=config['hidden_dims'],
    latent=config['latent'],
    cnn=config['CNN']
)
net = torch.load(Path(output_path, f'{timestamp}_m.pkl'))


# Define the transform to normalize the data
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,))])

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=(28, 28), scale=(0.85, 1.0))
    ])

# Load the MNIST test dataset
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
loss_fn = torch.nn.MSELoss()

# Extract a random sample from the test dataset
eval_loss = []
for i in range(10):
    # index = np.random.randint(0, len(testset))
    img, label = testset[i]
    img = img[None, ...]
    output = net(img)
    loss = loss_fn(output, img)
    eval_loss.append(loss.item())

    img = torch.cat([img, output], dim=0)
    torchvision.utils.save_image(img, Path(output_path, f'test_{i}_{label}.png'))

print(f'Reconstruction error: {sum(eval_loss) / len(eval_loss):.3f}')
