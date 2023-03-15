import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import torch
from ae import AE, AEEncoder, AEDecoder



# Load the model
model = torch.load('MagneticFields/EXE/AE/output/models/m.pkl')


# Define the transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST test dataset
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Extract a random sample from the test dataset
index = np.random.randint(0, len(testset))
sample, _ = testset[index]
image = sample.numpy().squeeze()

# Reshape and normalize the sample for input to the model
sample = sample.view(1, -1)
sample = (sample - 0.5) / 0.5

# Use the model to reconstruct the image
reconstruction = model(sample)

# Convert the tensor to a numpy array and reshape to an image
reconstruction = reconstruction.detach().numpy().squeeze()
reconstruction = (reconstruction + 1) / 2.0

# Compute the reconstruction error
mse_loss = torch.nn.MSELoss()
loss = mse_loss(torch.tensor(reconstruction), torch.tensor(image))

print('Reconstruction Error:', loss.item())
