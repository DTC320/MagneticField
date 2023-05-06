import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from CNNVAE import VAE

# 加载数据集
data = np.load("Megnat_3D.npy")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 将NumPy数组转换为张量
train_data_tensor = torch.from_numpy(train_data)
test_data_tensor = torch.from_numpy(test_data)

# 创建DataLoader
batch_size = 64
train_dataset = TensorDataset(train_data_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_data_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载VAE模型
vae_model = torch.load("your_vae_model.pth")

def eval_vae(vae_model, test_loader):
    vae_model.eval()
    recon_loss_fn = torch.nn.MSELoss(reduction='none')

    pca = PCA(n_components=2)
    latent_vectors = []
    labels = []

    for batch_idx, (data,) in enumerate(test_loader):
        data = data.float()
        with torch.no_grad():
            recon_batch, mu, logvar = vae_model(data)
            z = vae_model.reparameterize(mu, logvar)

            recon_loss = recon_loss_fn(recon_batch, data).mean(dim=[1, 2, 3, 4]).numpy()
            latent_vectors.append(z.numpy())
            labels.append(np.full((len(data),), batch_idx))

        if batch_idx < 10:
            for idx in range(len(data)):
                original_img = data[idx].numpy().transpose(1, 2, 3, 0)
                recon_img = recon_batch[idx].numpy().transpose(1, 2, 3, 0)

                # 显示原始图像和重建图像
                fig, axes = plt.subplots(1, 2)
                axes[0].imshow(original_img[..., 0], cmap='jet')
                axes[1].imshow(recon_img[..., 0], cmap='jet')
                axes[0].set_title("Original")
                axes[1].set_title("Reconstructed")
                plt.show()

    latent_vectors = np.concatenate(latent_vectors)
    labels = np.concatenate(labels)

    # 使用PCA将潜在空间投影到2D
    projected_vectors = pca.fit_transform(latent_vectors)

    # 可视化2D潜在空间
    plt.scatter(projected_vectors[:, 0], projected_vectors[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title("Latent Space")
    plt.show()

eval_vae(vae_model, test_loader)
