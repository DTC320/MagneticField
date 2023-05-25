import os
import h5py
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


# 定义模型类 CNNAE
class CNNAE(nn.Module):
    def __init__(self, cnum):
        super(CNNAE, self).__init__()
        # 编码器
        self.cnum = cnum
        self.encoder = nn.Sequential(
            nn.Conv2d(3, cnum, 3, stride=2, padding=1),
            nn.BatchNorm2d(cnum),
            nn.PReLU(),
            nn.Conv2d(cnum, cnum*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(cnum*2),
            nn.PReLU(),
            nn.Conv2d(cnum*2, cnum*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(cnum*4),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(cnum*4*12*12, 1024)
        )
        self.fc_z = nn.Linear(1024, cnum*4*12*12)
        # 解码器
        self.decoder = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(cnum*4, cnum*2, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(cnum*2),
            nn.PReLU(),
            nn.ConvTranspose2d(cnum*2, cnum, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(cnum),
            nn.PReLU(),
            nn.ConvTranspose2d(cnum, 3, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.fc_z(z)
        x = x.view(x.size(0), self.cnum*4, 12, 12)
        x = self.decoder(x)

        return x


# 物理约束损失函数
def div_loss(f, grad_z):
    Hx_x = torch.gradient(f[:,0], dim=2)[0]
    Hy_y = torch.gradient(f[:,1], dim=1)[0]
    div_mag = torch.stack([Hx_x, Hy_y, grad_z[:,2]], dim=1)

    return torch.mean(torch.abs(div_mag.sum(dim=1)))


def curl_loss(f, grad_z):
    Hx_y = torch.gradient(f[:,0], dim=1)[0]
    Hy_x = torch.gradient(f[:,1], dim=2)[0]
    Hz_x = torch.gradient(f[:,2], dim=2)[0]
    Hz_y = torch.gradient(f[:,2], dim=1)[0]
    curl_vec = torch.stack([Hz_y - grad_z[:,1], grad_z[:,0] - Hz_x, Hy_x - Hx_y], dim=1)
    curl_mag = curl_vec.square().sum(dim=1)

    return torch.mean(curl_mag)


def mape_loss(output, target):
    return (torch.mean(torch.abs(target - output)) / torch.mean(torch.abs(target))) * 100


class MagneticFieldDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, scaling):
        super(MagneticFieldDataset, self).__init__()
        self.db_path = datapath
        self.scaling = scaling
        self.size = h5py.File(self.db_path, mode='r')['field'].shape[0]
 
    def open_hdf5(self):
        self.db = h5py.File(self.db_path, mode='r')

    def __getitem__(self, idx):
        if not hasattr(self, 'db'):
            self.open_hdf5()
    
        f = self.db['field'][idx]
        fx_z = np.gradient(f[0], axis=2)[:,:,1]
        fy_z = np.gradient(f[1], axis=2)[:,:,1]
        fz_z = np.gradient(f[2], axis=2)[:,:,1]
        grad_z = np.stack((fx_z, fy_z, fz_z), axis=0)
        grad_z = torch.from_numpy(grad_z.astype('float32'))
        f_t = torch.from_numpy(f[:,:,:,1].astype('float32'))

        return f_t * self.scaling, grad_z

    def __len__(self):
        return self.size


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.realpath(__file__))
    data_path = file_path + "/../../../data/tianchi_magfield_3D_96.h5"

    cfg = {
        "lr": 0.0001,
        "batch_size": 64,
        "div_alpha": 0.1,
        "curl_alpha": 1,
        "cnum": 16,
        "scaling": 10,
        "num_epochs": 200
    }

    # 初始化WandB
    wandb.init(entity='te-st', project='MagAE_2D_hp-tuning', config=cfg)
    cfg = wandb.config

    # 创建模型和优化器
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    model = CNNAE(cfg['cnum']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.MSELoss()

    dataset = MagneticFieldDataset(data_path, cfg['scaling'])
    train_data, test_data = random_split(dataset, [0.8, 0.2], torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    n_batch = len(train_loader.dataset) // cfg['batch_size']
    n_test_batch = len(test_loader.dataset)

    # 训练
    for epoch in range(cfg['num_epochs']):
        model.train()
        total_loss = 0.0
        total_rec = 0.0
        total_div = 0.0
        total_curl = 0.0
        it = 0

        for field, grad_z in train_loader:
            field = field.to(device)
            grad_z = grad_z.to(device)
            optimizer.zero_grad()
            outputs = model(field)
            rec = criterion(outputs, field)
            div = div_loss(outputs, grad_z)
            curl = curl_loss(outputs, grad_z)
            loss = rec + cfg['div_alpha'] * div + cfg['curl_alpha'] * curl
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rec += rec.item()
            total_div += div.item()
            total_curl += curl.item()
            it += 1

        wandb.log({
            "loss": total_loss / n_batch,
            "MSE": total_rec / n_batch,
            "div": total_div / n_batch,
            "curl": total_curl / n_batch
        })

    model.eval()
    with torch.no_grad():
        total_mape = 0
        for batch in test_loader:
            input_images = batch.to(device)
            outputs = model(input_images)
            total_mape += mape_loss(outputs, input_images).item()

        wandb.log({"test_mape": total_mape / n_test_batch,})
