import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 定义模型类 CNNAE
class CNNAE(nn.Module):
    def __init__(self):
        super(CNNAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(64*12*12, 1024)
        self.fc2 = nn.Linear(1024, 64*12*12)
        # 解码器
        self.decoder = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 64, 12, 12)
        x = self.decoder(x)
        return x

# 加载数据
data = np.load("E:\MagneticFields\Project\TwoD\large94_2D.npy", allow_pickle=True)
data_2D = torch.from_numpy(data).float()
mean = torch.mean(data_2D, dim=(0, 2, 3))
std = torch.std(data_2D, dim=(0, 2, 3))

# 归一化函数
class Normalize3D:
    def __init__(self, mean, std):
        self.mean = mean.view(1, 3, 1, 1)
        self.std = std.view(1, 3, 1, 1)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

# 物理约束损失函数
def compute_physical_constraint_loss2(output):
    Hx_x = torch.gradient(output[:, 0], axis=2)[0]
    Hy_y = torch.gradient(output[:, 1], axis=1)[0]
    if output.shape[1] == 3:
        Hz_z = torch.gradient(output[:, 2], axis=2)[0]
        div_mag = torch.stack([Hx_x, Hy_y, Hz_z], dim=1)
    else:
        div_mag = torch.stack([Hx_x, Hy_y], dim=1)
    constraint_loss = torch.mean(torch.abs(div_mag.sum(dim=1)))
    return constraint_loss

# 定义训练函数
def train(model, train_loader, optimizer, criterion, alpha):
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        reconstruction_loss = criterion(outputs, data)
        constraint_loss = compute_physical_constraint_loss2(outputs)
        loss = reconstruction_loss + alpha * constraint_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(train_loader.dataset)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    # 初始化WandB
    wandb.init(project="MagAE_2D")

    # 设置超参数搜索空间
    sweep_config = {
        "method": "random",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"values": [0.001, 0.0005, 0.0001]},
            "batch_size": {"values": [32, 64, 128]},
            "alpha": {"values": [0.01, 0.1, 1.0]}
        }
    }

    # 初始化sweep
    sweep_id = wandb.sweep(sweep_config)

    # 运行sweep
    def sweep_train():
        # 获取超参数配置
        config_defaults = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "alpha": 0.1
        }
        wandb.init(config=config_defaults)
        config = wandb.config

        # 创建模型和优化器
        model = CNNAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        # 数据加载和归一化
        train_data, test_data = train_test_split(data_2D, test_size=0.2, random_state=42)
        normalizer_train_data = Normalize3D(mean=mean, std=std)(train_data)
        train_loader = DataLoader(normalizer_train_data, batch_size=config.batch_size, shuffle=True)
        num_epochs = 200
        # 训练
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, criterion, config.alpha)
            wandb.log({"Epochs": epoch + 1, "loss": train_loss})
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")

    wandb.agent(sweep_id, function=sweep_train)
