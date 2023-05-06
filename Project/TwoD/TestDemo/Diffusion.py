import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import wandb


class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()

        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(128, 3 * 32 * 32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 3, 32, 32)
        return x

if __name__ == "__main__":

    # 实例化网络
    model = DiffusionModel()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # 加载数据并划分为训练集、验证集和测试集
    data = np.load("../data4D.npy")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # 将NumPy数组转换为张量
    train_data_tensor = torch.from_numpy(train_data)
    val_data_tensor = torch.from_numpy(val_data)
    test_data_tensor = torch.from_numpy(test_data)

    # 创建DataLoader
    batch_size = 64
    train_dataset = TensorDataset(train_data_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_data_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(test_data_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    reg_coeff = 0.001  # L2正则化系数


    def calc_l2_reg(model):
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        return l2_reg


    # 训练模型
    num_epochs = 150
    learning_rate = 0.002
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    wandb.init(project="DIffusionMag",
               config={
                "Learning_rate": learning_rate,
                "Architecture": "Diffusion",
                "Dataset": "2D",
                "Num_epochs": num_epochs,
                "Batch_size": batch_size,
                "Regularization coefficient" : reg_coeff
    })

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            inputs = data[0].to(device)
            targets = data[0].to(device)  # 使用输入数据作为目标数据，WEIL重构输入

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            l2_reg = calc_l2_reg(model)
            total_loss = loss + reg_coeff * l2_reg

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
        wandb.log({"Epocch": epoch + 1, "Training loss": epoch_loss / len(train_loader)})
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss / len(train_loader)}')

        # 验证模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                inputs = data[0].to(device)
                targets = data[0].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        wandb.log({"Epocch": epoch + 1, "Validation loss": val_loss / len(val_loader)})
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss / len(val_loader)}')

    wandb.finish()
    torch.save(model)