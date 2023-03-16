import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
import os


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
                    nn.Conv2d(in_features, dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True)
                )
            )
            in_features = dim
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features * 8 * 8, out_features))
        self.encoder = nn.Sequential(*layers)

    def forward(self, imgs):
        """
        编码器
        :param imgs: [N, in_features]
        :return: [N, out_features]
        """
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
                    nn.Conv2d(in_features, dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU()
                )
            )
            in_features = dim
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features * 8 * 8, out_features))
        self.encoder = nn.Sequential(*layers[::-1])

    def forward(self, imgs):
        """
        编码器
        :param imgs: [N, in_features]
        :return: [N, out_features]
        """
        return self.encoder(imgs)


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
    

    batch_size = 16

    # Load data

    file_name = 'data4D.npy'
    folder_path = 'MagneticFields/Project/code'

    # 使用 os.path.join 函数构建跨平台的文件路径
    file_path = os.path.join(folder_path, file_name)


    if not os.path.exists(file_path):
        print("File not found:", file_path)
    else:
        # 读取数据集
        data = np.load(file_path)
        
    data = torch.tensor(data, dtype=torch.float32)

        
    class MyDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform
                
        def __len__(self):
            return len(self.data)
                
        def __getitem__(self, index):
            x = self.data[index]
            if self.transform:
                x = self.transform(x)
            return x



    transform = transforms.Compose([transforms.ToTensor()])

   
    # 假设你的数据集为 data，包含了所有的数据
    dataset = MyDataset(data, transform=None)

    # 划分训练集和测试集
    train_size = int(len(dataset) * 0.8)  # 80% 作为训练集
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    #创建 dataloader
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    #2 Net
    net = AE(1,32,32)
    loss_fn = nn.MSELoss()
    opt = optim.SGD(net.parameters(), lr = 0.001)

    # GPU
    """device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net.to(device)
    loss_fn.to(device)"""
    

    #3.train
    total_epoch = 10
    train_step = 0
    test_step = 0
    for epoch in range(total_epoch):
        #a. train
        net.train(True)
        train_loss = []
        for data in trainloader:
            images = data

            # forward
            _output = net(images)
            _loss = loss_fn(_output, images)
            #backward
            opt.zero_grad()
            _loss.backward()
            opt.step()

            if train_step % 100 == 0:
                #wandb.log({'train_loss': _loss})
                print(f"Train {epoch + 1}/{total_epoch}  {train_step} loss:{_loss.item():.3f}")
            train_step += 1

        #b. evaluate
        net.eval()
        with torch.no_grad():
            for data in testloader:
                images = data

                # forward
                _outputs = net(images)
                _loss = loss_fn(_outputs, images)

                if test_step % 10 == 0:
                    #wandb.log({'test_loss': _loss})
                    print(f"Test {epoch + 1}/{total_epoch}  {test_step} loss:{_loss.item():.3f}")
                test_step += 1


        #c. save
        if epoch % 2 == 0:
            torch.save(net, f'./Project/code/output/models/m_{epoch}.pkl')
        
    torch.save(net, f'./Project/code/output/models/m.pkl')  