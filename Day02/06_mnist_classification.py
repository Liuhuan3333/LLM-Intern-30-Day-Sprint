# MNIST 手写数字分类 (PyTorch 的 Hello World)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 超参数设置
batch_size = 64  # 每次给模型看64张图片
learning_rate = 0.001
epochs = 5       # 先跑5轮看看效果

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据准备 (MNIST)
# transforms.ToTensor() 能把图片变成 Tensor，并且把像素值从 0-255 缩放到 0-1
transform = transforms.ToTensor()

# 下载训练集 (如果本地没有，会自动下载到 data 文件夹)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 💡 核心：DataLoader 负责分批、打乱数据
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # shuffle=True 每轮训练前打乱数据，防止模型记住顺序

# 3. 定义 MLP 模型 (图像分类版)
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 🎯 填空部分 1：搭建分类网络
        # 提示：MNIST 图片是 28x28 像素，展平后输入维度是 784
        # 我们要分 0-9 共 10 个数字，所以输出维度是 10
        self.net = nn.Sequential(
            # 请在这里补全代码
            # 结构建议：输入784 -> 隐藏128 (加ReLU) -> 隐藏64 (加ReLU) -> 输出10
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        # 输入的 x 形状是 (batch_size, 1, 28, 28)
        # 我们需要把中间的通道维度和长宽展平，变成 (batch_size, 784) 才能喂给全连接层
        x = x.view(x.size(0), -1) # x.size(0) 是 batch_size，-1 让电脑自动算剩下的维度
        return self.net(x)

model = MNISTClassifier().to(device)

# 4. 损失函数和优化器
# 🎯 填空部分 2：分类任务用什么损失函数？
# 提示：多分类任务用交叉熵，PyTorch 里是 nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环 (加入 DataLoader)
total_steps = len(train_loader) # 一共有多少个 batch
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader): # 每次取出一个 batch 的图片和标签，enumerate给每个 batch 编号 i
        # 把数据搬到 GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # 🎯 填空部分 3：标准 5 步曲
        # 1. 前向传播算 outputs
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 2. 算 loss
        # 3. 梯度清零
        optimizer.zero_grad()
        # 4. 反向传播
        loss.backward()
        # 5. 更新参数
        optimizer.step()

        
        # 打印训练进度
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

print("\n训练完毕！你的模型已经学会识别手写数字了！")