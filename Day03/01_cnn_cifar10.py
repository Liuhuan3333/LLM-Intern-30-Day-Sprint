# CNN 实战

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 超参数与设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.001
epochs = 5

# 2. 数据准备 (CIFAR-10 彩色图)
# 彩色图有 R,G,B 三个通道，Normalize 里的三个数分别是 R,G,B 通道的均值和标准差
# Compose 是“组合”的意思，把多个预处理步骤串在一起，按顺序执行
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化，让像素值从 -1 到 1
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  #DataLoader 负责分批、打乱数据

# 3. 定义 CNN 模型
class SimpleCNN(nn.Module):  # CNN分为两大块：特征提取器（卷积层）和分类器（全连接层）
    def __init__(self):
        super().__init__()
        # 🎯 填空部分 1：搭建卷积层特征提取器
        # 提示：
        # nn.Conv2d(输入通道数, 输出通道数(卷积核个数), 卷积核大小)
        # CIFAR-10 是彩色图，输入通道数是 3
        # 第一层卷积：输入3通道，输出16通道，3x3的卷积核
        # 第二层卷积：输入16通道，输出32通道，3x3的卷积核
        # nn.MaxPool2d(2, 2) 会在长宽上减半 (32x32 -> 16x16)
        self.features = nn.Sequential(
            # 请在这里补全第一层卷积 + ReLU + 池化
            nn.Conv2d(3, 16, 3), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 最大池化，在2✖️2的窗口内取最大值，步长也是2，输出尺寸减半
            # 请在这里补全第二层卷积 + ReLU + 池化
            nn.Conv2d(16, 32, 3), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 🎯 填空部分 2：搭建全连接层分类器
        # 提示：经过两次卷积+池化后，特征图的形状变成了 (batch_size, 32, 6, 6)
        # 展平后是 32 * 6 * 6 = 1152 维
        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # CIFAR-10 有 10 个类别
        )

    def forward(self, x):
        x = self.features(x)          # 提取特征
        x = x.view(x.size(0), -1)     # 展平成一维
        x = self.classifier(x)        # 分类
        return x

model = SimpleCNN().to(device)

# 4. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练循环 (标准五步曲，照抄昨天的！)
total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 🎯 填空部分 3：写出标准五步曲
        # 1. 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 2. 梯度清零
        optimizer.zero_grad()
        # 3. 反向传播
        loss.backward()
        # 4. 更新参数
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

print("\nCNN 训练完毕！")