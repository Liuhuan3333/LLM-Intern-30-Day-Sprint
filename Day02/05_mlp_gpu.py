# 进阶挑战 —— 搭建多层感知机 (MLP) 并跑在 GPU 上
# 线性回归太简单了，只能处理直线关系。大模型处理的是极其复杂的非线性关系（比如文本、图像）。我们需要在网络里加上激活函数（如 ReLU），并把网络变深，这就是多层感知机（MLP）。

import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备非线性数据 (y = x^2 加上一点噪声)
X = torch.linspace(-1, 1, 100).reshape(-1, 1) # 100个点，形状(100, 1)
Y = X**2 + torch.randn(X.shape) * 0.1 # 抛物线数据 + 噪声

# 💡 核心：把数据搬到 GPU 上！
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = X.to(device)
Y = Y.to(device)

# 2. 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 🎯 填空部分 1：搭建网络层
        # 提示：nn.Sequential 可以把多层包在一起，按顺序执行
        # 我们需要一个输入维度1、隐藏维度10的线性层 -> 一个ReLU激活层 -> 一个输入10、输出1的线性层
        self.net = nn.Sequential(
            # 请在这里补全代码
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# 实例化模型，并把模型搬到 GPU 上！
model = MLP().to(device)

# 3. 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam 比 SGD 收敛更快

# 4. 训练循环
epochs = 200
for epoch in range(epochs):
    # 🎯 填空部分 2：标准 5 步曲
    # 请按照刚才学到的顺序，补全训练循环的代码
    # 1. 前向传播算 loss
    outputs = model(X)
    loss = criterion(outputs, Y)
    # 2. 梯度清零
    optimizer.zero_grad()
    # 3. 反向传播
    loss.backward()
    # 4. 更新参数
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f}")

print("\n训练完毕！MLP 已经学会拟合抛物线了。")