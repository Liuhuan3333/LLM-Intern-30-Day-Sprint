# 用 PyTorch 标准库重写线性回归

import torch
import torch.nn as nn  # nn 是 Neural Network 神经网络模块的缩写
import torch.optim as optim # optim 是优化器模块

# 1. 准备数据 (同上)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]]) # 注意这里改成了二维 [[1], [2]...]
Y = torch.tensor([[4.0], [7.0], [10.0], [13.0]])

# 2. 定义模型 (替代手写的 w, b 和 y_pred = w*X+b)
class LinearModel(nn.Module): # 继承昨天学的 nn.Module！
    def __init__(self):
        super().__init__() # 必须写这行！
        # nn.Linear(输入特征维度, 输出特征维度) 
        # 它内部自动帮我们创建了权重 w 和偏置 b，且默认 requires_grad=True
        self.linear = nn.Linear(1, 1) 
        
    def forward(self, x):
        # 前向传播，只需把数据丢进 self.linear 即可
        return self.linear(x)

model = LinearModel() # 实例化模型

# 3. 定义损失函数和优化器 (替代手写的 MSE 计算和 w -= lr * w.grad)
criterion = nn.MSELoss() # 均方误差损失函数
# 随机梯度下降优化器，告诉它要优化谁的参数，学习率是多少
optimizer = optim.SGD(model.parameters(), lr=0.01) 

# 4. 训练循环 (标准 5 步曲，一个字都不能少！)
epochs = 200
for epoch in range(epochs):
    # 步骤 1：前向传播
    outputs = model(X) # 自动调用 forward 函数
    loss = criterion(outputs, Y) # 算损失
    
    # 步骤 2：梯度清零 (替代手写的 w.grad.zero_())
    optimizer.zero_grad() 
    
    # 步骤 3：反向传播求梯度
    loss.backward()
    
    # 步骤 4：更新参数 (替代手写的 w -= lr * w.grad)
    optimizer.step() 
    
    # 步骤 5：打印日志
    if (epoch + 1) % 10 == 0:
        # 从模型中提取出学到的 w 和 b
        w_learned = model.linear.weight.item()
        b_learned = model.linear.bias.item()
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | w: {w_learned:.4f} | b: {b_learned:.4f}")

print("\n训练完毕！学到的规律: y = {:.2f}x + {:.2f}".format(w_learned, b_learned))