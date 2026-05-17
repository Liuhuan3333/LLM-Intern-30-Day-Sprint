# 手写线性回归（不用任何 nn.Module）

import torch

# 1. 准备训练数据 (模拟真实世界的数据)
# 假设真实规律是 y = 3x + 1
X = torch.tensor([1.0, 2.0, 3.0, 4.0])
Y = torch.tensor([4.0, 7.0, 10.0, 13.0]) # 3*1+1=4, 3*2+1=7...

# 2. 初始化模型参数 (权重 w 和偏置 b)
# 我们假装不知道 w=3, b=1，随机初始化一个，让 AI 自己学出来
# requires_grad=True 极其重要！
w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# 3. 设置超参数
learning_rate = 0.01 # 学习率：每次调整参数的步伐大小
epochs = 100         # 把所有数据学几遍

# 4. 训练循环
for epoch in range(epochs):
    # ===== 步骤 A：前向传播 (用当前的 w, b 算出预测值) =====
    y_pred = w * X + b
    
    # ===== 步骤 B：计算损失 (预测值和真实值差多远，用均方误差 MSE) =====
    # MSE = ((y_pred - Y)^2 的平均值)
    loss = ((y_pred - Y) ** 2).mean()
    
    # ===== 步骤 C：反向传播 (自动算出 loss 对 w 和 b 的梯度) =====
    # 每次算梯度前，必须把上一次的梯度清零！不然梯度会累加！
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()
        
    loss.backward()
    
    # ===== 步骤 D：更新参数 (朝着梯度的反方向走一步) =====
    # 请在下面补全参数更新代码！
    with torch.no_grad():
        # 提示：w 的新值 = w 的旧值 - 学习率 * w的梯度
        w -= learning_rate * w.grad
        # 提示：b 的新值 = b 的旧值 - 学习率 * b的梯度
        b -= learning_rate * b.grad

    # 每 10 轮打印一次进度
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | w: {w.item():.4f} | b: {b.item():.4f}")

print("\n训练完毕！学到的规律: y = {:.2f}x + {:.2f}".format(w.item(), b.item()))