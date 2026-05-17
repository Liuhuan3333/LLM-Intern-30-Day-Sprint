# Autograd 自动微分 —— 让 AI 拥有“反思”的能力
import torch

# ================= 1. 追踪梯度 =================
print("=== 1. 追踪梯度 ===")
# requires_grad=True 是灵魂！告诉 PyTorch：请记住对这个变量做的所有运算，等下我要算导数！说明梯度追踪已开启
x = torch.tensor([2.0, 3.0], requires_grad=True)
print("x:", x)

# 做一些运算
y = x ** 2      # y = x^2
z = y.sum()     # z = x1^2 + x2^2

print("y:", y)
print("z:", z)


# ================= 2. 反向传播求导 =================
print("\n=== 2. 反向传播求导 ===")
# 调用 backward()，PyTorch 会自动从 z 一路倒推算出 dz/dx
z.backward()

# 查看导数（梯度）存在了 x.grad 里
# 数学验证：z = x^2，对 x 求导是 2x。x=[2,3]，所以导数应该是 [4, 6]
print("x 的梯度 (dz/dx):", x.grad) 


# ================= 3. 大坑警告：更新权重时不要记录梯度 =================
print("\n=== 3. 关闭梯度追踪 ===")
# 神经网络在更新权重时，这个“更新动作”本身不应该被记录到计算图里，否则会越算越乱！
# 所以更新权重时，必须放在 with torch.no_grad(): 下面
with torch.no_grad():
    # 假设学习率是 0.1，我们要更新 x
    x_new = x - 0.1 * x.grad
    print("更新后的 x:", x_new)
    print("更新后的 x 是否需要梯度:", x_new.requires_grad) # 默认变成了 False