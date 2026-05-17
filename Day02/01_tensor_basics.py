# PyTorch Tensor —— 能上 GPU 的超级 NumPy
# PyTorch 的 Tensor 和昨天的 NumPy ndarray 几乎一模一样，但它有一个超能力：可以在 GPU 上飞快地计算。大模型训练全靠 GPU 加速。

import torch  # torch是PyTorch的核心库，提供了Tensor数据结构和各种操作函数，是我们进行深度学习的基础工具。
import numpy as np  # 用于数据预处理，处理完后转成Tensor喂给模型

# ================= 1. 创建 Tensor =================
print("=== 1. 创建 Tensor ===")  # Tensor本质上就是“能在GPU上跑的NumPy数组”，数据结构几乎一样，但多了一个关键能力：GPU加速！所以创建Tensor的方式也和NumPy类似，但要注意 dtype 和 device 参数。
# 从列表创建
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print("从列表创建:\n", a)

# 从 NumPy 创建 (常用：处理数据用Pandas/NumPy，训练转成Tensor)
np_array = np.array([5, 6, 7])  
b = torch.from_numpy(np_array)  # torch.from_numpy()把NumPy数组转换成Tensor，注意它们共享内存，修改一个会影响另一个！如果不想共享，可以用 torch.tensor(np_array) 来创建一个新的Tensor。
print("从 NumPy 创建:", b)

# 常用的初始化方式
c = torch.zeros(2, 3)   # 2行3列全0
d = torch.randn(2, 3)   # 2行3列正态分布随机数 (大模型权重初始化经常用这个)
print("随机正态分布:\n", d)


# ================= 2. Tensor 与 NumPy 的无缝转换 =================
print("\n=== 2. 与 NumPy 互转 ===")
# Tensor 转 NumPy
tensor_to_numpy = d.numpy()  # .numpy() 方法把 Tensor 转成 NumPy 数组，注意如果这个 Tensor 在 GPU 上，这个方法会报错！必须先把 Tensor 移动到 CPU 上（d.cpu().numpy()）才能转换。
print("类型:", type(tensor_to_numpy))


# ================= 3. 核心魔法：CPU 与 GPU 互转 =================
print("\n=== 3. GPU 加速 ===")
# 检查是否有 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"  # 三元表达式，如果 torch.cuda.is_available() 返回 True 就用 "cuda"，否则用 "cpu"
print(f"当前使用设备: {device}")

# 把 Tensor 移动到 GPU 上
x = torch.randn(3, 3)
x_gpu = x.to(device)  # 或者 x.cuda()，把 Tensor 从 CPU 移动到 GPU 上，注意这个操作会返回一个新的 Tensor，原来的 x 仍然在 CPU 上。
print("Tensor 所在设备:", x_gpu.device)

# 💡 导师提示：只有同在 GPU 上的 Tensor 才能互相计算！
# x (CPU) + x_gpu (GPU) 会报错！必须保证计算的两端在同一设备上。