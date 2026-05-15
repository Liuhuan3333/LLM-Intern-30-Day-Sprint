# NumPy核心（张量运算的基石）

import numpy as np

# ================= 1. 致命易错点：矩阵乘法 vs 逐元素相乘 =================
print("=== 1. 两种乘法的区别 ===")
A = np.array([[1, 2], 
              [3, 4]])
B = np.array([[5, 6], 
              [7, 8]])

# 1.1 逐元素相乘：用 *
# 对应位置的元素相乘，形状必须一致
element_wise = A * B
print("A * B (逐元素相乘):\n", element_wise)
# 计算过程：[[1*5, 2*6], [3*7, 4*8]]

# 1.2 矩阵乘法：用 @ 或 np.dot
# 这是线性代数里的矩阵乘法，大模型里算 Q*K^T 就用这个！
matrix_mul = A @ B  # 等同于 np.dot(A, B)
print("\nA @ B (矩阵乘法):\n", matrix_mul)
# 计算过程：[[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]


# ================= 2. 神奇的广播机制 =================
print("\n=== 2. 广播机制 ===")
# 广播：当两个形状不同的数组进行运算时，NumPy会自动把小的数组“拉伸”成大的数组的形状
C = np.array([[1, 2, 3],   # 形状: (2, 3)  -> 2行3列
              [4, 5, 6]])
D = np.array([10, 20, 30]) # 形状: (3,)    -> 1行3列 (一维)

# C + D，按理说形状不一样不能加，但NumPy把D当成了[[10,20,30], [10,20,30]]去加
broadcast_result = C + D
print("C + D (广播):\n", broadcast_result)

# 大模型中的应用：算完 QK^T 后要除以 sqrt(d_k)
# QK^T 是一个大矩阵，sqrt(d_k) 是一个标量（单个数字），直接用除法，NumPy会自动把标量广播到矩阵每个元素
d_k = 64
QKT = np.array([[10.0, 20.0], [30.0, 40.0]])
# 自动广播：每个元素都除以了 8.0
scaled_QKT = QKT / np.sqrt(d_k) 
print("\n缩放后的QK^T:\n", scaled_QKT)


# ================= 3. 维度变换：reshape =================
print("\n=== 3. 维度变换 reshape ===")
# 大模型中经常需要改变数据的维度，比如把一维的句子向量变成二维的批次向量
E = np.arange(12) # 生成 0 到 11 的一维数组
print("原始一维数组 E:\n", E)

# 变成 3行4列 的二维矩阵
E_2d = E.reshape(3, 4)
print("\nreshape(3, 4) 后:\n", E_2d)

# 变成 2行2列3深 的三维张量（常用于表示：批次 x 句子长度 x 词向量维度）
E_3d = E.reshape(2, 2, 3)
print("\nreshape(2, 2, 3) 后:\n", E_3d)


# 练习：模拟计算 Self-Attention 的核心公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
# 要求：
# 1. 创建矩阵 Q，内容是 [[1, 0], [0, 1]]，形状是 (2, 2)。
# 2. 创建矩阵 K，内容是 [[1, 2], [3, 4]]，形状是 (2, 2)。
# 3. 创建矩阵 V，内容是 [[5, 6], [7, 8]]，形状是 (2, 2)。
# 4. 设定 d_k = 2。
# 5. 计算 Q 和 K 的转置 (K.T) 的矩阵乘法，存为 QKT。打印结果。
# 6. 计算 QKT / np.sqrt(d_k)，存为 scaled_QKT。打印结果。
# 7. 计算 scaled_QKT 和 V 的矩阵乘法，存为 attention_score。打印结果。
Q = np.array([[1, 0],
              [0, 1]])
K = np.array([[1, 2],
              [3, 4]])
V = np.array([[5, 6],
              [7, 8]])
d_k = 2
QKT = Q @ K.T
print("\nQK^T:\n", QKT)
scaled_QKT = QKT / np.sqrt(d_k)
print("\nQK^T / sqrt(d_k):\n", scaled_QKT)
attention_score = scaled_QKT @ V
print("\nAttention(Q, K, V) 的结果:\n", attention_score)