# 纯手写 Self-Attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. 设置超参数
batch_size = 2    # 也就是同时处理2个句子
seq_len = 5       # 每个句子5个词
d_model = 64      # 每个词用64维的向量表示 (也就是QKV的维度)

# 模拟输入：一个形状为 (batch_size, seq_len, d_model) 的张量
# 就像是已经把文字转成了数字向量
x = torch.randn(batch_size, seq_len, d_model)

# 2. 定义可学习的 Q, K, V 矩阵
# 在 Transformer 中，QKV 不是固定的，而是通过线性变换从 x 学出来的
# nn.Linear(d_model, d_model) 就相当于乘以一个权重矩阵 W
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

# 🎯 填空部分 1：生成 Q, K, V
# 提示：把 x 分别丢进 W_q, W_k, W_v 得到 q, k, v
q = W_q(x)
k = W_k(x)
v = W_v(x)

print(f"Q 的形状: {q.shape}") # 期望输出: torch.Size([2, 5, 64])
print(f"K 的形状: {k.shape}") # 期望输出: torch.Size([2, 5, 64])
print(f"V 的形状: {v.shape}") # 期望输出: torch.Size([2, 5, 64])


# 3. 计算 Attention Scores (搭讪指数)
# 🎯 填空部分 2：Q 和 K 的转置相乘
# 提示：q 的形状是 (2, 5, 64)，k 的形状是 (2, 5, 64)
# 我们需要让每个 q 和每个 k 做点积，所以要把 k 的最后两个维度转置
# 使用 torch.bmm (批量矩阵乘法) 或 q @ k.transpose(-2, -1)
scores = q @ k.transpose(-2, -1)  # .transpose(-2, -1) 是把 k 的最后两个维度转置，变成 (2, 64, 5)

# 4. 缩放
# 🎯 填空部分 3：除以根号 d_k
# 提示：这是为了防止点积结果过大导致 softmax 梯度消失
# d_k 就是 d_model 的维度，即 64
scores = scores / math.sqrt(d_model)

# 5. Softmax 归一化
# 让每一行的搭讪指数加起来等于 1
# dim=-1 表示在最后一个维度（5个词）上做 softmax
attention_weights = F.softmax(scores, dim=-1)

print(f"\n注意力权重的形状: {attention_weights.shape}") # 期望: torch.Size([2, 5, 5])
# 打印第一个句子的第一个词对其他词的注意力权重，加起来应该等于1
print(f"第一个词的注意力权重: {attention_weights[0, 0, :]}") 
print(f"权重和: {attention_weights[0, 0, :].sum().item()}") 

# 6. 将注意力权重应用到 V 上
# 🎯 填空部分 4：加权求和
# 提示：用 attention_weights 和 v 做矩阵乘法
output = attention_weights @ v

print(f"\n最终输出的形状: {output.shape}") # 期望: torch.Size([2, 5, 64])