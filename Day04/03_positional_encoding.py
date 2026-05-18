import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建一个足够长的位置矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)  # max_len = 5000, d_model = 64
        
        # 生成位置索引 (0, 1, 2, 3...)，并增加一个维度变成列向量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # unsequeeze(1)把一维变成二维（列向量）
        
        # 计算分母的缩放因子 (频率)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 0::2 表示从第0列开始，每隔2列取一次（偶数列）
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度 (1, max_len, d_model)，方便后续相加
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer，不参与梯度更新，但随模型保存在 GPU
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状是 (batch, seq, d_model)
        # 把位置编码加到 x 上！注意：位置编码只加，不求梯度
        x = x + self.pe[:, :x.size(1), :]
        return x

# 🎯 测试体验
d_model = 64
seq_len = 10
batch_size = 2

# 模拟词嵌入向量 (只有语义，没有位置)
word_embeddings = torch.randn(batch_size, seq_len, d_model)
print(f"原始词向量第一个词的前5维: {word_embeddings[0, 0, :5]}")

# 注入位置编码
pos_encoder = PositionalEncoding(d_model)
embedded_with_pos = pos_encoder(word_embeddings)
print(f"加位置后第一个词的前5维: {embedded_with_pos[0, 0, :5]}")

# 观察变化：数值变了，说明位置信息被成功融合进去了！