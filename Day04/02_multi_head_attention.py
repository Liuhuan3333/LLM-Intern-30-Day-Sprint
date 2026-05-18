# 多头注意力
# 怎么实现多头呢？把原来 64 维的 QKV，切成 8 份（8个头），每份 8 维。各自算 Attention，最后拼回去。

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度，64 // 8 = 8
        
        # 定义 Q, K, V 的线性变换（一次性生成所有头的参数，不用循环）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # 最后的输出线性层
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. 生成 Q, K, V
        q = self.W_q(x)  # (batch, seq, 64)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 🎯 填空部分 1：拆分多头
        # 我们要把 (batch, seq, 64) 变成 (batch, num_heads, seq, d_k=8)
        # 提示：先 view 成 (batch, seq, 8, 8)，再用 transpose 把 seq 和 8(num_heads) 换位置
        # .contiguous() 是为了确保内存连续，后面 view 不会报错
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        
        # 2. 计算 Attention (和单头一模一样，只是维度变小了)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        out = attention_weights @ v  # (batch, num_heads, seq, d_k)
        
        # 🎯 填空部分 2：拼接多头
        # 把 (batch, num_heads, seq, d_k) 变回 (batch, seq, d_model)
        # 提示：先 transpose(1, 2) 把形状换回 (batch, seq, num_heads, d_k)
        # 然后 .contiguous().view(batch_size, -1, self.d_model) 拍平
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 3. 经过最后的线性层
        out = self.fc_out(out)
        return out

# 测试
batch_size, seq_len, d_model, num_heads = 2, 5, 64, 8
x = torch.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model, num_heads)
output = mha(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}") # 期望输出: torch.Size([2, 5, 64])