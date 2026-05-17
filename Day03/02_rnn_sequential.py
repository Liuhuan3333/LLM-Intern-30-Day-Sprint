# RNN 实战

import torch
import torch.nn as nn

# 1. 准备简单的序列数据
# 规律：第3个数字 = 第1个数字 + 第2个数字
# 输入：[1, 2] -> 输出：3
# 输入：[5, 4] -> 输出：9
X = torch.tensor([[1.0, 2.0], [5.0, 4.0], [3.0, 7.0], [2.0, 8.0]]).unsqueeze(-1)   # unsqueeze(-1) 在最后一个维度加个维度，调整形状为 -> 4个样本，每个样本序列长度为2，每个词用1维向量表示
# 调整形状为 -> 4个样本，每个样本序列长度为2，每个词用1维向量表示

Y = torch.tensor([[3.0], [9.0], [10.0], [10.0]]) # 预测下一个数字

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, Y = X.to(device), Y.to(device)

# 2. 定义 RNN 模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 🎯 填空部分 1：定义 RNN 层
        # nn.RNN(输入维度, 隐藏层维度, 层数, batch_first=True)
        # batch_first=True 极其重要！表示输入数据的第一个维度是 batch_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        
        # 全连接层，把 RNN 最后的隐藏状态映射成输出
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 🎯 填空部分 2：前向传播
        # RNN 会返回所有步的输出 out，以及最后一个步的隐藏状态 h_n
        # 我们只需要最后一个步的输出来做预测，所以取 out[:, -1, :]
        out, h_n = self.rnn(x) # out 形状是 (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :]) # 取序列最后一个输出送入全连接层
        return out

model = SimpleRNN(input_size=1, hidden_size=16, output_size=1).to(device)

# 3. 训练
criterion = nn.MSELoss() # 回归任务用均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 测试一下它学没学会
test_x = torch.tensor([[4.0, 5.0]]).unsqueeze(-1).to(device) # 4+5=9
pred = model(test_x)
print(f"\n测试输入 [4, 5]，模型预测下一个数字是: {pred.item():.2f} (期望是 9)")