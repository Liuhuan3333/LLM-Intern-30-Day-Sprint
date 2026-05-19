# GPT-2 文本生成实战
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. 加载分词器和模型
# GPT-2 是 OpenAI 开源的小型生成模型，非常适合做实验
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 把模型放到 GPU 上
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval() # 设置为评估模式，关闭 Dropout，推理（生成文本）时必须使用eval()模式，否则每次生成的结果都不稳定

# 2. 准备输入文本
prompt = "The future of Artificial Intelligence is"
# 将文本转化为数字 ID (Tokenization)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

print(f"输入文本: {prompt}")
print(f"转换成的数字 ID: {input_ids}")

# 3. 生成文本
# 🎯 核心参数体验区
with torch.no_grad():
    outputs = model.generate(
        input_ids, 
        max_length=50,       # 最多生成多少个词
        # 🎯 体验 1：Temperature (温度)
        # 温度越低，模型越保守，总是选概率最高的词（容易重复废话）
        # 温度越高，模型越奔放，喜欢选概率低但更有创意的词（容易胡说八道）
        temperature=0.1,      # 试试改成 0.1 或 1.5 看看效果
        
        # 🎯 体验 2：Top-p (核采样)
        # 只在累积概率达到 p 的那些词里选，过滤掉太离谱的词
        top_p=2.0,            # 试试改成 0.1 或 1.0
        
        do_sample=True,       # 必须为 True，temperature 和 top_p 才生效
        pad_token_id=tokenizer.eos_token_id
    )

# 4. 解码输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n生成的文本:\n{generated_text}")