# 实战！用 PEFT 库微调 Qwen 大模型
# HuggingFace 的 PEFT (Parameter-Efficient Fine-Tuning) 库是 LoRA 的官方神装。我们今天用通义千问最新的 Qwen2.5-0.5B 来做指令微调。

import torch

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载原模型和分词器 (Qwen2.5-0.5B)
model_id = "Qwen/Qwen2.5-0.5B"
print(f"正在下载/加载模型: {model_id} ... (首次运行需下载，请耐心等待)")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",          # 自动把模型放到 GPU 上
    torch_dtype=torch.float16   # 使用半精度，省显存，省了一半显存，精度损失极小
)

# 🎯 核心观察 1：打印原模型的可训练参数
def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / 总参数: {total:,} | 占比: {100*trainable/total:.4f}%")

print("\n--- 原 Qwen 模型参数 ---")
print_trainable_params(model)


# 2. 配置 LoRA
# 🎯 这是最核心的配置区域！面试常考！
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型：因果语言模型 (GPT类)
    r=8,                          # 秩！即我们刚才讲的那个小维度 8。越大表达能力越强，但显存越耗
    lora_alpha=16,                # 缩放系数，一般设为 r 的 2 倍
    lora_dropout=0.1,             # 防止过拟合的 Dropout
    target_modules=["q_proj", "k_proj", "v_proj"] # 告诉 LoRA 只在 Attention 的 QKV 矩阵上加小矩阵！
)

# 3. 将原模型包装成 LoRA 模型
peft_model = get_peft_model(model, peft_config)

# 🎯 核心观察 2：打印 LoRA 模型的可训练参数
print("\n--- 加入 LoRA 后的模型参数 ---")
print_trainable_params(peft_model)
# 你会震惊地发现，可训练参数占比从 100% 降到了不到 1%！

# 4. 模拟一条训练数据 (指令微调格式)
# 教模型一个新知识：我们公司名叫 "星火科技"
instruction = "你叫什么名字？你属于哪家公司？"
output = "我是星火大模型，我属于星火科技。"

# 构造 Prompt (Alpaca 格式)
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

# 编码
inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

# 5. 简单训练几步 (模拟微调过程)
print("\n开始模拟微调训练...")
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

peft_model.train() # 开启训练模式
for step in range(5):
    outputs = peft_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step+1} | Loss: {loss.item():.4f}")

# 6. 测试微调后的效果
peft_model.eval() # 开启评估模式
test_prompt = "### Instruction:\n你属于哪家公司？\n\n### Response:\n"
test_inputs = tokenizer(test_prompt, return_tensors="pt").to(peft_model.device)

with torch.no_grad():
    generated_ids = peft_model.generate(
        **test_inputs, 
        max_new_tokens=20,
        temperature=0.1,
        do_sample=True
    )

response = tokenizer.decode(generated_ids[0][test_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\n微调后模型回答: {response}")