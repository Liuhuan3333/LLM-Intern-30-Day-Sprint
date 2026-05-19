# BERT 情感分类实战
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import pipeline

# 1. 创建一个文本分类的 pipeline
# HuggingFace 会自动帮你下载一个微调好的 BERT 模型
# 第一次运行会下载模型，需要等一会儿
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0) # device=0 表示使用 GPU

# 2. 准备测试文本
texts = [
    "I love studying Large Language Models, it's fantastic!",
    "This movie was a complete waste of time and money.",
    "The weather is okay, nothing special."
]

# 3. 进行预测
results = classifier(texts)

# 4. 打印结果
for text, result in zip(texts, results):
    label = result['label']   # POSITIVE (积极) 或 NEGATIVE (消极)
    score = result['score']   # 置信度 (0 到 1 之间)
    print(f"文本: {text}")
    print(f"情感: {label} (置信度: {score:.4f})\n")