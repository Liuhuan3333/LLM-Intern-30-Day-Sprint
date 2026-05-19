# Tokenizer 探秘实战
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer  # AutoTokenizer 是Hugging Face提供的万能分词加载器，可以根据模型名称自动选择合适的分词器

# 1. 加载 BERT 的分词器
print("=== BERT Tokenizer (WordPiece 算法) ===")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I have a new GPU!"
tokens = bert_tokenizer.tokenize(text)
ids = bert_tokenizer.encode(text)

print(f"原句: {text}")
print(f"切词: {tokens}")  # 注意看 gpu 被切成了什么
print(f"数字 ID: {ids}")
print(f"特殊标记: CLS在头 {ids[0]}, SEP在尾 {ids[-1]}")

# 解码回来
decoded = bert_tokenizer.decode(ids)
print(f"解码回文本: {decoded}")

print("\n" + "="*50 + "\n")

# 2. 加载 GPT-2 的分词器
print("=== GPT-2 Tokenizer (BPE 算法) ===")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

text_gpt = "Unbelievable performance!"
tokens_gpt = gpt2_tokenizer.tokenize(text_gpt)
ids_gpt = gpt2_tokenizer.encode(text_gpt)

print(f"原句: {text_gpt}")
print(f"切词: {tokens_gpt}") # 注意看 Unbelievable 被切成了什么
print(f"数字 ID: {ids_gpt}")