# 三大任务实战
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

# ==================== 任务1: 命名实体识别 (NER) ====================
print("="*20 + " 任务1: 命名实体识别 (NER) " + "="*20)
# NER: 从文本里揪出人名、地名、机构名
ner = pipeline("ner", aggregation_strategy="simple", device=0)
text_ner = "Elon Musk is the CEO of Tesla, which is based in Austin."
ner_results = ner(text_ner)

print(f"文本: {text_ner}")
for entity in ner_results:
    print(f"  实体: {entity['word']:10s} | 类型: {entity['entity_group']:8s} | 置信度: {entity['score']:.4f}")


# ==================== 任务2: 问答系统 (QA) ====================
# 新版 transformers 移除了 "question-answering" pipeline，改用手动加载模型
print("\n" + "="*20 + " 任务2: 问答系统 (QA) " + "="*20)

qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad").to("cuda")
qa_model.eval()

context = "The LLM internship requires knowledge of PyTorch and Transformer architecture. The salary is 500 per day."
question = "What skills are required for the internship?"

# 分词并转成 Tensor
inputs = qa_tokenizer(question, context, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = qa_model(**inputs)

# 从模型输出中找到答案的起止位置
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits) + 1
answer_ids = inputs["input_ids"][0][start_idx:end_idx]
answer = qa_tokenizer.decode(answer_ids)

print(f"背景: {context}")
print(f"问题: {question}")
print(f"答案: {answer}")


# ==================== 任务3: 文本摘要 ====================
# 新版 transformers 移除了 "summarization" pipeline，改用手动加载 BART 摘要模型
print("\n" + "="*20 + " 任务3: 文本摘要 " + "="*20)

sum_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to("cuda")
sum_model.eval()

long_text = """
Large language models (LLMs) have revolutionized the field of natural language processing. 
They are trained on vast amounts of text data and can generate human-like text, answer questions, 
and even write code. However, they also have limitations, such as hallucinating facts and 
requiring massive computational resources. Despite these challenges, LLMs are being integrated 
into various applications, from chatbots to advanced search engines.
"""

inputs = sum_tokenizer(long_text, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
with torch.no_grad():
    summary_ids = sum_model.generate(inputs["input_ids"], max_length=30, min_length=10, num_beams=4)
summary_text = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"原文: {long_text.strip()}")
print(f"摘要: {summary_text}")