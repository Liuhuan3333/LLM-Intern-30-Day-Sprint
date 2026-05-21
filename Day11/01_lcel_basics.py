# LangChain 核心魔法 —— LCEL (LangChain 表达式语言)

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 1. 初始化本地大模型 (复用我们的 Qwen)
model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

# 创建一个 HuggingFace 的 pipeline (文本生成任务)
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=50,
    temperature=0.1
)

# 将 HuggingFace pipeline 包装成 LangChain 能识别的 LLM 组件
llm = HuggingFacePipeline(pipeline=pipe)

# 2. 定义 Prompt 模板
prompt = ChatPromptTemplate.from_template("请用一句话解释什么是 {concept}")

# 3. 定义输出解析器 (把 LLM 输出的对象变成纯字符串)
output_parser = StrOutputParser()

# 🎯 核心高光：LCEL 链式调用
# 使用管道符 | 将组件串联：Prompt -> LLM -> OutputParser
chain = prompt | llm | output_parser

# 4. 运行链
print("运行 LCEL Chain...")
result = chain.invoke({"concept": "大模型微调"})
print(f"模型回答: {result}")