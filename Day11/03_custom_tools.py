# LangChain Tools —— 定义 Agent 的手脚

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFacePipeline
from langchain_core.tools import tool
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 1. 定义自定义工具
# 🎯 注意 description 的写法：必须清晰说明工具何时使用！
@tool
def calculate_length(text: str) -> int:
    """计算输入字符串的长度。当需要知道一段文字有多少个字符或字时使用。"""
    return len(text)

# 看看 LangChain 把它变成了什么
print("工具名称:", calculate_length.name)
print("工具描述:", calculate_length.description)
print("工具参数结构:", calculate_length.args_schema.schema())

# 2. 直接调用工具测试
print(f"\n计算'星火科技'的长度: {calculate_length.invoke('星火科技')}")  # 在 LangChain 中，所有组件（LLM、Tool、Chain）统一用 `.invoke()` 调用。这是 LangChain 的标准接口
