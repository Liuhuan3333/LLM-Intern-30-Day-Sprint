# 带记忆的对话系统
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 1. 初始化 LLM (复用我们的 Qwen 模型)
model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
llm = HuggingFacePipeline(pipeline=pipe)

# 2. 定义带有历史记录占位符的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的AI助手。"),
    MessagesPlaceholder(variable_name="chat_history"), # 🎯 核心：把历史记录插入这里！
    ("human", "{input}")
])

# 3. 模拟记忆存储 (真实场景会存入 Redis 或数据库)
chat_history = []

# 🎯 构建 Chain (注意这里使用了 LCEL 的高级语法，自动处理历史记录注入)
chain = prompt | llm | StrOutputParser()

# 4. 多轮对话模拟
def chat(user_input):
    global chat_history
    print(f"👤 用户: {user_input}")
    
    # 调用 Chain，传入当前输入和历史记录
    response = chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    print(f"🤖 助手: {response}\n")
    
    # 🎯 核心逻辑：将这一轮的问答存入历史记录，供下一轮使用！
    chat_history.append(HumanMessage(content=user_input))
    # 截取干净的回答 (因为小模型可能复读)
    clean_response = response.split("\n")[-1].strip() 
    chat_history.append(AIMessage(content=clean_response))

# 测试多轮对话
chat("你好，我叫星火，我是研究生。")
chat("我刚才说我叫什么名字？我是什么身份？") # 🎯 测试模型是否记住了上一轮的上下文