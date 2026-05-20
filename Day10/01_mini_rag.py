# 实战！从 0 搭建本地 PDF 问答系统

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model_cache"

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ================= 1. 文档解析与分块 (极其关键！) =================
print("1. 正在解析文档并分块...")

# 模拟读取一个 PDF (如果没有PDF，我们直接用内存中的文本来演示)
# reader = PdfReader("your_paper.pdf")
# text = "".join([page.extract_text() for page in reader.pages])

# 为了演示方便，我们造一段长文本（模拟论文摘要）
text = """
Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address the hallucination and outdated knowledge issues of Large Language Models. 
However, naive RAG suffers from irrelevant retrieval results, which severely degrades the generation quality. 
To solve this, advanced RAG techniques like HyDE (Hypothetical Document Embeddings) and Re-ranking have been proposed.
HyDE generates a hypothetical answer to the query, and uses this hypothetical answer to retrieve similar documents, which significantly improves recall for short queries.
Re-ranking uses a Cross-Encoder to re-evaluate the retrieved documents, ensuring the top-K results are highly relevant to the query.
Evaluating RAG systems requires metrics like Faithfulness and Answer Relevance, which are implemented in frameworks like RAGAS.
"""

# 🎯【研究生必懂】分块策略：RecursiveCharacterTextSplitter
# 它不是死板地按字数切，而是按 [\n\n", "\n", " ", ""] 的优先级尝试切分，尽量保持段落的完整性！
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # 每个块的最大长度
    chunk_overlap=20,     # 🎯 重叠长度！极其重要！避免把一句话切成两半丢失语义
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.create_documents([text])

print(f"文档被切分成了 {len(chunks)} 个块。")
for i, chunk in enumerate(chunks):
    print(f"  块 {i+1}: {chunk.page_content[:50]}...")


# ================= 2. 向量化与存储 =================
print("\n2. 正在向量化并存入 Chroma 数据库...")

# 🎯【研究生必懂】选择 Embedding 模型
# all-MiniLM-L6-v2 是最经典的轻量级英文模型，速度快，效果不错
# 如果做中文，务必换成如 "shibing624/text2vec-base-chinese" 或 BGE 系列
embeddings = HuggingFaceEmbeddings(
    model_name="./models/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}
)

# 将文本块向量化并存入 Chroma (持久化到本地磁盘)
vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")


# ================= 3. 检索 =================
print("\n3. 正在检索与问题相关的上下文...")
query = "What is HyDE?"

# 🎯【研究生必懂】相似度计算方法
# 默认在 Chroma 中使用的是 L2 距离 (越小白越好)
# 也可以指定使用余弦相似度 (Cosine Similarity，越接近1越好)
retriever = vector_db.as_retriever(search_kwargs={"k": 2}) # k=2 表示返回最相关的 2 个块
retrieved_docs = retriever.invoke(query)

print(f"检索到 {len(retrieved_docs)} 个相关块：")
context = ""
for doc in retrieved_docs:
    print(f"  - {doc.page_content}")
    context += doc.page_content + "\n"


# ================= 4. 生成 (RAG 的最后一步) =================
print("\n4. 基于 Context 生成回答...")

# 加载本地 Qwen 模型做生成
model_id = "Qwen/Qwen2.5-0.5B" # 为了速度用0.5B，实际业务至少1.5B/7B
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

# 🎯【核心】构建 RAG Prompt 模板
# 严禁模型使用外部知识，必须基于 Context 回答！
rag_prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context. 
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}

Answer: """

# 生成
inputs = tokenizer(rag_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(f"问题: {query}")
print(f"RAG 回答: {answer}")