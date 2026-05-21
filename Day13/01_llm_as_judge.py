# 实战 LLM-as-Judge（核心必会）
# 使用本地 Ollama 的 qwen2.5:1.5b 模型作为裁判

import ollama
import json

# 1. 指定裁判模型（使用服务器上已有的 Ollama 模型）
JUDGE_MODEL = "qwen2.5:1.5b"  # 你服务器上已有的模型，无需下载

# 2. 定义评估数据集
# 包含问题、标准答案、待评估的模型输出
eval_data = [
    {
        "question": "法国的首都是哪里？",
        "golden_answer": "巴黎",
        "model_output": "法国的首都是巴黎。"
    },
    {
        "question": "法国的首都是哪里？",
        "golden_answer": "巴黎",
        "model_output": "法国最大的城市是里昂。"  # 故意给一个错误的回答
    }
]

# 🎯 核心高光：设计裁判 Prompt
# 必须包含：1. 角色设定 2. 评估维度 3. 评分标准 4. 输出格式要求
JUDGE_PROMPT = """
你是一个严格且公正的AI助手评估专家。你的任务是评估AI助手对用户问题的回答质量。

【评估维度】：相关性与正确性
【评分标准】：
- 1分：回答完全错误或与问题无关
- 2分：回答部分正确，但包含明显错误或遗漏
- 3分：回答完全正确且切中要害

【用户问题】：{question}
【标准答案】：{golden_answer}
【待评估回答】：{model_output}

请严格按照以下JSON格式输出你的评估结果，不要输出其他任何内容：
{{"score": <1或2或3>, "reason": "<简短的评分理由>"}}
"""

def generate_with_ollama(prompt, max_tokens=100):
    """用本地 Ollama 调用模型生成回答"""
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": max_tokens}
    )
    return response['message']['content']

def evaluate(question, golden, output):
    # 将数据填入裁判 Prompt 模板
    prompt = JUDGE_PROMPT.format(question=question, golden_answer=golden, model_output=output)
    
    # 调用本地 Ollama 模型生成评估结果
    result_text = generate_with_ollama(prompt)
    
    # 尝试解析 JSON（小模型可能输出不规范，这里做简单容错）
    try:
        # 找到第一个 { 和最后一个 } 之间的内容
        start = result_text.find("{")
        end = result_text.rfind("}") + 1
        if start != -1 and end != 0:
            json_str = result_text[start:end]
            return json.loads(json_str)
        else:
            return {"score": 0, "reason": "模型输出格式错误"}
    except Exception as e:
        return {"score": 0, "reason": f"解析失败: {result_text}"}

# 3. 运行评估
print(f"===== 开始 LLM-as-Judge 评估 (裁判模型: {JUDGE_MODEL}) =====")
for i, data in enumerate(eval_data):
    print(f"\n--- 评估样本 {i+1} ---")
    print(f"问题: {data['question']}")
    print(f"待评输出: {data['model_output']}")
    
    result = evaluate(data["question"], data["golden_answer"], data["model_output"])
    
    print(f"裁判打分: {result.get('score', 'N/A')} 分")
    print(f"裁判理由: {result.get('reason', 'N/A')}")

print("\n===== 评估完毕 =====")