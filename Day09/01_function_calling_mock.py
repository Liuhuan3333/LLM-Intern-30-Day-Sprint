# Function Calling 模拟
# 这是大模型能够“使用工具”的秘密

import json

# 1. 定义工具库 (模拟 Python 后端的能力)
def get_weather(city):
    # 模拟调用外部天气 API
    weather_data = {
        "北京": {"temp": 25, "condition": "晴"},
        "上海": {"temp": 28, "condition": "多云"},
    }
    return weather_data.get(city, {"temp": "未知", "condition": "未知"})

# 2. 模拟大模型的判断过程
def llm_thinking(user_query, available_tools):
    print(f"用户提问: {user_query}")
    print(f"可用工具: {available_tools}")
    
    # 真实情况中，这一步是由大模型根据 Prompt 自动生成 JSON 的
    # 这里我们模拟大模型决定调用 get_weather 工具
    if "天气" in user_query:
        # 🎯 核心：大模型输出的不是自然语言，而是结构化的 JSON 指令！
        llm_output = {
            "thought": "用户想知道天气，我需要调用 get_weather 工具。",
            "action": "get_weather",
            "action_input": {"city": "北京"}
        }
        return llm_output
    else:
        return {"thought": "普通问题，直接回答", "action": "direct_answer", "action_input": None}

# 3. Agent 执行循环
available_tools = ["get_weather"]
user_query = "今天北京天气怎么样？"

# 步骤 1：大模型思考要做什么
llm_response = llm_thinking(user_query, available_tools)
print(f"\n[大模型输出 JSON]: \n{json.dumps(llm_response, ensure_ascii=False, indent=2)}")

# 步骤 2：Python 解析大模型的 JSON，并执行对应代码
if llm_response["action"] == "get_weather":
    tool_name = llm_response["action"]
    tool_args = llm_response["action_input"]
    
    print(f"\n[Python 执行器]: 正在调用工具 {tool_name}，参数: {tool_args}...")
    observation = get_weather(**tool_args) # 解包参数调用函数
    
    print(f"[Python 执行器]: 获取到工具返回结果 -> {observation}")
    
    # 步骤 3：把工具结果再喂给大模型，让它总结（这里我们直接打印模拟）
    print(f"\n[大模型最终回复]: 今天北京的天气是{observation['condition']}，气温{observation['temp']}度。")