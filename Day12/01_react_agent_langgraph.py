# 用 LangGraph 构建一个“开锁 Agent”（模拟博弈反思过程）

import os
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ==========================================
# 1. 定义 Agent 的状态 - 极其关键！
# ==========================================
# 在博弈中，Agent 必须记住之前发生了什么。
# 这里的 State 就是 Agent 的“短期记忆”。
class AgentState(TypedDict):
    messages: Annotated[list, add_messages] # 聊天历史，add_messages 是 LangGraph 提供的工具，自动把消息追加到列表里，方便我们追踪对话历史
    attempts: int                           # 尝试次数
    current_guess: str                      # 当前猜测
    is_correct: bool                        # 是否猜对

# 模拟的密码锁环境（真实场景中可能是外部API或游戏引擎）
SECRET_CODE = "472"

def check_lock(guess: str) -> str:
    """检查密码是否正确，返回环境反馈"""
    if guess == SECRET_CODE:
        return "Lock opened! Congratulations!"
    
    # 给出一点点反馈，模拟博弈对手的暴露
    feedback = []
    for i in range(3):
        if guess[i] == SECRET_CODE[i]:
            feedback.append(f"Position {i+1} is correct.")
        else:
            feedback.append(f"Position {i+1} is wrong.")
    return "Lock is still locked. Feedback: " + " | ".join(feedback)

# ==========================================
# 2. 定义图中的节点 - Agent 的脑和手
# ==========================================

# 节点A：思考与决策 (这里我们模拟 LLM 的推理，实际工程中这里是 llm.invoke)
def agent_think(state: AgentState):
    print("\n--- [Agent 正在思考] ---")
    
    # 模拟 LLM 根据 Observation 调整策略
    last_message = state['messages'][-1].content if state['messages'] else "Start guessing."
    
    if state['attempts'] == 0:
        # 第一次，随便猜一个
        new_guess = "123"
        thought = "I have no info. Let's try 123 first."
    else:
        # 反思阶段：根据反馈调整
        thought = f"Last time I guessed {state['current_guess']}. Observation: {last_message}. I need to adjust."
        # 这里写死调整逻辑（真实情况由大模型生成）
        if "Position 1 is correct" in last_message and "472" not in state['current_guess']:
             new_guess = "472" # 假设大模型推理出了正确的调整
        else:
             new_guess = str(int(state['current_guess']) + 1) # 随便调一下

    print(f"Thought: {thought}")
    print(f"Action: Trying code {new_guess}")
    
    return {"current_guess": new_guess, "attempts": state['attempts'] + 1}

# 节点B：执行动作并观察环境
def agent_act(state: AgentState):
    print("\n--- [Agent 正在行动] ---")
    observation = check_lock(state['current_guess'])
    print(f"Observation: {observation}")
    
    is_correct = "opened" in observation
    
    # 将观察结果存入消息历史，供下次思考使用
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=observation)],
        "is_correct": is_correct
    }

# ==========================================
# 3. 构建状态图 - 把脑和手连起来
# ==========================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("think", agent_think)
workflow.add_node("act", agent_act)

# 设置入口点：Agent 一开始先思考
workflow.set_entry_point("think")

# 添加边：思考完 -> 行动
workflow.add_edge("think", "act")

# 🎯 核心高光：条件边 —— 行动完，是继续思考还是结束？
def should_continue(state: AgentState):
    if state['is_correct']:
        return "end" # 密码对了，结束循环
    if state['attempts'] >= 5:
        return "end" # 尝试太多次了，强制结束（防止死循环）
    return "continue" # 否则，回去继续思考

workflow.add_conditional_edges(
    "act",          # 从行动节点出发
    should_continue, # 判断函数
    {
        "continue": "think", # 如果继续，回到思考节点 (形成循环！)
        "end": END           # 如果结束，终止图
    }
)

# 编译图
app = workflow.compile()

# ==========================================
# 4. 运行 Agent
# ==========================================
print("===== 开始开锁博弈 =====")
initial_state = {"messages": [], "attempts": 0, "current_guess": "", "is_correct": False}

# 运行图，直到结束
for output in app.stream(initial_state):
    pass # 打印逻辑已在节点函数中处理

print("\n===== 博弈结束 =====")
if SECRET_CODE == "472":
    print("Agent 成功破解了密码！")