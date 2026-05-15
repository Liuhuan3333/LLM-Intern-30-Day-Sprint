# 任务 2: 字典的进阶应用
# 模拟一个Agent的配置信息
agent_config = {
    "model_name": "Qwen-7B",
    "temperature": 0.7,
    "role": "Assistant"
}

# ================= 1. get() 方法 =================
# 为什么常用？因为直接用 agent_config["key"] 取不存在的 key，程序会直接报 KeyError 崩溃！
# 在大模型开发中，API返回的数据经常缺斤少两，用 get() 可以防止程序崩溃。

# 错误示范（不要轻易尝试，会报错）：
# result = agent_config["max_tokens"] 

# 正确姿势：找不着就返回 None，或者你指定的默认值
max_tokens = agent_config.get("max_tokens") # 找不到，返回 None
max_tokens_default = agent_config.get("max_tokens", 2048) # 找不到，默认给 2048
print(f"get()获取不存在的key: {max_tokens}, 带默认值: {max_tokens_default}")

# ================= 2. items() 方法 =================
# 为什么常用？写Agent日志时，我们经常需要把配置全部打印出来。
# 如果只用 for key in dict，还要再用 dict[key] 取值，太麻烦。

print("\n--- Agent 配置详情 (使用 items) ---")
for key, value in agent_config.items():
    # f-string 格式化输出，让日志更美观
    print(f"配置项: {key} = {value}")

# ================= 3. update() 方法 =================
# 为什么常用？比如我们要更新Agent的设定，或者把两个字典合并，不需要一个个赋值。

new_settings = {"temperature": 0.2, "top_p": 0.9} # 我们想调低温度，并增加top_p
print(f"\n更新前的温度: {agent_config['temperature']}")

# update() 会把 new_settings 里的内容覆盖/加进 agent_config
agent_config.update(new_settings) 

print(f"更新后的温度: {agent_config['temperature']}")
print(f"新增的top_p: {agent_config.get('top_p')}")


# 练习：大模型调用完成后，返回了一个结果字典。
# response = {"text": "你好", "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
# 要求：
# 1.使用 get() 方法，尝试从 response 中获取 "finish_reason" 这个 key。因为这次返回的数据里没有这个 key，请给它一个默认值 "stop"。
# 2.将获取到的值打印出来，格式如：本次请求的结束原因: stop
response = {"text": "你好", "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
finish_reason = response.get("finish_reason")  # 直接获取，找不到会返回 None
finish_reason_default = response.get("finish_reason", "stop")  # 获取，找不到就返回 "stop"
print(f"本次请求的结束原因: {finish_reason_default}")