# 任务 5：装饰器 (Decorators)

import time

# ================= 1. 定义装饰器 =================
# 装饰器本身是一个函数，它接收一个函数作为参数
def timer(func):  # func 就是被装饰的函数，比如 train_model 或 inference
    # wrapper (包装器) 是装饰器内部定义的新函数，用来替代原有的函数
    # *args, **kwargs 是魔法符号，表示接收任意数量的位置参数和关键字参数
    # 这样不管原函数需要什么参数，wrapper都能原封不动地接住
    def wrapper(*args, **kwargs):   # 为什么要用*args和**kwargs？因为我们希望这个装饰器能装饰任何函数，不管它需要什么参数，都能适用
        print(f"⏱️ [计时开始] 准备执行: {func.__name__}")  # func.__name__ 是被装饰函数的名字，比如 "train_model"，双下划线__name__ 是 Python 内置属性，表示函数的名字
        start_time = time.time()  #time.time() 返回当前时间的时间戳，单位是秒，记录开始时间
        
        # ========== 核心行 ==========
        # 真正执行原来的函数！如果不写这行，原函数就不会跑了
        # 把原函数的执行结果存到 result 里
        result = func(*args, **kwargs) 
        # ==========================
        
        end_time = time.time()
        cost = end_time - start_time
        print(f"⏱️ [计时结束] 函数 {func.__name__} 耗时: {cost:.4f}秒")  # 计算耗时，并格式化输出，保留4位小数
        
        # 必须把原函数的结果返回去，不然外面调用的人拿不到结果了
        return result 
    
    # 装饰器最后要把包装好的新函数返回出去
    return wrapper 

# ================= 2. 使用装饰器 =================
# @timer 是语法糖，等同于在定义完函数后写: train_model = timer(train_model)
@timer
def train_model(epochs):
    print(f"🚀 开始训练模型，共 {epochs} 个 epoch...")
    time.sleep(2) # 模拟训练耗时2秒
    return "模型训练完成"

@timer
def inference(text):
    print(f"🤖 正在推理文本: {text}")
    time.sleep(1) # 模拟推理耗时1秒
    return "推理结果: 你好呀"

# ================= 3. 测试运行 =================
print("--- 调用 train_model ---")
res1 = train_model(3) # 表面上调用了 train_model，实际上调用了 wrapper
print("拿到返回值:", res1)

print("\n--- 调用 inference ---")
res2 = inference("今天天气怎么样？")
print("拿到返回值:", res2)


# 练习：我们需要给 Agent 的行动加上日志追踪。
# 要求：
# 1. 定义一个名为 log_agent_action 的装饰器。
# 2. 在内部的 wrapper 函数中：
#       在执行原函数前，打印：📝 [日志] Agent 开始行动...
#       执行原函数，并保存结果。
#       在执行原函数后，打印：📝 [日志] Agent 行动结束。
#       返回原函数的结果。
# 3. 定义一个普通函数 search_web(query)，打印 正在搜索: {query}，并返回 搜索结果: 天气晴朗。
# 4. 用 @log_agent_action 装饰 search_web 函数。
# 5. 调用 search_web("北京天气")，并打印它的返回值。
def log_agent_action(func):
    def wrapper(*args, **kwargs):
        print("📝 [日志] Agent 开始行动...")
        result = func(*args, **kwargs)
        print("📝 [日志] Agent 行动结束.")
        return result
    return wrapper

@log_agent_action
def search_web(query):
    print(f"🔍 正在搜索: {query}")
    return f"搜索结果: {query} 天气晴朗"

search_web_result = search_web("北京天气")
print("拿到返回值:", search_web_result)