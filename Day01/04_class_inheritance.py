# 任务 4：类与继承 (class, init, super)
# 核心概念：为什么必须学？
# 写深度学习代码，所有的神经网络模型都要写成类，并且继承 PyTorch 提供的 nn.Module 这个父类。如果你不懂 __init__ 和 super()，深度学习的代码你会像看天书一样。

# ================= 1. 父类：基础AI模型 =================


class BaseModel:
    # __init__ 是初始化方法，创建对象时自动调用
    def __init__(self, name, version):
        # self.xxx 表示给这个对象绑定属性
        self.name = name       
        self.version = version
        print(f"[BaseModel] 基础模型初始化完成 -> 名称: {self.name}, 版本: {self.version}")

    def predict(self, text):
        return f"[{self.name}] 正在预测文本: {text}"


# ================= 2. 子类：大语言模型继承基础模型 =================
class LLM(BaseModel): # 括号里写父类的名字，表示 LLM 继承了 BaseModel
    def __init__(self, name, version, vocab_size):
        # 重点来了！！！
        # super().__init__() 的意思是：先去调用父类 BaseModel 的 __init__ 方法
        # 把 name 和 version 交给老爹去处理，不用自己再写 self.name = name 了
        # 如果不写这行，父类的初始化就不会执行，后续继承来的方法可能会报错！
        super().__init__(name, version) 
        
        # 子类自己特有的属性，老爹没有，自己初始化
        self.vocab_size = vocab_size 
        print(f"[LLM] 词汇表大小: {self.vocab_size}")

    # 子类可以有自己的新方法
    def chat(self, prompt):
        return f"[{self.name}] 回答你的问题: {prompt}"


# ================= 3. 测试运行 =================
print("--- 创建 LLM 实例 ---")
# 执行这行时，会自动调用 LLM 的 __init__ 方法
qwen = LLM(name="Qwen", version="2.0", vocab_size=151936)

# 观察终端的打印顺序：
# 1. 先打印了 BaseModel 的内容（因为 super().__init__ 执行了）
# 2. 再打印了 LLM 自己的内容

print("\n--- 调用方法 ---")
# LLM 没有定义 predict，但它是从 BaseModel 继承来的，所以可以直接用
print(qwen.predict("测试文本")) 

# chat 是 LLM 自己定义的，当然也能用
print(qwen.chat("什么是大模型？"))


# 练习：我们要基于大语言模型，创造一个能使用工具的 Agent。
# 要求：
# 1. 定义一个类 Agent，让它继承自上面的 LLM 类。
# 2. Agent 的 __init__ 方法需要接收四个参数：name, version, vocab_size, tools（工具列表，比如 ["搜索", "计算器"]）。
# 3. 在 Agent 的 __init__ 里，先用 super() 把前三个参数交给父类 LLM 处理。
# 4. 然后给 Agent 绑定自己特有的属性 self.tools = tools，并打印一句话：[Agent] 装载工具: {self.tools}。
# 5. 创建一个 Agent 对象并传入参数，测试它是否能调用 chat() 方法（继承来的），以及是否能打印出工具列表。
class Agent(LLM):
    def __init__(self, name, version, vocab_size, tools):
        super().__init__(name, version, vocab_size)
        self.tools = tools
        print(f"[Agent] 装载工具: {self.tools}")

# 测试创建 Agent 实例
print("\n--- 创建 Agent 实例 ---")
my_agent = Agent(name="MyAgent", version="1.0", vocab_size=151936, tools=["搜索", "计算器"])
# 测试 Agent 是否能调用 chat() 方法（继承来的）
print(my_agent.chat("请帮我搜索一下天气预报。"))
