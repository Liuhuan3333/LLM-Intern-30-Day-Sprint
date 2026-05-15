# 任务 3：生成器 (yield)
# 核心概念：为什么需要生成器？
# 训练大模型时，数据集动辄几个G甚至几十个G。如果你用传统的 return 把数据一次性读进一个列表，你的服务器内存可能瞬间就爆了（OOM，Out of Memory）。
# 生成器就像一根吸管，你喝一口（调用 next()），它给一口；你不喝，它就不动。它不会把所有数据提前装进内存，而是按需生成。

import time  # 用来模拟处理数据的时间

# ================= 传统函数：用 return =================
def get_data_with_return(texts):
    print("🚀 [Return版] 开始处理数据...")
    result = []
    for text in texts:
        print(f"  [Return版] 正在处理: {text}")
        result.append(text.upper()) # 把所有结果都装进列表
    print("🚀 [Return版] 处理完毕，一次性全部交出！")
    return result # 一次性交出所有数据

# ================= 生成器函数：用 yield =================
def get_data_with_yield(texts):
    print("\n⚡ [Yield版] 生成器启动！")
    for text in texts:
        print(f"  [Yield版] 正在处理: {text}")
        yield text.upper() # 遇到 yield，程序会暂停，把当前结果交出去，等你下次要的时候再继续
    print("⚡ [Yield版] 所有数据处理完毕！")


# 测试数据
my_texts = ["第一条文本", "第二条文本", "第三条文本"]

# ---------- 测试 Return 版 ----------
print("--- 调用 Return 版函数 ---")
return_result = get_data_with_return(my_texts) 
# 注意观察：上面这行代码执行时，函数内部的所有代码都跑完了，数据全在 return_result 里了
print("拿到 Return 结果:", return_result)


# ---------- 测试 Yield 版 ----------
print("\n--- 调用 Yield 版函数 ---")
yield_result = get_data_with_yield(my_texts)
# 惊喜发现：上面这行代码执行后，函数内部的一句 print 都没执行！
# 因为生成器函数调用时不会立刻执行，而是返回一个“生成器对象”
print("类型:", type(yield_result))

print("\n--- 第一次向 Yield 要数据 ---")
item1 = next(yield_result) # 只有调用 next()，函数才会真正开始跑，跑到 yield 处停下
print("我拿到了:", item1)

print("\n--- 第二次向 Yield 要数据 ---")
item2 = next(yield_result) # 再次调用 next()，从上次停下的地方继续跑，再跑到 yield 处停下
print("我拿到了:", item2)

print("\n--- 剩下的数据用 for 循环拿（最常用） ---")
for item in yield_result: # for 循环会自动不断调用 next()，直到结束
    print("循环拿到了:", item)



# 练习：模拟一个制造微调数据集的生成器。
# 要求：
# 定义一个生成器函数 generate_mock_data(num)，它接收一个整数 num。
# 在函数内部，用 for i in range(num): 循环。
# 在循环里，用 yield 返回一个格式化字符串，比如 f"这是第 {i+1} 条训练数据"。
# 在函数外部，调用这个生成器，生成 5 条数据。
# 使用 next() 取出第一条数据并打印。
# 使用 for 循环把剩下的数据全部取出并打印。
def generate_mock_data(num):
    for i in range(num):
        yield f"这是第 {i+1} 条训练数据"

# 调用生成器，生成 5 条数据
mock_data_generator = generate_mock_data(5)

# 使用 next() 取出第一条数据并打印
first_data = next(mock_data_generator)
print("\n第一条数据:", first_data)

# 使用 for 循环把剩下的数据全部取出并打印
print("\n剩下的数据:")
for data in mock_data_generator:  # 注意：这个 for 循环会从上次 next() 停下的地方继续取数据，直到生成器没有数据了，生成器内部有一个隐含的状态指针
    print(data)