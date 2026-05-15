# 任务 1：列表推导
# 需求：把一个句子里的单词全部变成小写，并且只保留长度大于3的单词
sentence = "The Quick Brown Fox Jumps Over The Lazy Dog"

# 传统写法（不推荐）
result_old = []  # 先创建一个空列表
for word in sentence.split():  # 先把句子分割成单词，split默认以空格为分隔符，按空格拆分字符串，返回一个列表
    if len(word) > 3:
        result_old.append(word.lower())  #append()方法用于在列表末尾添加新的对象
print("传统写法:", result_old)

# 列表推导（推荐！一行搞定）
result_new = [word.lower() for word in sentence.split() if len(word) > 3]  # 列表推导式的语法： [表达式 for 变量 in 可迭代对象 if 条件]
print("列表推导:", result_new)


# 练习 1:把 ["apple", "banana", "cat", "dog", "elephant"] 中包含字母 "a" 的找出来，并变成大写。
fruits = "apple banana cat dog elephant"
fruits_a = [fruit.upper() for fruit in fruits.split() if 'a' in fruit]
print("包含字母 'a' 的单词:", fruits_a)