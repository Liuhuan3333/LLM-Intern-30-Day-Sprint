# Pandas基础（大模型数据清洗利器）

import pandas as pd
import numpy as np

# ================= 1. 创建与读取 =================
print("=== 1. 创建 DataFrame ===")
# 模拟大模型评测数据
# data 是一个字典，键是列名，值是列数据（列表形式）
data = {
    "model": ["Qwen-7B", "Qwen-7B", "Llama3-8B", "Llama3-8B", "Qwen-7B"],
    "task": ["推理", "代码", "推理", "代码", "数学"],
    "score": [85.5, 90.0, 88.0, 92.5, 78.0]
}
df = pd.DataFrame(data)  # 把字典转换成 DataFrame 表格（二维表格）
print(df)

# 写入 CSV 文件（持久化保存）
df.to_csv("eval_results.csv", index=False) # index=False 表示不保存行号
print("\n数据已保存到 eval_results.csv")

# 读取 CSV 文件
df_from_csv = pd.read_csv("eval_results.csv")
print("\n从 CSV 读回的数据:\n", df_from_csv)


# ================= 2. 增删改查 =================
print("\n=== 2. 增删改查 ===")
# 查：选择一列（得到的是 Series 对象）
print("\n所有模型名称:\n", df['model'])

# 查：条件筛选（布尔索引）- 极常用！
# 找出分数大于 85 的记录
high_score = df[df['score'] > 85]
print("\n分数大于85的记录:\n", high_score)

# 改：修改特定位置的值
# 把 Llama3-8B 推理任务的分数改成 89.0
df.loc[2, 'score'] = 89.0 
print("\n修改后的表:\n", df)

# 增：增加一列
# 给每个任务加上权重
df['weight'] = [0.3, 0.4, 0.3, 0.4, 0.3]
print("\n增加权重列后:\n", df)

# 删：删除一列
df = df.drop(columns=['weight']) # 删完记得重新赋值给 df
print("\n删除权重列后:\n", df)


# ================= 3. Groupby 分组聚合 =================
print("\n=== 3. Groupby 分组聚合 ===")
# 大模型评测最常用的操作：算每个模型的平均分
# 按照 model 列分组，对 score 列求均值
mean_scores = df.groupby('model')['score'].mean()
print("各模型平均分:\n", mean_scores)

# 也可以一次算多个统计量
stats = df.groupby('model')['score'].agg(['mean', 'max', 'min'])
print("\n各模型详细统计:\n", stats)


# ================= 4. Merge 表拼接 =================
print("\n=== 4. Merge 表拼接 ===")
# 假设我们还有一张表，记录了模型的参数量和开源机构
model_info = pd.DataFrame({
    "model": ["Qwen-7B", "Llama3-8B"],
    "params(B)": [7, 8],
    "org": ["Alibaba", "Meta"]
})
print("模型信息表:\n", model_info)

# 将评测表和模型信息表根据 model 列拼起来（类似 SQL 的 JOIN）
merged_df = pd.merge(df, model_info, on='model')
print("\n合并后的大表:\n", merged_df)


# 练习：对大模型生成的结果进行质量分析。
# 要求：
# 1. 创建一个 DataFrame eval_data，包含三列：question_id (1, 2, 3, 4, 5), category ("数学", "推理", "数学", "代码", "推理"), passed (True, False, True, True, False)。
# 2. 使用 groupby，按 category 分组，计算每个类别的通过率。(提示：在 Python 中，True 等于 1，False 等于 0。所以通过率就是对 passed 列求均值 .mean())
# 3. 打印出通过率最高的类别。
eval_data = pd.DataFrame({
    "question_id": [1, 2, 3, 4, 5],
    "category": ["数学", "推理", "数学", "代码", "推理"],
    "passed": [True, False, True, True, False]
})

pass_rate = eval_data.groupby('category')['passed'].mean()
print("\n各类别通过率:\n", pass_rate)

# 找出通过率最高的类别
best_category = pass_rate.idxmax()  # idxmax() 返回通过率最高的类别的名字，返回最大值对应的索引标签
print(f"\n通过率最高的类别是: {best_category}")