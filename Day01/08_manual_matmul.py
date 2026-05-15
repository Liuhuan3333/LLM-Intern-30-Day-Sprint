# 手写矩阵乘法
def matrix_multiply(A, B):
    # 1. 获取维度
    m = len(A)      # A的行数
    n = len(A[0])   # A的列数 (也是B的行数)
    p = len(B[0])   # B的列数
    
    # 2. 验证维度是否匹配 (A的列数必须等于B的行数)
    if len(B) != n:
        raise ValueError("矩阵维度不匹配，无法相乘！")
    
    # 3. 初始化结果矩阵 C，全是 0
    # 列表推导初始化 m行 p列 的二维列表
    C = [[0 for _ in range(p)] for _ in range(m)]  # _ 是一个占位符，表示这个变量我们不关心它的名字，只需要它的数量
    
    # 4. 核心逻辑：三重循环
    for i in range(m):          # 遍历A的行
        for j in range(p):      # 遍历B的列
            total = 0
            # ===== 请在这里补全内层循环 =====
            # 提示：遍历 k from 0 to n，累加 A[i][k] * B[k][j]
            for k in range(n):
                total += A[i][k] * B[k][j]
            # ================================
            C[i][j] = total
            
    return C

# 测试用例
A = [[1, 2], 
     [3, 4]]
B = [[5, 6], 
     [7, 8]]

result = matrix_multiply(A, B)
print("手写矩阵乘法结果:")
for row in result:  # result是一个二维列表，这个2维列表的每个元素都是一行，我们逐行打印出来
    print(row)

# 验证：对比之前 NumPy 的 A @ B 结果是否一致
import numpy as np
np_result = np.array(A) @ np.array(B)
print("\nNumPy 矩阵乘法结果:\n", np_result)