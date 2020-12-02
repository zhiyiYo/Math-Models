import pulp
import numpy as np


# 创建线性规划问题对象
lp = pulp.LpProblem('线性规划', pulp.LpMaximize)

# 创建决策变量
x1 = pulp.LpVariable('x1', lowBound=0)
x2 = pulp.LpVariable('x2', lowBound=0)

# 设置目标函数
lp += 2 * x1 + 3 * x2

# 设置约束条件
lp += x1 + 2 * x2 <= 8
lp += 4 * x1 <= 16
lp += 4 * x2 <= 12

# 求最优解
lp.solve(pulp.PULP_CBC_CMD(msg=False))

# 打印最优解
print('最优值', pulp.value(lp.objective))
print('x1最优解', pulp.value(x1))
print('x2最优解', pulp.value(x2))

