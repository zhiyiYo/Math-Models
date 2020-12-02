# coding:utf-8

from typing import Any, Iterable, Tuple

import pulp
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog



class LP:
    """ 线性规划问题 """

    @classmethod
    def linprog(cls, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                bounds: Iterable[Tuple[int, Any]] = None, is_maximize: bool = False):
        """ 求解线性规划问题

        Parameters
        ----------
        c : 1-D array_like
            目标函数的系数向量

        A_ub : 2-D array_like
            约束条件为 `<=` 的不等式方程组的系数矩阵

        b_ub : 1-D array_like
            与A_ub对应的不等式右边的常数项向量

        A_eq : 2-D array_like
            约束条件为 `==` 的等式方程组的系数矩阵

        b_eq : 1-D array_like
            与A_eq对应的等式右边的常数项向量

        bounds : Iterable[Tuple[int, Any]]
            决策变量取值范围组成的集合，取值范围的格式为: `(min, max)` 或者 `(min, None)`

        is_maximize : bool
            目标函数的类型是否为最大值问题
        """
        c = -np.array(c) if is_maximize else np.array(c)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
        # 如果是最大值问题需要将最优值取反
        if is_maximize:
            res['fun'] *= - 1
        return res

    @classmethod
    def linearAssignment(cls, cost_mat, is_maximize: bool = False):
        """ 求解指派问题

        Parameters
        ----------
        cost_mat : 2-D array_like
            消耗系数矩阵

        is_maximize : bool
            是否为最大值问题

        Returns
        -------
        row_index : 1-D `~np.ndarray`
            从 0 到 n-1 的一维向量, 代表第i个人

        col_index : 1-D `~np.ndarray`
            col_index[i] = j 代表第 i 个人进行的工作为 j

        total_cost : 总消耗

        """
        row_index, col_index = linear_sum_assignment(cost_mat, is_maximize)
        total_cost = cost_mat[row_index, col_index].sum()
        return (row_index, col_index, total_cost)

    @classmethod
    def linearTransform(cls, cost_mat, a, b, is_maximize: bool = False):
        """ 求解运输问题

        Parameters
        ----------
        c : 2-D array_like
            消耗系数矩阵

        a : 1-D array_like
            供应量组成的向量

        b : 1-D array_like
            需求量组成的向量

        is_maximize : bool
            是否为最大值问题

        Returns
        -------
        best_value : 最优值
        best_solution : 最优解矩阵
        """
        cost_mat = np.array(cost_mat)
        row, col = cost_mat.shape
        sense = pulp.LpMaximize if is_maximize else pulp.LpMinimize
        # 构造问题对象
        lp = pulp.LpProblem('运输问题', sense)
        # 创建决策变量
        x_mat = np.array([[pulp.LpVariable(f'x{i}{j}') for j in range(col)]
                          for i in range(row)])
        # 设置目标函数
        lp += (cost_mat*x_mat).sum()
        # 设置约束条件
        for i in range(row):
            lp += pulp.lpSum(x_mat[i]) <= a[i]

        for j in range(col):
            lp += pulp.lpSum(x_mat[:, j]) <= b[j]
        # 求解问题
        lp.solve(pulp.PULP_CBC_CMD(msg=False))
        # 最优值和最优解矩阵
        best_value = pulp.value(lp.objective)
        best_solution_mat = np.array([[pulp.value(x_mat[i, j]) for j in range(
            x_mat.shape[1])] for i in range(x_mat.shape[0])])

        return (best_value, best_solution_mat)


if __name__ == "__main__":

    # 线性规划问题
    c = [2, 3]
    A_ub = [[1, 2], [4, 0], [0, 4]]
    b_ub = [8, 16, 12]
    bounds = [(0, None), (0, None)]
    res = LP.linprog(c, A_ub, b_ub, bounds=bounds, is_maximize=True)

    # 指派问题
    efficiency_matrix = np.array([
        [12, 7, 9, 7, 9],
        [8, 9, 6, 6, 6],
        [7, 17, 12, 14, 12],
        [15, 14, 6, 6, 10],
        [4, 10, 7, 10, 6]
    ])
    row_ind, col_ind, cost = LP.linearAssignment(efficiency_matrix)

    # 运输问题
    costs = np.array([[500, 550, 630, 1000, 800, 700],
                      [800, 700, 600, 950, 900, 930],
                      [1000, 960, 840, 650, 600, 700],
                      [1200, 1040, 980, 860, 880, 780]])

    max_plant = [76, 88, 96, 40]
    max_cultivation = [42, 56, 44, 39, 60, 59]
    best_value, best_solution_mat = LP.linearTransform(
        costs, max_plant, max_cultivation,True)

    print(best_value)