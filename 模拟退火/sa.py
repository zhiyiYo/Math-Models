# coding:utf-8

from math import cos, exp, sin
from random import random, uniform
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class SA:
    """ 模拟退火 """

    FIND_MIN_VALUE = 0
    FIND_MAX_VALUE = 1

    def __init__(self, func, search_interval: Tuple[float, float], value_type: int = 0,
                 initial_T: float = 100, final_T: float = 1, alpha: float = 0.9):
        """创建模拟退火对象

        Parameters
        ----------
        func : 用于产生数值的可调用函数

        search_interval : Tuple[float, float]
            搜索区间，第一个元素为左边界，第二个元素为右边界

        value_type : int, optional
            寻找的最值类型, 有以下两种::

                - SA.FIND_MIN_VALUE : 寻找最小值
                - SA.FIND_MAX_VALUE : 寻找最大值

        initial_T : float
            初始温度

        final_T : float
            停止温度

        alpha : float
            降温系数
        """
        self.func = func
        self.alpha = alpha
        self.final_T = final_T
        self.initial_T = initial_T
        self.__setValueType(value_type)
        self.search_interval = search_interval
        # 寻找结果
        self.find_result = (0, 0)  # type:Tuple[float,float]
        self.best_x = 0  # type:float
        self.best_y = 0  # type:float

    def find(self, iter_times: int = 10, value_type: int = None, is_plot: bool = True):
        """ 寻找最值

        Parameters
        ----------
        iter_times : int
            迭代次数

        value_type : int
            最值类型，如果为None则为初始化对象时设定的类型

        is_plot : bool
            是否绘制目标函数曲线以及搜索结果
        """
        if value_type:
            self.__setValueType(value_type)
        # 记录历史最值
        results = []  # type:List[Tuple[float,float]]
        for i in range(iter_times):
            # 设置起始温度
            T = self.initial_T
            # 设置一个初始x，从该位置开始寻找最值
            x_min, x_max = self.search_interval[0], self.search_interval[1]
            x_best = x = uniform(x_min, x_max)
            y_best = y = self.func(x)
            # 在温度降至最低温之前一直迭代
            while T > self.final_T:
                # 开始迭代50次
                for i in range(50):
                    # 产生扰动
                    delta_x = random()-0.5
                    # 保证新的x仍在搜索区间中
                    if x_min < delta_x + x < x_max:
                        x_new = delta_x + x
                    else:
                        x_new = x - delta_x
                    # 比较新的y和旧的y
                    y_new = self.func(x_new)
                    if self.__compare(y_new, y) or exp(-(y - y_new) / T) > random():
                        x = x_new
                        y = y_new
                        # 更新最优解
                        if self.__compare(y_best, y_new):
                            y_best = y_new
                            x_best = x_new
                # 更新下一次的搜索起点
                x = x_best
                y = y_best
                # 更新温度
                T *= self.alpha

            # 记录每一个最值
            results.append((x_best, y_best))
        # 打印最值
        self.find_result = sorted(
            results, key=lambda i: i[1], reverse=self.value_type)[0]
        self.best_x, self.best_y = self.find_result
        print(f'最{"大" if self.value_type else "小"}值为{self.find_result}')
        # 绘制曲线
        if is_plot:
            self.plot()

    def __compare(self, y_old, y_new):
        """ 根据指定的最值类型对输入值进行比较 """
        return y_new > y_old if self.value_type else y_new < y_old

    def __setValueType(self, value_type: int):
        """ 设置最值类型 """
        if not 0 <= value_type <= 1:
            raise Exception('最值类型错误')
        self.value_type = value_type

    def plot(self):
        """ 绘制对象函数曲线以及寻找到的最值 """
        x = np.linspace(self.search_interval[0], self.search_interval[1], 1000)
        y = [self.func(i) for i in x]
        plt.plot(x, y)
        plt.plot(self.best_x, self.best_y, 'r*')
        plt.xlim(self.search_interval[0], self.search_interval[1])


if __name__ == "__main__":
    def func(x):
        return (x**2-5*x)*sin(x**2)

    plt.style.use('matlab')
    # 寻找最大值
    solver = SA(func, (-4, 5), SA.FIND_MAX_VALUE)
    solver.find(20)
    plt.show()
