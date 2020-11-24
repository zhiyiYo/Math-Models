# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from random import sample, random


class SA_TSP:
    """ 模拟退火法解决旅行商问题 """

    def __init__(self, coordinate_list: list, initial_T: float = 100, final_T: float = 1, alpha: float = 0.9):
        """创建模拟退火法旅行商问题对象

        Parameters
        ----------
        coordinate_list : list
            每个元素都是一个坐标元组的列表

        initial_T : float, optional
            初始温度

        final_T : float, optional
            停止温度

        alpha : float, optional
            降温系数
        """
        self.coordinate_list = coordinate_list
        self.node_num = len(coordinate_list)
        self.best_path = [0]
        self.best_distance = 0
        self.initial_T = initial_T
        self.final_T = final_T
        self.alpha = alpha

    def findPath(self, iter_times: int = 10, is_plot: bool = True):
        """ 寻找最佳路径

        Parameters
        ----------
        iter_times : int
            寻找最优路径的迭代次数

        is_plot : bool
            是否绘制最优路径
        """
        self.__getDistanceMat()
        path_list = []
        distance_list = []
        for i in range(iter_times):
            # 初始化温度和路径
            T = self.initial_T
            path = path_best = list(range(self.node_num)) + [0]
            dis = dis_best = self.__getPathDistance(path)
            while T > self.final_T:
                # 迭代100次
                for _ in range(100):
                    # 随机交换两个节点产生新路径
                    i, j = sample(range(1, self.node_num), 2)
                    path_new = path.copy()
                    path_new[i] = path[j]
                    path_new[j] = path[i]
                    dis_new = self.__getPathDistance(path_new)
                    # 更新当前最优路径
                    if dis_new < dis or np.exp(-(dis - dis_new) / T) > random():
                        path = path_new
                        dis = dis_new
                        if dis_best > dis_new:
                            path_best = path_new
                            dis_best = dis_new
                # 更新最优路径
                path = path_best
                dis = dis_best
                # 更新温度
                T *= self.alpha
            # 将每次迭代的最短路径添加到列表中
            path_list.append(path_best)
            distance_list.append(dis_best)

        # 选择最佳的路径
        self.best_distance = sorted(distance_list)[0]
        self.best_path = path_list[distance_list.index(self.best_distance)]
        print('最佳路径为：', ' → '.join(str(i) for i in self.best_path))
        print('最短路程为：', self.best_distance)
        # 绘制路径
        if is_plot:
            self.plot()

    def __getDistanceMat(self):
        """ 根据坐标列表计算每两个坐标间的距离 """
        self.distance_mat = np.zeros(
            (self.node_num, self.node_num))  # type:np.ndarray
        for i in range(self.node_num):
            for j in range(self.node_num):
                # 计算距离
                dis = np.sqrt((self.coordinate_list[i][0]-self.coordinate_list[j][0])**2+(
                    self.coordinate_list[i][1]-self.coordinate_list[j][1])**2)
                self.distance_mat[i, j] = self.distance_mat[j, i] = dis

    def __getPathDistance(self, path: list):
        """ 计算路径的长度 """
        dis = sum(self.distance_mat[node, path[i+1]]
                  for i, node in enumerate(path[:-1]))
        return dis

    def plot(self):
        """ 绘制最佳路径 """
        x = [self.coordinate_list[i][0] for i in self.best_path]
        y = [self.coordinate_list[i][1] for i in self.best_path]
        plt.plot(x, y)
        plt.scatter(x, y, marker='o', color='r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Shortest distance : {self.best_distance:.1f}')
        for i in range(len(x)-1):
            plt.text(x[i]-0.3, y[i]-3, str(self.best_path[i]))


if __name__ == "__main__":
    coordinate_list = [(66.83, 25.36), (61.95, 26.34), (40, 44.39),
                       (24.39, 14.63), (17.07, 22.93), (22.93, 76.1),
                       (51.71, 94.14), (87.32, 65.36), (68.78, 52.19),
                       (84.88, 36.09), (50, 30), (40, 20), (25, 26)]
    plt.style.use('matlab')
    solver = SA_TSP(coordinate_list)
    solver.findPath(20)
    plt.show()
