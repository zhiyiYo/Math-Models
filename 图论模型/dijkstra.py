# coding:utf-8

from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from draw_graph import drawNetwork, drawPath

INF = 1 << 20


class Dijkstra:
    """ Dijkstra算法解最短路 """

    INF = 1 << 20

    def __init__(self, start_vertex: int, end_vertex: int, adjacency_mat):
        """ 创建Dijkstra对象

        Parameters
        ----------
        start_vertex : int\n
            起点标号，标号最小值为0

        end_vertex : int\n
            终点标号，标号最小值为0

        adjacency_mat : 二维array_like\n
            邻接矩阵
        """
        self.setModel(start_vertex, end_vertex, adjacency_mat)

    def setModel(self, start_vertex: int, end_vertex: int, adjacency_mat):
        """ 重置模型

        Parameters
        ----------
        start_vertex : int\n
            起点标号，标号最小值为0

        end_vertex : int\n
            终点标号，标号最小值为0

        adjacency_mat : 二维array_like\n
            邻接矩阵
        """
        # 初始化起点、终点和邻接矩阵
        self.is_find_path = False
        self.end_vertex = end_vertex
        self.start_vertex = start_vertex
        self.adjacency_mat = np.array(adjacency_mat)

        if self.adjacency_mat.ndim != 2:
            raise Exception('adjacency_mat必须为二维array_like数组')

        # 定点数和从当前顶点到其他顶点的路径数组
        self.vertex_num = len(self.adjacency_mat)
        self.distance_array = self.adjacency_mat[self.start_vertex].copy()

        # 未处理的顶点列表
        self.__passed_vertexes = [self.start_vertex]
        self.__nopassed_vertexes = [i for i in range(
            self.vertex_num) if i != self.start_vertex]

        # 初始化从起始顶点到其他顶点的路径矩阵以及最短路列表
        self.__path_mat = [[self.start_vertex, i]
                           for i in range(self.vertex_num)]
        self.__path_mat[0] = [self.start_vertex]
        self.shortest_path = self.__path_mat[self.end_vertex]

    def findPath(self, is_show_gragh: bool = True) -> list:
        """寻找并返回最短路

        Parameters
        ----------
        is_show_gragh : bool, optional\n
            是否显示最短路, 默认为True

        Returns
        -------
        shortest_path : list\n
            返回最短路的顶点列表
        """
        while self.__nopassed_vertexes:
            # 当前未经过的顶点的距离列表中的最短路及其对应的顶点
            min_path = np.min(self.distance_array[self.__nopassed_vertexes])
            vertexes = np.where(self.distance_array == min_path)[0]
            # 找出不在passed_vertexes列表中的第一个最小值对应的顶点
            vertex = [i for i in vertexes if i not in self.__passed_vertexes][0]
            self.__nopassed_vertexes.remove(vertex)
            self.__passed_vertexes.append(vertex)

            for i in range(self.vertex_num):
                new_path = self.adjacency_mat[vertex, i] + min_path
                # 如果经过当前顶点到下一个顶点的路程小于起点到下一个顶点的路程就更新路
                if new_path < self.distance_array[i]:
                    self.distance_array[i] = new_path
                    self.__path_mat[i] = self.__path_mat[vertex] + [i]

        # 打印路径信息并显示图窗
        self.__printPathInfo()
        if is_show_gragh:
            self.showGraph()
        return self.shortest_path

    def __printPathInfo(self):
        """ 打印最短路的信息 """
        self.is_find_path = (self.distance_array[self.end_vertex] != INF)
        if self.is_find_path:
            # 最短路列表
            self.shortest_path = self.__path_mat[self.end_vertex]  # type:list
            print(f'{"最短路为：":<7}', ' → '.join(
                [str(i) for i in self.shortest_path]))
            print("最短路程为：", self.distance_array[self.end_vertex])
        else:
            print(f'无法找到从顶点{self.start_vertex}到顶点{self.end_vertex}的路径')

    def showGraph(self):
        """ 显示网络图和最短路 """
        # 绘制网络图
        fig = plt.figure()  # type:plt.Figure
        ax = fig.add_subplot(121) if self.is_find_path else fig.add_subplot()
        self.net_gragh, pos = drawNetwork(self.adjacency_mat, ax=ax)

        # 绘制最短路
        if self.is_find_path:
            path_ax = fig.add_subplot(122)  # type:plt.Axes
            self.path_graph, _ = drawPath(
                self.adjacency_mat, self.shortest_path, path_ax, pos)


if __name__ == "__main__":
    _ = INF
    adjacency_mat = [[0, 1, 12, _, _,  _],
                     [_, 0, 9, 3, _,  _],
                     [_, _, 0, _, 5,  _],
                     [_, _, 4, 0, 13, 15],
                     [_, _, _, _, 0,  4],
                     [_, _, _, _, _,  0]]

    solution = Dijkstra(0, 5, adjacency_mat)
    solution.findPath()
    plt.show()
