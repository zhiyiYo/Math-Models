from itertools import product
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from draw_graph import drawNetwork, drawPath

INF = 1<<20


class Floyd:
    """ Floyd算法求最短路 """

    INF = 1<<20

    def __init__(self, start_vertex: int, end_vertex: int, adjacency_mat):
        """ 创建Dijkstra对象

        Parameters
        ----------
        start_vertex : 起点标号，标号最小值为0\n
        end_vertex : 终点标号，标号最小值为0\n
        adjacency_mat : 邻接矩阵，类型为二维array_like数组
        """
        self.setModel(start_vertex, end_vertex, adjacency_mat)

    def setModel(self, start_vertex: int, end_vertex: int, adjacency_mat):
        """ 重置模型

        Parameters
        ----------
        start_vertex : 起点标号，标号最小值为0\n
        end_vertex : 终点标号，标号最小值为0\n
        adjacency_mat : 邻接矩阵，类型为二维array_like数组
        """
        # 初始化起点、终点和邻接矩阵
        self.is_find_path = False
        self.end_vertex = end_vertex
        self.start_vertex = start_vertex
        self.adjacency_mat = np.array(adjacency_mat)
        if self.adjacency_mat.ndim != 2:
            raise Exception('adjacency_mat必须为二维array_like数组')

        # 顶点数
        self.vertex_num = len(self.adjacency_mat)
        # 从一个顶点到另外一个顶点的最短路长的矩阵
        self.distance_mat = self.adjacency_mat.copy()
        # 记录从一个顶点到另外一个顶点的中间节点的矩阵
        self.path_mat = np.array([[i for i in range(self.vertex_num)]
                                  for _ in range(self.vertex_num)], dtype=int)
        self.shortest_path = []  # type:List[int]

    def findPath(self):
        """ 寻找最短路 """
        ra = range(self.vertex_num)
        for l, u, v in product(ra, ra, ra):
            new_distance = self.distance_mat[u, l] + self.distance_mat[l, v]
            if self.distance_mat[u, v] > new_distance:
                # 更新中间节点和最短路程
                self.distance_mat[u, v] = new_distance
                self.path_mat[u, v] = self.path_mat[u, l]

        # 打印最短路信息并显示图
        self.__printPathInfo()
        self.showGraph()

    def __printPathInfo(self):
        """ 打印最短路信息 """
        self.is_find_path = (self.distance_mat[self.start_vertex,
                                               self.end_vertex] != INF)
        if self.is_find_path:
            path = self.getShortestPath(self.start_vertex, self.end_vertex)
            self.shortest_path = path
            print(
                '最短路程为：', self.distance_mat[self.start_vertex, self.end_vertex])
            print(f'{"最短路为：":<7}', ' → '.join([str(i) for i in path]))
        else:
            print(f'无法找到从顶点{self.start_vertex}到顶点{self.end_vertex}的路径')

    def showGraph(self):
        """ 显示网络图和最短路 """
        ax = plt.subplot(121) if self.is_find_path else plt.subplot(111)
        self.net_gragh, pos = drawNetwork(self.adjacency_mat, ax)

        if self.is_find_path:
            path_ax = plt.subplot(122)  # type:plt.Axes
            self.path_graph, _ = drawPath(
                self.adjacency_mat, self.shortest_path, path_ax, pos)

    def getShortestPath(self, start_vertex: int, end_vertex: int):
        """ 从已有的路径矩阵中查找最短路

        Parameters
        ----------
        start_vertex : 起点标号，标号最小值为0\n
        end_vertex : 终点标号，标号最小值为0
        """
        if np.array_equal(self.distance_mat, self.adjacency_mat):
            self.findPath()

        vertex = self.path_mat[self.start_vertex, self.end_vertex]
        shortest_path = [self.start_vertex, vertex]
        while vertex != self.end_vertex:
            vertex = self.path_mat[vertex, self.end_vertex]
            shortest_path.append(vertex)

        return shortest_path


if __name__ == "__main__":
    adjacency_mat = [[0, 1, 12, INF, INF, INF],
                     [INF, 0, 9, 3, INF, INF],
                     [INF, INF, 0, INF, 5, INF],
                     [INF, INF, 4, 0, 13, 15],
                     [INF, INF, INF, INF, 0, 4],
                     [INF, INF, INF, INF, INF, 0]]

    solution = Floyd(1, 5, adjacency_mat)
    solution.findPath()
    print(solution.getShortestPath(1, 5))
    plt.show()
