# coding:utf-8

import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['simsun']
plt.rcParams['font.size'] = 15
INF = float('inf')


class Dijkstra:
    """ Dijkstra算法解最短路 """

    INF = float('inf')

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
        self.path = self.__path_mat[self.end_vertex]

    def findPath(self):
        """ 寻找最短路 """
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
        self.showGraph()

    def __printPathInfo(self):
        """ 打印最短路的信息 """
        self.is_find_path = (self.distance_array[self.end_vertex] != INF)
        if self.is_find_path:
            # 最短路列表
            self.path = self.__path_mat[self.end_vertex]  # type:list
            print(f'{"最短路为：":<7}', ' → '.join(str(i)for i in self.path))
            print("最短路程为：", self.distance_array[self.end_vertex])
        else:
            print(f'无法找到从顶点{self.start_vertex}到顶点{self.end_vertex}的路径')

    def showGraph(self):
        """ 显示网络图和最短路 """
        self.net_gragh = nx.DiGraph(name='网络图')
        self.path_graph = nx.DiGraph(name='最短路')

        # 创建节点和边
        ra = range(self.vertex_num)
        for i, j in product(ra, ra):
            if self.adjacency_mat[i, j] != INF and i != j:
                self.net_gragh.add_edge(i, j, weight=self.adjacency_mat[i, j])

        # 每条边的权重标签
        labels = {(u, v): weight for u, v,
                  weight in self.net_gragh.edges.data('weight')}

        # 绘图参数
        pos = nx.shell_layout(self.net_gragh)  # 布局
        options = {
            "font_family": "consolas",
            "edge_color": "#A0D0F0",
            "font_color": "white",
            "arrowstyle": '->',
            "node_size": 500,
            "arrowsize": 14,
            "font_size": 16,
            "width": 1.5,       # edge的宽度
        }

        # 绘制网络图
        net_ax = plt.subplot(121) if self.is_find_path else plt.subplot(
            111)  # type:plt.Axes
        net_ax.set_title('网络图')
        nx.draw_networkx(self.net_gragh, pos, ax=net_ax, **options)
        nx.draw_networkx_edge_labels(
            self.net_gragh, pos, labels, font_family='Times New Roman', font_size=13)

        # 绘制最短路
        if not self.is_find_path:
            return
        # 最短路的每个节点和权重字典
        path = {}
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i+1]
            path[(u, v)] = self.adjacency_mat[u, v]
            self.path_graph.add_edge(u, v, weight=self.adjacency_mat[u, v])

        path_ax = plt.subplot(122)  # type:plt.Axes
        path_ax.set_title('最短路')
        nx.draw_networkx(self.path_graph, pos, ax=path_ax, **options)
        nx.draw_networkx_edge_labels(
            self.path_graph, pos, path, font_family='Times New Roman', font_size=13)


if __name__ == "__main__":
    adjacency_mat = [[0, 1, 12, INF, INF, INF],
                     [INF, 0, 9, 3, INF, INF],
                     [INF, INF, 0, INF, 5, INF],
                     [INF, INF, 4, 0, 13, 15],
                     [INF, INF, INF, INF, 0, 4],
                     [INF, INF, INF, INF, INF, 0]]

    solution = Dijkstra(1, 5, adjacency_mat)
    solution.findPath()
    print(solution.path)
    plt.show()
