# coding:utf-8

import networkx as nx
import matplotlib.pyplot as plt

from draw_graph import *

INF = 1 << 20


class Prim:
    """ Prim算法求最小生成树 """

    INF = 1 << 20

    def __init__(self, adjacency_mat):
        """ 创建Prime算法对象

        Paramters
        ---------
        adjacency_mat : 二维array_like邻接矩阵
        """
        self.setModel(adjacency_mat)

    def setModel(self, adjacency_mat):
        """ 重置模型

        Paramters
        ---------
        adjacency_mat : 二维邻接矩阵
        """
        self.adjacency_mat = adjacency_mat
        self.node_num = len(self.adjacency_mat)
        self.is_find_MST = False
        # 最小生成树 n×3 矩阵，每一行代表一个树枝，分别为起点、终点和权值
        self.MST = []

    def findMST(self):
        """ 寻找最小生成树 """
        # 已遍历的节点列表
        nopassed_nodes = list(range(self.node_num))
        # head_nodes[i]保存与节点i最近的节点 j
        head_nodes = [0] * self.node_num
        # 记录已遍历的节点的最短树枝的列表，distance_list[i]表示与i节点最近的树枝
        distance_list = [INF] * self.node_num
        while nopassed_nodes:
            node = 0     # 起始节点
            min_distance = INF + 1
            # 寻找最短树枝及其对应的结点
            for i in range(self.node_num):
                if distance_list[i] < min_distance and i in nopassed_nodes:
                    min_distance = distance_list[i]
                    node = i
            # 将节点移出到列表
            nopassed_nodes.remove(node)
            # 更新树枝矩阵
            self.MST.append(
                [node, head_nodes[node], min_distance])
            # 更新与已遍历的节点最近的树枝列表
            for i in range(self.node_num):
                if i in nopassed_nodes and distance_list[i] > self.adjacency_mat[node][i]:
                    distance_list[i] = self.adjacency_mat[node][i]
                    head_nodes[i] = node
        # 删除0-0树枝
        self.MST.pop(0)
        # 如果树枝列表中除了第一个元素外仍然存在无穷大说明图不连通
        self.is_find_MST = sum(i == INF for i in distance_list) < 2
        self.showGraph()

    def showGraph(self):
        """ 显示网络图和最短路 """
        # 绘制网络图
        ax = plt.subplot(121) if self.is_find_MST else plt.subplot(111)
        self.net_gragh, pos = drawNetwork(
            self.adjacency_mat, ax, is_digragh=False)
        # 如果图联通则绘制最小生成树
        if self.is_find_MST:
            MST_ax = plt.subplot(122)
            self.MST_graph, pos = drawMST(self.MST, MST_ax, pos)


if __name__ == "__main__":
    _ = INF
    adjacency_mat = [[0,  1, 12, _, _,  _],
                     [1,  0, 9, 3,  _,  _],
                     [12, 9, 0, 4,  5,  _],
                     [_,  3, 4, 0,  13, 15],
                     [_, _,  5, 13, 0,  4],
                     [_, _,  _, 15, 4,  0]]

    pr = Prim(adjacency_mat)
    pr.findMST()
    plt.show()
