# coding:utf-8

import networkx as nx
import matplotlib.pyplot as plt

from draw_graph import drawMST, drawNetwork

INF = 1 << 20


class MinSpanningTree:
    """ 最小生成树 """

    INF = 1 << 20
    PRIM = 0
    KRUSKAL = 1

    def __init__(self, adjacency_mat):
        """ 创建Prime算法对象

        Paramters
        ---------
        adjacency_mat : 二维array_like邻接矩阵
        """
        self.__algorithm_dict = {0: self.__prim_mst, 1: self.__kruskal_mst}
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

    def findMST(self, algorithm: int = 0) -> list:
        """ 寻找最小生成树

        Parameters
        ----------
        algorithm : int\n
            使用的算法，有以下几种:
              - MinSpanningTree.PRIM   :  prim算法
              - MinSpanningTree.KRUSKAL:  kruskal算法  
        """
        if not 0 <= algorithm <= 1:
            raise Exception('不存在这种算法')
        self.MST.clear()
        self.__algorithm_dict[algorithm]()
        # 如果树枝列表中除了第一个元素外仍然存在无穷大说明图不连通
        self.is_find_MST = sum(edge[2] == INF for edge in self.MST) < 2
        self.showGraph()
        return self.MST
        

    def __kruskal_mst(self):
        """ kruskal算法求最小生成树 """
        edges = [[i, j, self.adjacency_mat[i][j]]
                 for i in range(self.node_num) for j in range(i, self.node_num) if i != j]
        edges.sort(key=lambda edge: edge[2])
        # ancestor_nodes[i]记录节点i的祖先节点
        ancestor_nodes = list(range(self.node_num))

        while len(self.MST) < self.node_num - 1:
            u, v, w = edges[0]
            edges.pop(0)
            # 如果节点的祖先不同(不构成环)则将其添加到最小生成树中
            if ancestor_nodes[u] != ancestor_nodes[v]:
                self.MST.append([u, v, w])
                # 得到两个节点的祖先
                min_node = min(ancestor_nodes[u], ancestor_nodes[v])
                max_node = max(ancestor_nodes[u], ancestor_nodes[v])
                # 更新祖先节点
                ancestor_nodes = [min_node if i ==
                                  max_node else i for i in ancestor_nodes]

    def __prim_mst(self):
        """ prim算法求最小生成树 """
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

    def showGraph(self):
        """ 显示网络图和最短路 """
        # 绘制网络图
        fig = plt.figure()  #type:plt.Figure
        ax = fig.add_subplot(121) if self.is_find_MST else fig.add_subplot()
        self.net_gragh, pos = drawNetwork(
            self.adjacency_mat, ax=ax, is_digragh=False)
        # 如果图联通则绘制最小生成树
        if self.is_find_MST:
            MST_ax = fig.add_subplot(122)
            self.MST_graph, pos = drawMST(self.MST, MST_ax, pos)
            MST_ax.set_title('MST')


if __name__ == "__main__":
    _ = INF
    adjacency_mat = [[0,  10, _,  _,  _,  11, _,  _,  _],
                     [10, 0,  18, _,  _,  _,  16, _,  12],
                     [_,  18, 0,  22, _,  _,  _,  _,  8],
                     [_,  _,  22, 0,  20,  _, _,  16, 21],
                     [_,  _,  _,  20, 0,  26, 7,  19, _],
                     [11, _,  _,  _,  26, 0,  17, _,  _],
                     [_,  16, _,  _,  7,  17, 0,  19, _],
                     [_,  _,  _,  16, 19, _,  19, 0,  _],
                     [_,  12, 8,  21, _,  _,  _, _,   0]]

    pr = MinSpanningTree(adjacency_mat)
    pr.findMST(pr.KRUSKAL)
    pr.findMST(pr.PRIM)
    plt.show()
