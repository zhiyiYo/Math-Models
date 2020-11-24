from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import networkx as nx

from draw_graph import drawFlow

INF = 1 << 20


class MaximumFlow:
    """ 求取最大流 """

    def __init__(self, edge_list: list, start_node: int, end_node: int):
        """创建最大流对象

        Parameters
        ----------
        edge_list : list
            每个元素代表一条边，元素的格式为 `(u, v, capacity)` 或者 `(u, v, capacity, cost)`

        start_node : int
            起始节点

        end_node : int
            终止节点
        """
        self.edge_list = edge_list
        self.start_node = start_node
        self.end_node = end_node
        self.max_flow_value = 0
        self.max_flow_dict = {}  # type:Dict[Any, Dict[Any, int]]
        self.max_flow_min_cost_dict = {}  # type:Dict[Any, Dict[Any, int]]

    def findMaxFlow(self, is_plot_graph: bool = True):
        """ 寻找最大流 """
        self.flow_graph = nx.DiGraph()
        for edge in self.edge_list:
            self.flow_graph.add_edge(edge[0], edge[1], capacity=edge[2])

        # 计算最大流即每条边的流量
        self.max_flow_value, self.max_flow_dict = nx.maximum_flow(
            self.flow_graph, self.start_node, self.end_node)
        print('最大流为：', self.max_flow_value)
        
        # 将最大流字典转换为列表
        self.max_flow_edges = []
        for node, nodes in self.max_flow_dict.items():
            for node_, flow in nodes.items():
                if flow > 0:
                    self.max_flow_edges.append((node, node_, flow))
        # 绘制图像
        if is_plot_graph:
            # 原始图
            fig = plt.figure()  # type:plt.Figure
            ax = fig.add_subplot(121)
            _, pos = drawFlow(self.edge_list, ax)
            # 最大流图
            ax_ = fig.add_subplot(122)
            self.max_flow_graph, _ = drawFlow(self.max_flow_edges, ax_, pos)
            ax_.set_title('Max Flow Graph')

    def findMaxFlowMinCost(self, is_plot_graph: bool = True):
        """ 寻找最小费用最大流 """
        self.flow_cost_graph = nx.DiGraph()
        for edge in self.edge_list:
            self.flow_cost_graph.add_edge(
                edge[0], edge[1], capacity=edge[2], weight=edge[3])

        # 计算最小费用最大流
        self.max_flow_min_cost_dict = nx.max_flow_min_cost(
            self.flow_cost_graph, self.start_node, self.end_node)
        self.min_cost_value = nx.cost_of_flow(
            self.flow_cost_graph, self.max_flow_min_cost_dict)  # type:float
        print('最小费用为：', self.min_cost_value)

        # 将最大流字典转换为列表
        self.max_flow_min_cost_edges = []  # type:List[Tuple[int,int,int]]
        for node, nodes in self.max_flow_min_cost_dict.items():
            for node_, flow in nodes.items():
                if flow > 0:
                    self.max_flow_min_cost_edges.append((node, node_, flow))
        # 绘制图像
        if is_plot_graph:
            # 原始图
            fig = plt.figure()  # type:plt.Figure
            ax = fig.add_subplot(121)
            _, pos = drawFlow(self.edge_list, ax)
            # 最大流图
            ax_ = fig.add_subplot(122)
            self.max_flow_graph, _ = drawFlow(
                self.max_flow_min_cost_edges, ax_, pos)
            ax_.set_title('Max Flow Min Cost Graph')


if __name__ == "__main__":
    edges = [(0, 1, 10, 4),
             (0, 2, 8, 1),
             (1, 3, 2, 6),
             (2, 1, 5, 2),
             (2, 3, 10, 3),
             (1, 4, 7, 1),
             (3, 4, 4, 2)]
    solver = MaximumFlow(edges, 0, 4)
    solver.findMaxFlowMinCost()
    plt.show()
