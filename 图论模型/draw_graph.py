from itertools import product
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

INF = float('inf')
import lmfit

lmfit.Model

# 控制绘图的样式
draw_network_options = {
    "font_family": "consolas",
    "edge_color": "#A0D0F0",
    "font_color": "white",
    "arrowstyle": '->',
    "node_size": 500,
    "arrowsize": 14,
    "font_size": 16,
    "width": 1.5,       # edge的宽度
}


def drawNetwork(adjacency_mat, ax: plt.Axes = None, pos: dict = None) -> Tuple[nx.DiGraph, dict]:
    """ 绘制网络图

    Parameters
    ----------
    adjacency_mat : 二维array_like邻接矩阵\n
    ax : `~matplotlib.axes.Axes` 用于绘图的坐标区，如果为None则新建一个\n
    pos : 控制绘图的布局字典

    Returns
    -------
    G : `~nx.DiGragh` 有向图\n
    pos : 记录G中各节点位置的字典
    """
    G = nx.DiGraph(name='网络图')
    adjacency_mat = np.array(adjacency_mat)
    if adjacency_mat.ndim != 2:
        raise Exception('邻接矩阵必须是二维array_like数组')

    # 添加节点和边
    ra = range(len(adjacency_mat))
    for i, j in product(ra, ra):
        if adjacency_mat[i, j] != INF and i != j:
            G.add_edge(i, j, weight=adjacency_mat[i, j])

    # 每条边的权重标签
        labels = {(u, v): weight for u, v,
                  weight in G.edges.data('weight')}

     # 绘图参数
    pos = nx.shell_layout(G) if not pos else pos  # 布局

    # 如果ax为None则新建坐标区
    ax = plt.subplot(111) if not ax else ax
    ax.set_title('Network')
    nx.draw_networkx(G, pos, ax=ax, **draw_network_options)
    nx.draw_networkx_edge_labels(
        G, pos, labels, font_family='Times New Roman', font_size=13)

    return (G, pos)


def drawPath(adjacency_mat, path: list, ax: plt.Axes = None, pos: dict = None) -> Tuple[nx.DiGraph, dict]:
    """ 绘制最短路

    Parameters
    ----------
    adjacency_mat : 二维array_like邻接矩阵\n
    path : 记录路径中各节点的列表\n
    ax : `~matplotlib.axes.Axes` 用于绘图的坐标区，如果为None则新建一个\n
    pos : 控制绘图的布局字典

    Returns
    -------
    G : `~nx.DiGragh` 有向图\n
    pos : 记录G中各节点位置的字典
    """
    G = nx.DiGraph()

    # 最短路的每个节点和权重字典
    labels = {}  # type:Dict[Tuple[Any, Any], Any]
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        labels[(u, v)] = adjacency_mat[u, v]
        G.add_edge(u, v, weight=adjacency_mat[u, v])

    pos = nx.shell_layout(G) if not pos else pos
    ax = plt.subplot(111) if not ax else ax
    ax.set_title('Shortest Path')
    nx.draw_networkx(G, pos, ax=ax, **draw_network_options)
    nx.draw_networkx_edge_labels(
        G, pos, labels, font_family='Times New Roman', font_size=13)

    return (G, pos)
