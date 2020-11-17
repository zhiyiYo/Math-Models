import lmfit
from itertools import product
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

INF = 1 << 20


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


def drawNetwork(adjacency_mat, ax: plt.Axes = None, pos: dict = None,
                is_digragh: bool = True) -> Tuple[Union[nx.DiGraph, nx.Graph], dict]:
    """ 绘制网络图

    Parameters
    ----------
    adjacency_mat : 二维`array_like`\n
        邻接矩阵

    ax : `~matplotlib.axes.Axes` \n
        用于绘图的坐标区，如果为None则新建一个

    pos : dict\n
        控制绘图的布局字典

    is_digragh : bool\n
        是否绘制为有向图，默认为True

    Returns
    -------
    G : `~nx.DiGragh`\n
        有向图

    pos : dict\n
        记录G中各节点位置的字典
    """
    G = nx.DiGraph() if is_digragh else nx.Graph()
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


def drawPath(adjacency_mat, path: list, ax: plt.Axes = None,
             pos: dict = None, is_digragh: bool = True) -> Tuple[Union[nx.DiGraph, nx.Graph], dict]:
    """ 绘制最短路

    Parameters
    ----------
    adjacency_mat : 二维`array_like`\n
        邻接矩阵

    path : list\n
        记录路径中各节点的列表

    ax : `~matplotlib.axes.Axes` \n
        用于绘图的坐标区，如果为None则新建一个

    pos : dict\n
        控制绘图的布局字典

    is_digragh : bool\n
        是否绘制为有向图，默认为True

    Returns
    -------
    G : `~nx.DiGragh`\n
        有向图

    pos : dict\n
        记录G中各节点位置的字典
    """
    G = nx.DiGraph() if is_digragh else nx.Graph()

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


def drawMST(MST_mat, ax: plt.Axes = None, pos: dict = None) -> Tuple[Union[nx.DiGraph, nx.Graph], dict]:
    """ 绘制最小生成树

    Parameters
    ----------
    MST_mat : 二维`array_like`\n
        最小生成树矩阵，矩阵的每一行的前两个元素是树枝的节点，第三个元素是树枝权值

    ax : ~matplotlib.axes.Axes`\n
        用于绘图的坐标区，如果为None则新建一个

    pos : dict\n
        控制绘图的布局字典
    """
    # 生成邻接矩阵
    MST_mat = np.array(MST_mat)
    node_num = len(MST_mat) + 1
    adjacency_mat = np.zeros((node_num, node_num))
    for i in range(node_num - 1):
        row, col = int(MST_mat[i, 0]), int(MST_mat[i, 1])
        adjacency_mat[row, col] = MST_mat[i, 2]
        adjacency_mat[col, row] = MST_mat[i, 2]
    adjacency_mat[adjacency_mat == 0] = INF

    # 绘图
    return drawNetwork(adjacency_mat, ax, pos, False)
