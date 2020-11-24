import networkx as nx
import matplotlib.pyplot as plt


edges = [(0, 1, {'weight': 1}),
         (0, 2, {'weight': 12}),
         (1, 2, {'weight': 9}),
         (1, 3, {'weight': 3}),
         (2, 4, {'weight': 5}),
         (3, 2, {'weight': 4}),
         (3, 4, {'weight': 13}),
         (3, 5, {'weight': 15}),
         (4, 5, {'weight': 4})]
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
edge_labels = {(u, v): dic['weight'] for u, v, dic in edges}

# 创建网络图并绘制网络图
graph = nx.DiGraph(edges)
pos = nx.shell_layout(graph)  # 布局
nx.draw_networkx(graph, pos, **options)
nx.draw_networkx_edge_labels(graph, pos, edge_labels)

# 使用dijkstra算法寻找最短路
source, target = 0, 5
path = nx.dijkstra_path(graph, source, target)
min_len = nx.dijkstra_path_length(graph, source, target)
predecessor, dist = nx.dijkstra_predecessor_and_distance(graph, source)

# 使用弗洛伊德算法寻找最短路
distance_mat = nx.floyd_warshall_numpy(graph)  # 计算每个节点之间的最短路程矩阵
pred, distance_dict = nx.floyd_warshall_predecessor_and_distance(graph)

# 从pred中提取最短路
print(f'从起点{source}到终点{target}的最短路为：{nx.reconstruct_path(0,5,pred)}\n'
      f'从起点{source}到终点{target}的最短路为：{path}\n'
      f'从起点{source}到终点{target}的最短路长：{min_len}\n'
      f'从起点{source}到各个顶点的最短路长为：{dist}')

# 使用Prim算法寻找最小生成树
G = graph.to_undirected()
MST = nx.minimum_spanning_tree(G,algorithm='prim')  # type:nx.Graph
print(MST.edges(data=True))

plt.title('Network')
plt.show()
