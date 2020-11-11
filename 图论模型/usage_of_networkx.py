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

print(f'从起点{source}到终点{target}的最短路为：{path}\n'
      f'从起点{source}到终点{target}的最短路长：{min_len}\n'
      f'从起点{source}到各个顶点的最短路长为：{dist}')

plt.title('Network')
plt.show()
