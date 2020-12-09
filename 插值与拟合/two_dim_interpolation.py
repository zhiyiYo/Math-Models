# coding:utf-8

from typing import List

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def surface(x, y):
    """ 返回曲面方程的值 """
    return (x+y)*np.exp(-5.0*(x**2 + y**2))


# 创建栅格化数据，步长为复数表示点数，左闭右闭
# 步长为实数表示步长，左闭右开
y, x = np.mgrid[-1:1:20j, -1:1:20j]
z = surface(x, y)

# 使用3阶B样条插值
interp_func = interpolate.interp2d(x, y, z, kind='cubic')
z_interp = interp_func(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

# 创建插值后的坐标点
y_interp, x_interp = np.mgrid[-1:1:100j, -1:1:100j]

# 绘制3D图像
fig = plt.figure('二维插值')  # type:plt.Figure

ax_1 = fig.add_subplot(1, 2, 1, projection='3d')  # type:Axes3D
surf_1 = ax_1.plot_surface(x, y, z, cmap=plt.cm.viridis)
ax_1.set(xlabel='x', ylabel='y', zlabel='z',
                  title='Before Interpolation')
plt.colorbar(surf_1)

ax_2 = fig.add_subplot(1, 2, 2, projection='3d')
surf_2 = ax_2.plot_surface(x_interp, y_interp, z_interp, cmap=plt.cm.viridis)
ax_2.set(xlabel='x', ylabel='y', zlabel='z', title='Cubic Interpolation')
plt.colorbar(surf_2)

plt.show()
