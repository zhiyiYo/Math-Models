# coding:utf-8

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


plt.style.use('matlab')

# 原始数据
x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)

# 用于插值的x
x_interp = np.linspace(0, 2*np.pi, 100)

# 使用多种插值方式
fig, axes = plt.subplots(3, 3, num='一维插值')
axes = axes.flatten()  # type:np.ndarray

# 插值方式列表
# nearest, zero为阶梯插值，相当于0阶B样条插值
# linear, slinear 线性插值，相当于1阶B样条插值
# quadratic, cubic 为2阶、3阶B样条插值
# 也可以直接指定拟合的阶次（只能是奇数次）
interp_types = ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
                'previous', 'next', 1]

for i, interp_type in enumerate(interp_types):
    interp_func = interpolate.interp1d(x, y, interp_type)
    y_interp = interp_func(x_interp)
    axes[i].plot(x, y, 'r.', label='origin data')
    axes[i].plot(x_interp, y_interp, label=str(interp_type))
    axes[i].plot(x_interp, np.sin(x_interp), label='sin(x)')
    axes[i].set_title(f'{interp_type} interpolation')
    axes[i].legend()
    
fig.set_tight_layout(True)
plt.show()
