# coding:utf-8

import numpy as np
from scipy.integrate.odepack import odeint
import matplotlib.pyplot as plt

from typing import Tuple


class PopulationCompetitionModel:
    """ 种群竞争模型 """

    def __init__(self, r1=1, r2=1, n1=100, n2=100, s1=1, s2=1, x10=0, x20=0):
        """创建种群竞争模型对象

        Parameters
        ----------
        r1 : float
            种群2的自然增长率

        r2 : float
            种群2的自然增长率

        n1 : int
            种群1的最大环境容纳量, by default 100

        n2 : int
            种群2的最大环境容纳量, by default 100

        s1 : float
            种群2的消耗资源量对种群1的资源消耗量的比值

        s2 : float
            种群1的消耗资源量对种群2的资源消耗量的比值

        x10 : int
            种群1的初始值

        x20 : int
            种群2的初始值
        """
        self.r1 = r1
        self.r2 = r2
        self.n1 = n1
        self.n2 = n2
        self.s1 = s1
        self.s2 = s2
        self.x10 = x10
        self.x20 = x20

    def solve(self, t=None, is_plot_curve: bool = True, is_plot_phase_orbit: bool = True) -> tuple:
        """ 求解种群竞争问题

        Parameters
        ----------
        t : 1 dim array_like
            时间向量

        is_plot_curve : bool
            是否绘制种群生长曲线

        is_plot_phase_orbit : bool
            是否绘制相轨图

        Returns
        -------
        t : `~np.ndarray`
            1维时间向量

        x1 : `~np.ndarray`
            种群1的数量数组

        x2 : `~np.ndarray`
            种群2的数量数组
        """
        self.t = np.arange(0, 20, 0.02) if t is None else np.array(t)
        self.x_list = odeint(
            self.__deriv, [self.x10, self.x20], self.t)  # type:np.ndarray
        self.x1 = self.x_list[:, 0]
        self.x2 = self.x_list[:, 1]
        # 绘制曲线
        if is_plot_curve:
            self.plotPopulationCurve()
        if is_plot_phase_orbit:
            self.plotPhaseOrbitDiagram()

        return (self.t, self.x1, self.x2)

    def __deriv(self, x_list, t):
        """ 创建微分方程组 """
        x1, x2 = x_list
        eq1 = self.r1 * x1 * (1 - x1 / self.n1 - self.s1 * x2 / self.n2)
        eq2 = self.r2 * x2 * (1 - x2 / self.n2 - self.s2 * x1 / self.n1)
        return [eq1, eq2]

    def plotPopulationCurve(self) -> Tuple[plt.Figure, plt.Axes]:
        """ 绘制种群生长曲线

        Returns
        -------
        fig : `~matplotlib.figure.Figure` 图窗

        ax : `~matplotlib.axes.Axes` 坐标区
        """
        fig, ax = plt.subplots(1, 2, num='种群生长曲线')
        ax[0].plot(self.t, self.x1, self.t, self.x2)
        ax[0].legend(['Population 1', 'Population 2'])
        ax[0].set(xlabel=r'$t/day$', ylabel=r'$Population$',
                  title=r'$Population\ Growth\ Curve$')
        # 绘制相轨迹
        ax[1].plot(self.x1, self.x2)
        ax[1].set(xlabel=r'$x_1$', ylabel=r'$x_2$',
                  title=r'$Phase\ Trajectories$')
        fig.set_tight_layout(True)
        return (fig, ax)

    def plotPhaseOrbitDiagram(self) -> Tuple[plt.Figure, plt.Axes]:
        """ 绘制相轨迹流图

        Returns
        -------
        fig : `~matplotlib.figure.Figure` 图窗

        ax : `~matplotlib.axes.Axes` 坐标区
        """
        fig = plt.figure('种群竞争模型相轨图')  # type:plt.Figure
        ax = fig.add_subplot()  # type:plt.Axes
        X2, X1 = np.mgrid[-100: 2 * self.n1 :2, -100: 2 * self.n2 :2]
        u, v = self.__deriv([X1, X2], 0)
        r = np.sqrt(u ** 2 + v ** 2)
        stream = ax.streamplot(X1, X2, u, v, color=r, cmap=plt.cm.brg,density=1.5)
        fig.colorbar(stream.lines)
        ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$',
               title=r'$Population\ Phase\ Trajectories$')
        return (fig, ax)


if __name__ == "__main__":
    t = np.arange(0, 100, 0.01)
    model = PopulationCompetitionModel(0.1, 0.3, 500, 333, 1/3, 1/3, 100, 150)
    plt.style.use('matlab')
    model.solve(t)
    plt.show()
