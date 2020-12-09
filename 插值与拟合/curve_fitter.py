# coding:utf-8

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from lmfit.model import ModelResult
from matplotlib.pyplot import Axes, Figure
from matplotlib.backend_bases import MouseButton, MouseEvent

plt.style.use('matlab')


class CurveFitter:
    """ 拟合曲线 """

    def __init__(self):
        self.fitModel = None         # type: Model
        self.polyExpr = None         # type: np.poly1d
        self.fitResult = None      # type: ModelResult

    def polyfit(self, n: int, x, y, isPlot: bool = True, numpoints: int = None, figKwargs: dict = None,
                fig: Figure = None, ax: Axes = None, showLegend: bool = True, **axKwargs) -> Tuple[Figure, Axes]:
        """ 多项式拟合

        Parameters
        ----------
        n : int
            拟合阶次

        x : 1-D array_like
            x轴数据

        y : 1-D array_like
            y轴数据

        isPlot : bool
            是否绘制拟合结果

        numpoints : int
            拟合曲线中包含的数据点数，点数越多曲线越光滑

        figKwargs : dict
            控制figure外观的参数，类型为字典

        fig : `~matplotlib.figure.Figure`
            指定图窗用于绘图，如果为None则新建一个图窗

        ax : `~matplotlib.axes.Axes`
            指定坐标区用于绘图，如果为None没有则新建一个坐标区

        showLegend : bool
            是否显示图例

        **axKwargs : 控制axes外观的参数
        """
        # 检查数据是否符合要求
        x, y = self.__checkData(x, y)
        # 计算拟合多项式的系数
        poly_coeff = np.polyfit(x, y, n)
        # 将多项式系数转换为poly1d对象
        self.polyExpr = np.poly1d(poly_coeff)
        print('多项式拟合表达式为：\n', self.polyExpr, '\n', '----'*30)
        # 绘制曲线
        if not isPlot:
            return
        # 对x轴源数据进行插值
        interplationXdata = self.__getInterplationXData(x, numpoints)
        return self.__plotfit(x, y, interplationXdata, self.polyExpr(
            interplationXdata), figKwargs if figKwargs else {}, fig, ax, showLegend,**axKwargs)

    def modelfit(self, func, params: Parameters, x, y, isPlot: bool = True, numpoints: int = None,
                 figKwargs: dict = None, fig: Figure = None, ax: Axes = None, showLegend: bool = True,
                 **axKwargs) -> Tuple[Figure, Axes]:
        """使用自定义模型进行拟合

        Parameters
        ----------
        func : 可调用对象，函数签名如下::

            func(x, param1, param2, ...)

        params : `~lmfit.Parameters`
            用于曲线拟合的待定参数，必须为每一个参数提供初始值

        x : 1-D array_like
            x轴数据

        y : 1-D array_like
            y轴数据

        isPlot : bool
            是否绘制拟合结果

        numpoints : int
            拟合曲线中包含的数据点数，点数越多曲线越光滑

        figKwargs : dict
            控制figure外观的参数，类型为字典

        fig : `~matplotlib.figure.Figure`
            指定图窗用于绘图，如果为None则新建一个图窗

        ax : `~matplotlib.axes.Axes`
            指定坐标区用于绘图，如果为None没有则新建一个坐标区

        showLegend : bool
            是否显示图例

        **axKwargs : 控制axes外观的参数

        Examples
        --------

        >>> def gaussian(x, amp, mu, wid):
            ... return amp * np.exp(-(x - mu) ** 2 / wid)

        >>> y = gaussian(x, 8, 3, 0.5) + np.random.normal(0, 0.2, x.size)
        >>> x = np.arange(0, 10, 0.1)

        >>> curve_fitter = CurveFitter()

        >>> params = Parameters()
        >>> params.add_many(('amp', 5), ('mu', 2), ('wid', 1))

        >>> curve_fitter.modelfit(gaussian, params, x, y, numpoints=200, title='Gaussian Model Fit')
        >>> plt.show()
        """
        x, y = self.__checkData(x, y)
        # 创建模型并拟合曲线
        self.fitModel = Model(func)
        self.fitResult = self.fitModel.fit(y, params, x=x)
        # 打印拟合结果
        print('模型拟合结果：\n', self.fitResult.fit_report())
        # 绘制拟合曲线
        if not isPlot:
            return
        # 对x轴源数据进行插值
        interplationXdata = self.__getInterplationXData(x, numpoints)
        return self.__plotfit(x, y, interplationXdata, self.fitResult.eval(x=interplationXdata),
                              figKwargs if figKwargs else {}, fig, ax, showLegend, **axKwargs)

    def __plotfit(self, x, y, interplationXdata, fitData, figKwargs: dict = None,
                  fig: Figure = None, ax: Axes = None, showLegend=True, **axKwargs) -> Tuple[Figure, Axes]:
        """ 绘制源数据和拟合曲线 """
        if not fig:
            fig = plt.figure(**figKwargs)  # type:Figure
        if not ax:
            ax = fig.add_subplot()  # type:Axes
        
        ax.plot(x, y, 'r.', interplationXdata, fitData)
        
        ax.set(**axKwargs)
        if showLegend:
            ax.legend(['original data', 'fitted curve'])
        # 鼠标点击画布时显示绘制点
        fig.canvas.mpl_connect(
            'button_release_event', lambda e: self.__mouseRealseEvent(e, fig, ax))
        return (fig, ax)

    def __mouseRealseEvent(self, e: MouseEvent, fig: Figure, ax: Axes):
        """ 鼠标点击画布时显示点击位置 """
        # 当鼠标左键的点击位置是ax并且当前工具栏模式不是缩放或者移动时，打印点击位置并显示
        if e.inaxes is ax and e.button == MouseButton.LEFT and \
                plt.get_current_fig_manager().toolbar.mode not in {'zoom rect', 'pan/zoom'}:
            # 移除上一个点击的点
            self.__removePoint(ax, 'clicked point')
            # 绘制当前点击的点
            ax.plot(e.x, e.y, 'g*', label='clicked point')
            fig.suptitle(f'Clicked Pos : {(e.x, e.y)}')
            fig.canvas.draw()
            print(f'Clicked Pos : {(e.x, e.y)}')

    def __removePoint(self, ax: Axes, label: str):
        """ 移除上一个点击的点 """
        for line in reversed(ax.lines):
            if line.get_label() == label:
                line.remove()
                del line

    def __checkData(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """ 检查数据是否都是一维array_like数组 """
        x = np.array(x)
        y = np.array(y)
        if (x.ndim, y.ndim) != (1, 1):
            raise Exception('源数据必须都是1维的向量')
        return (x, y)

    def __getInterplationXData(self, x, numpoints: int = None):
        """ 得到插值后的x轴数据 """
        if numpoints and numpoints > len(x):
            interplationXdata = np.linspace(
                np.min(x), np.max(x), numpoints)
        else:
            interplationXdata = x
        return interplationXdata


if __name__ == "__main__":

    def gaussian(x, amp, mu, wid):
        return amp * np.exp(-(x - mu) ** 2 / wid)

    # 原始数据
    x = np.arange(0, 10, 0.1)
    y = gaussian(x, 8, 3, 0.5) + np.random.normal(0, 0.2, x.size)
    x0 = np.array([1.1, 1, 0.8, 0.6, 0.4, 0.2, 0])
    y0 = np.array([1600, 1615, 1637, 1660, 1690, 1722, 1748])
    x1 = np.arange(1, 6, 1)
    y1 = 2 * x1 ** 3 - x1 - 1.5 + np.random.randn(x1.size)

    curve_fitter = CurveFitter()
    # 多项式拟合
    curve_fitter.polyfit(1, x0, y0, xlabel=r'$T/N\cdot m$', ylabel=r'$n/r\cdot min^{-1}$',
                         title='First Order Polynomial Fitting')
    curve_fitter.polyfit(3, x1, y1, numpoints=100)
    # 模型拟合
    params = Parameters()
    params.add_many(('amp', 5), ('mu', 2), ('wid', 1))
    curve_fitter.modelfit(gaussian, params, x, y,
                          numpoints=200, title='Gaussian Model Fit')
    plt.show()