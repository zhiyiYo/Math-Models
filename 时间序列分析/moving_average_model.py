# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


class MovingAverageModel:
    """ 移动平均模型 """
    def __init__(self, time_series):
        """ 初始化对象

        Parameters
        ----------
        time_series : 1-dim array-like
            时间序列
        """
        self.setTimeSeires(time_series)

    def simpleAverage(self, N, time_series=None):
        """ 简单平均法

        Parameters
        ----------
        N : int
            移动平均项数

        time_series : 1-dim array-like
            时间序列，为 None 时使用初始化对象时用的 time_series
        """
        time_series = self.time_series if time_series is None else np.array(
            time_series)
        n = len(time_series)
        if N >= n:
            raise Exception('引动平均项数必须小于时间序列项数')
        # 计算预测序列
        forecast_series = np.array(
            [time_series[i - N:i].mean() for i in range(N, n + 1)])
        # 计算标准误差
        S = np.sqrt(((time_series[N:] - forecast_series[:-1])**2).mean())
        return forecast_series, S

    def weightedAverage(self, N, weights, time_series=None):
        """ 加权平均法

        Paramters
        ---------
        N : int
            移动平均项数

        weights : 1-dim array-like
            权重向量，从左到右离预测项越来越近

        time_series : 1-dim array-like
            时间序列，为 None 时使用初始化对象时用的 time_series
        """
        time_series = self.time_series if time_series is None else np.array(
            time_series)
        n = len(time_series)
        if N >= n:
            raise Exception('引动平均项数必须小于时间序列项数')
        forecast_series = np.zeros(n - N + 1)  # type:np.ndarray
        weights = np.array(weights)
        for i in range(len(forecast_series)):
            forecast_series[i] = (time_series[i:i + N] * weights /
                                  weights.sum()).sum()
        # 计算相对误差
        relative_error = (1 -
                          forecast_series[:-1].sum() / time_series[N:].sum())
        # 修正预测值
        forecast_series[-1] = forecast_series[-1] / (1 - relative_error)
        return forecast_series

    def trendAveraging(self, N, T=1):
        """ 趋势平均法

        Paramters
        ---------
        N : int
            移动平均项数

        T : int
            预测项数
        """
        # 一次移动平均
        M1_series, _ = self.simpleAverage(N)
        # 二次移动平均
        M2_series, _ = self.simpleAverage(N, M1_series)
        # 计算预测序列
        a = 2 * M1_series[-1] - M2_series[-1]
        b = 2 * (M1_series[-1] - M2_series[-1]) / (N - 1)
        forecast_series = np.array([a + b * i for i in range(1, T + 1)])
        return forecast_series

    def setTimeSeires(self, time_series):
        """ 设置时间序列

        time_series : 1-dim array-like
            时间序列
        """
        self.time_series = np.array(time_series)  # type:np.ndarray
        self.n = len(self.time_series)


if __name__ == "__main__":

    y = [
        533.8, 574.6, 606.9, 649.8, 705.1, 772.0, 816.4, 892.7, 963.9, 1015.1,
        1102.7
    ]
    model = MovingAverageModel(y)
    # 均值滤波
    model.simpleAverage(4)
    model.setTimeSeires(
        [6.35, 6.20, 6.22, 6.66, 7.15, 7.89, 8.72, 8.94, 9.28, 9.8])
    forecast_series = model.weightedAverage(3, [1, 2, 5])
    # 绘制曲线
    plt.style.use('matlab')
    fig, ax = plt.subplots()
    t = [1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988]
    ax.plot(t, model.time_series, 'bo-')
    ax.plot(t[3:] + [1989], forecast_series, 'r^--')
    ax.legend(['original data', 'forecast data'])
    plt.xticks(t + [1989], t + [1989])
    # 趋势移动平均
    y = [
        676, 825, 774, 716, 940, 1159, 1384, 1524, 1668, 1688, 1958, 2031,
        2234, 2566, 2820, 3006, 3093, 3277, 3514, 3770, 4107
    ]
    model.setTimeSeires(y)
    print(model.trendAveraging(6, 2))
    plt.show()
