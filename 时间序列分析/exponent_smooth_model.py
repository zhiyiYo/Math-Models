import numpy as np
import matplotlib.pyplot as plt


class ExponentSmoothModel:
    """ 指数平滑模型 """
    def __init__(self, time_series):
        """ 初始化对象

        Parameters
        ----------
        time_series: 1-dim array-like
            时间序列
        """
        self.time_series = np.array(time_series)  # type:np.ndarray
        self.n = len(self.time_series)
        self.__smooth_func = {1: self.__firstOrderSmooth}

    def forecast(self, alpha=0.5, order=1, time_series=None):
        """ 预测数据

        Parameters
        ----------
        alpha: float
            平滑系数

        order: int
            平滑阶数
        """
        time_series = self.time_series if time_series is None else np.array(
            time_series)
        if order < 1 or order > 3:
            raise Exception('阶数只能为 1、2 或 3')
        return self.__smooth_func[order](time_series, alpha)

    def __firstOrderSmooth(self, time_series: np.ndarray, alpha):
        """ 一次平滑 """
        # 初始预测值
        forecast_series = [time_series[:2].mean()]
        # 计算预测值
        for i in range(len(time_series)):
            forecast_series.append(forecast_series[i] * (1 - alpha) +
                                   alpha * time_series[i])
        # 计算标准误差
        S = np.sqrt(((forecast_series[:-1] - time_series)**2).mean())
        return forecast_series, S


if __name__ == '__main__':
    t = [i + 1976 for i in range(12)]
    t_ = t + [1988]
    y = [50, 52, 47, 51, 49, 48, 51, 40, 48, 52, 51, 59]
    model = ExponentSmoothModel(y)
    # 一次指数平滑
    forecast_series, _ = model.forecast(0.5)
    # 绘制曲线
    plt.style.use('matlab')
    fig_1, ax_1 = plt.subplots(num='一次指数平滑')
    ax_1.plot(t, y, label='original data')
    ax_1.plot(t_, forecast_series, label='forecast data')
    ax_1.legend()
    plt.xticks(t_, t_)
    plt.show()