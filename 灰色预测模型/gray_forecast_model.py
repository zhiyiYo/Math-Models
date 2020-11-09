from bisect import bisect

import numpy as np
import matplotlib.pyplot as plt


class GrayForecastModel:
    """ 灰色预测模型 """

    GM_1_1 = 0
    GM_2_1 = 1
    GM_1_N = 2
    VERHULST = 3

    def __init__(self, origin_data, model_type: int = 0):
        """ 重置模型

        Parameters
        ----------
        origin_data : 一维array_like原始序列或者二维array_like原始序列\n
            如果是二维原始序列，要求每一行对应一种原始序列，第一行必须是系统特征数据序列，其余行是相关因素数据序列，
            只有模型类型为 GM(1, N) 时允许输入二维数组

        model_type : 灰色预测模型类型，有以下几种::

            - GrayForecastModel.GM_1_1 :   GM(1, 1) 模型，适用于按J型规律发展的序列
            - GrayForecastModel.GM_2_1 :   GM(2, 1) 模型，适用于按Γ型规律发展的序列
            - GrayForecastModel.GM_1_N :   GM(1, N) 模型，适用于具有多影响因素的序列
            - GrayForecastModel.VERHULST : Verhulst 模型，适用于按S型规律发展的序列
        """
        self.setModel(origin_data, model_type)

    def __checkInputData(self, origin_data, model_type: int) -> np.ndarray:
        """ 检查输入序列是否满足要求 """
        origin_data = np.array(origin_data)
        if model_type != self.GM_1_N and origin_data.ndim != 1:
            raise Exception('原始序列必须是一维array_like对象')
        elif model_type == self.GM_1_N and origin_data.ndim != 2:
            raise Exception('原始数据必须是每行对应一种原始序列的二维array_like对象')

        # 进行级比检验
        if model_type != self.GM_1_N:
            n = len(origin_data)
            for k in range(1, n):
                lamda = origin_data[k - 1] / origin_data[k]
                if not np.exp(-2 / (n + 1)) < lamda < np.exp(2 / (n + 2)):
                    print('警告：原始数据无法通过级比检验，需对数据进行平移变换')
                    break
        else:
            n = origin_data.shape[1]
            b = np.vstack([origin_data[:, i] / origin_data[:, i + 1]
                           for i in range(n - 1)]).T
            if not np.logical_and(np.exp(-2 / (n + 1)) < b, b < np.exp(2 / (n + 2))).all():
                print('警告：原始数据无法通过级比检验，需对数据进行平移变换')
        return origin_data

    def forecast(self, num=1, is_plot: bool = True, t=None, t_=None, **ax_kwargs) -> np.ndarray:
        """ 进行预测

        Parameters
        ----------
        num : 预测点数\n
        is_plot : 是否绘制曲线\n
        t : 与原始序列相对应的一维array_like时间序列\n
        t_ : 与预测序列相对应的一维array_like时间序列\n
        **ax_kwargs : 控制axes外观的参数
        """
        self.forecast_series = np.zeros(
            (len(self.origin_series) + num,))  # type:np.ndarray
        self.forecast_series[0] = x_0 = self.origin_series[0]
        a, b, b_vec = self.a, self.b, self.b_vec

        # 根据模型类型计算预测序列
        if self.model_type in [self.GM_1_1, self.VERHULST]:
            for i in range(1, len(self.forecast_series)):
                if self.model_type == self.GM_1_1:
                    self.forecast_series[i] = (
                        x_0 - b / a) * (1 - np.exp(a)) * np.exp(-a * i)
                elif self.model_type == self.VERHULST:
                    self.forecast_series[i] = a * x_0 / \
                        (b * x_0 + (a - b * x_0) * np.exp(a * i))
        elif self.model_type == self.GM_2_1:
            x_1 = [(b / a ** 2 - x_0 / a) * np.exp(-a * i) + b / a * (i + 1) +
                   (x_0 - b / a) * (1 + a) / a for i in range(len(self.forecast_series))]
            self.forecast_series[1:] = np.ediff1d(x_1)
        elif self.model_type == self.GM_1_N:
            x_1 = [(x_0 - np.sum(1 / a * b_vec * self.X_1_mat)) * np.exp(-a * i) + 1 / a *
                   np.sum(b_vec * self.X_1_mat) for i in range(len(self.forecast_series))]
            self.forecast_series[1:] = np.ediff1d(x_1)
        # 模型精度检验：后验差检验
        self.__checkForecastData()
        # 绘图
        if is_plot:
            fig, ax = plt.subplots()
            ax.plot(t, self.origin_series, 'r.', label=r'$X^{(0)}$')
            ax.plot(t_, self.forecast_series, 'b^-',
                    label=r'$\hat{X} ^{(0)}$', markersize=3)
            ax.set(**ax_kwargs)
            plt.legend()
        return self.forecast_series

    def __checkForecastData(self):
        """ 对预测数据进行后验差检验 """
        # 绝对残差序列
        self.abs_error_series = np.abs(
            self.forecast_series[: len(self.origin_series)] - self.origin_series)  # type:np.ndarray
        self.relative_error_series = self.abs_error_series / self.origin_series
        # 方差比
        self.var_ratio = float(
            self.abs_error_series.std() / self.origin_series.std())
        # 平均相对误差
        self.avr_relative_error = float(np.sum(self.relative_error_series) /
                                        len(self.abs_error_series))
        # 关联度
        org_data = self.origin_series / self.origin_series[0]  # 初值化序列
        forecast_data = self.forecast_series[:len(
            self.origin_series)] / self.forecast_series[0]
        error = np.abs(org_data - forecast_data)
        max_error, min_error = np.max(error), np.min(error)
        cor_degree = (min_error + 0.5 * max_error) / \
            (error + 0.5 * max_error)
        self.cor_degree = float(sum(cor_degree) / len(cor_degree))

        print(f'{"关联度：":<17}', np.round(self.cor_degree, 4))
        print(f'{"均方差比值：":<15}', np.round(self.var_ratio, 4))
        print(f'{"平均相对误差：":<14}', np.round(self.avr_relative_error, 5))
        # 相对误差检验
        model_level = ['优', '合格', '勉强合格', '不合格']
        print(
            f"{'关联度检验：':<16}{['不通过','通过'][self.cor_degree > 0.6]}\n"
            f"{'后验差检验精度：':<14}{model_level[bisect([0.35, 0.5, 0.65], self.var_ratio)]}\n"
            f"{'相对误差检验精度：':<13}{model_level[bisect([0.01,0.05,0.1], self.avr_relative_error)]}")
        print('--'*30)

    def setModel(self, origin_data, model_type: int = 0):
        """ 重置模型

        Parameters
        ----------
        origin_data : 一维array_like原始序列或者二维array_like原始序列\n
            如果是二维原始序列，要求每一行对应一种原始序列，第一行必须是系统特征数据序列，其余行是相关因素数据序列，
            只有模型类型为 GM(1, N) 时允许输入二维数组

        model_type : 灰色预测模型类型，有以下几种::

            - GrayForecastModel.GM_1_1 :   GM(1, 1) 模型，适用于按J型规律发展的序列
            - GrayForecastModel.GM_2_1 :   GM(2, 1) 模型，适用于按Γ型规律发展的序列
            - GrayForecastModel.GM_1_N :   GM(1, N) 模型，适用于具有多影响因素的序列
            - GrayForecastModel.VERHULST : Verhulst 模型，适用于按S型规律发展的序列
        """
        # 发展系数和灰色作用量
        self.a = 0
        self.b = 0
        # 原始数据
        self.origin_data = self.__checkInputData(origin_data, model_type)
        # 原始系统特征数据序列
        self.origin_series = self.origin_data if model_type != self.GM_1_N else self.origin_data[
            0]
        # 检查阶数
        if not 0 <= model_type <= 3:
            raise Exception('不存在该灰色预测模型')
        self.model_type = model_type
        # 预测序列
        self.forecast_series = self.origin_series.copy()
        # 一次累加生成序列
        self.X_1 = np.cumsum(self.origin_series)  # type:np.ndarray
        self.X_1_mat = np.cumsum(
            self.origin_data[1:], axis=0 if self.origin_data.ndim == 1 else 1)
        # 一次累减生成序列
        self.X_0 = np.append(
            self.origin_series[0], np.diff(self.origin_series))

        # 紧邻均值生成序列 Z_1、Z_1_mat 和 Yn
        n = len(self.origin_series) - 1
        if model_type in [self.GM_1_1, self.GM_1_N]:
            data = self.X_1
            Yn = self.origin_series[1:].reshape((n, 1))
        else:
            data = self.origin_series
            Yn = self.X_0[1:].reshape((n, 1))
        self.Z_1 = np.array([0.5 * (data[i] + data[i + 1]) for i in range(n)])

        # 根据模型类型构造矩阵B
        if model_type == self.GM_1_1:
            B = np.hstack((-self.Z_1.reshape((n, 1)), np.ones((n, 1))))
        elif self.model_type == self.GM_2_1:
            B = np.hstack(
                (-self.origin_series[1:].reshape((n, 1)), -self.Z_1.reshape((n, 1)), np.ones((n, 1))))
        elif self.model_type == self.GM_1_N:
            B = np.hstack((-self.Z_1.reshape((n, 1)),
                           self.X_1_mat.T[1:, :]))
        elif model_type == self.VERHULST:
            B = np.hstack((-self.Z_1.reshape((n, 1)),
                           (self.Z_1 ** 2).reshape((n, 1))))

        # 最小二乘法估计参数
        a_hat = (np.linalg.inv(B.T @ B) @ B.T @ Yn).flatten()
        self.a = a_hat[0]
        self.b = a_hat[1]
        self.b_vec = a_hat[1:].reshape((len(a_hat[1:]), 1))


if __name__ == "__main__":
    # 票房预测
    year = np.arange(2007, 2018)
    year_ = np.arange(2007, 2020)
    # 原始票房
    origin_data = np.array([33.27, 43.41, 62.06, 101.72, 131.15,
                            170.73, 217.69, 296.39, 440.69, 457.12, 559.11])
    GM = GrayForecastModel(origin_data, GrayForecastModel.GM_1_1)
    plt.style.use('matlab')
    plt.rcParams['xtick.minor.visible'] = False
    # 预测并绘图
    GM.forecast(2, True, year, year_, xlabel='Year', ylabel='Box office')
    plt.legend(['actual box office', 'predict box office'])
    plt.xticks(year_, [str(i) for i in range(2007, 2020)])

    # GM(2,1)预测
    origin_data = [2.874, 3.278, 3.39, 3.679, 3.77, 3.8]
    hour = np.arange(0, 6)
    hour_ = np.arange(0, 16)
    GM.setModel(origin_data, GrayForecastModel.VERHULST)
    GM.forecast(10, True, hour, hour_, xlabel='Hour/h',
                ylabel='Hour/h', xlim=(0, 16))
    plt.legend(['actual num', 'predict num'])
    _ = plt.xticks(hour_, [str(i) for i in hour_])

    # GM(1,N)预测
    org_data = [[560823, 542386, 604834, 591248, 583031, 640636,
                 575688, 689637, 570790, 519574, 614677],
                [104, 101.8, 105.8, 111.5, 115.97, 120.03,
                 113.3, 116.4, 105.1, 83.4, 73.3],
                [135.6, 140.2, 140.1, 146.9, 144, 143,
                 133.3, 135.7, 125.8, 98.5, 99.8],
                [131.6, 135.5, 142.6, 143.2, 142.2,
                 138.4, 138.4, 135, 122.5, 87.2, 96.5],
                [54.2, 54.9, 54.8, 56.3, 54.5, 54.6,
                 54.9, 54.8, 49.3, 41.5, 48.9]]
    GM.setModel(org_data, GM.GM_1_N)
    GM.forecast(2, True, np.arange(0, 11), np.arange(
        0, 13), xlabel='Hour/h', ylabel='Hour/h')
    print(GM.a, '\n', GM.b_vec)
    plt.show()
