# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR


class LinearRegression:
    """ 多元线性回归 """

    def __init__(self):
        self.model = None

    def UnitaryRegression(self, x, y) -> LR:
        """ 一元回归

        Parameters
        ----------
        x : 1-D array_like
            特性值

        y : 1-D array_like
            标签值

        Returns
        -------
        model : `sklearn.linear_model.LinearRegression`
            完成拟合的模型
        """
        x = np.array(x).reshape(-1, 1)  # type:np.ndarray
        y = np.array(y).reshape(-1, 1)  # type:np.ndarray
        self.model = LR()
        self.model.fit(x, y)
        self.printRegressionEq()
        return self.model

    def MultipleRegression(self, X, y, isBoxPlot: bool = True) -> LR:
        """ 多元回归

        Parameters
        ----------
        X : 2-D array_like 
            形状为 (n_samples, n_features) 的特性值矩阵

        y : 1-D array_like
            长度为 n_samples 的标签值

        isBoxPlot : bool
            是否绘制箱线图，如果绘制，要求X和y必须`pandas.DataFrame`和`pandas.Series`

        Returns
        -------
        model : `sklearn.linear_model.LinearRegression`
            完成拟合的模型
        """
        if isBoxPlot and isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            df = pd.concat([X, y], axis=1)
            df.boxplot()

        X = np.array(X)  # type:np.ndarray
        y = np.array(y)  # type:np.ndarray
        # 数据标准化
        X_std = X.std(axis=0)
        X_mean = X.mean(axis=0)
        X_standard = (X - X_mean) / X_std
        # 计算标准化回归方程
        self.model = LR()
        self.model.fit(X_standard, y)
        # 转换为原始数据的回归方程
        self.model.intercept_ = self.model.intercept_ - \
            self.model.coef_.dot(X_mean / X_std)
        self.model.coef_ = self.model.coef_ / X_std
        # 打印回归方程表达式
        self.printRegressionEq()
        return self.model

    def printRegressionEq(self):
        """ 打印回归方程 """
        if self.model is None:
            print('无拟合模型')
        # 打印回归方程表达式
        expr = f'y = {self.model.intercept_:.4f}'
        for i in range(self.model.coef_.shape[0]):
            num = self.model.coef_[i]
            op = ' + ' if num > 0 else ' - '
            expr += f'{op}{round(abs(num), 4)}·x{i+1}'
        print('回归方程为：\n', expr)


if __name__ == '__main__':
    plt.style.use('matlab')
    data = pd.read_excel(r'resource\产品的销量与广告媒体的投入表.xlsx')
    linear_regress = LinearRegression()
    linear_regress.MultipleRegression(data.iloc[:, :-1], data.iloc[:, -1])
    plt.show()
