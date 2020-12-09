# coding:utf-8

from typing import List

from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


class VarianceAnalysis:
    """ 方差分析 """

    def __init__(self):
        self.alpha = 0.05
        self.is_accepet_H0 = False

    def oneWayANOVA(self, *args, axis=0, alpha=0.05):
        """ 单因素方差分析

        Parameters
        ----------
        *args : 多个 `~np.ndarray`
            每组样本的测量值，且方差必须相等

        axis : int
            对数组进行方差分析的方向

        alpha : float
            显著值
        """
        self.alpha = alpha
        self.F, self.p_ = stats.f_oneway(*args, axis=axis)
        # 当p_值大于 α 时接受原假设，即认为样本间的均值无显著差异
        self.is_accepet_H0 = self.p_ > self.alpha
        # 打印消息
        print('单因素方差分析结果：')
        for i in np.array(self.is_accepet_H0, ndmin=1):
            print(f'{"接受" if i else "拒绝"}原假设H0')

    def multiFactorANOVA(self, dependent_arg, *independent_args, dependent_arg_name: str,
                         independent_arg_names: List[str], alpha=0.05):
        """多因素方差分析

        Parameters
        ----------
        dependent_arg : 1-D array-like
            因变量测量值

        *independent_args : 多个 1-D array_like
            自变量测量值，要求每组的长度相等且一一对应形成一个n维向量

        dependent_arg_name : str
            因变量名字

        independent_arg_names : List[str]
            自变量名字列表

        alpha : float, optional
            显著值, by default 0.05
        """
        data = dict(zip(independent_arg_names, independent_args))
        data[dependent_arg_name] = dependent_arg
        self.df = pd.DataFrame(data)
        model_result = ols(
            f'{dependent_arg_name}~{"+".join(independent_arg_names)}\
             +{":".join(independent_arg_names)}', self.df).fit()
        self.anova_df = anova_lm(model_result)
        self.anova_df.rename(columns={'PR(>F)': 'p_'}, inplace=True)
        print('多因素方差分析结果：\n', self.anova_df)


if __name__ == "__main__":
    va = VarianceAnalysis()

    # 单因素方差分析
    a = stats.norm.rvs(loc=2, scale=2, size=200)
    b = stats.norm.rvs(loc=2.31, scale=2, size=200)
    c = stats.norm.rvs(loc=2.05, scale=2, size=100)
    va.oneWayANOVA(a, b, c)

    # 多因素方差分析
    A = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    B = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    Y = [58.2, 56.2, 65.3, 49.1, 54.1, 51.6,
         60.1, 70.9, 39.2, 75.8, 58.2, 48.7]
    va.multiFactorANOVA(Y, A, B, dependent_arg_name='Y',
                        independent_arg_names=['A', 'B'])
