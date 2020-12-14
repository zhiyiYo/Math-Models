# coding:utf-8
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class PCA:
    """ 主成分分析法 """

    def __init__(self, excel_path: str, inverse_indexes: List[str] = None):
        """ 创建主成分分析法对象

        Parameters
        ----------
        excel_path : str
            原始数据 excel 表格路径, 表格的第一列作为行索引

        inverse_indexes : List[str]
            逆指标列表, 每个元素都是逆指标的名称
        """
        # 读入第一个表格中的数据
        self.origin_data = pd.read_excel(
            excel_path, index_col=0)  # type:pd.DataFrame
        self.n_samples = self.origin_data.shape[0]      # 样本数
        self.n_features = self.origin_data.shape[1]     # 原始特征数
        self.inverse_indexes = inverse_indexes
        # Z-score标准化数据
        self.standard_data = self.origin_data.copy()  # type:pd.DataFrame
        if inverse_indexes:
            self.standard_data[inverse_indexes] *= - 1
        self.standard_data = (
            self.origin_data - self.origin_data.mean()) / self.origin_data.std()

    def getPrincipalComponent(self, percentage=0.85):
        """ 计算主成分

        Parameters
        ----------
        percentage : float
            确定所取的主成分个数 m 的累计主成分贡献度阈值

        Returns
        -------
        F_score_mat : 2-D `~np.ndarray`
            形状为 n×m 的主成分决策(得分)矩阵，n为原始数据的行数

        F : 1-D `~np.ndarray`
            综合成分向量
        """
        # 相关系数矩阵
        self.correlation_df = self.standard_data.T @ self.standard_data / \
            (self.n_samples - 1)  # type:pd.DataFrame
        # n_features个特征值和形状为 (n_features, n_features) 的特征向量矩阵
        self.eig_values, self.eig_vector_mat = np.linalg.eig(
            self.correlation_df)

        # 对特征值从小到大进行排序
        slice_index = np.argsort(self.eig_values)[::-1]
        self.eig_values = self.eig_values[slice_index]
        self.eig_vector_mat = self.eig_vector_mat[:, slice_index]

        # 选出满足阈值条件的前m个特征值和特征向量
        self.contribution = self.eig_values / self.eig_values.sum()     # 主成分贡献度
        self.cum_contribution = self.contribution.cumsum()              # 累计贡献度
        self.prin_comp_num = m = np.where(                              # 主成分个数
            self.cum_contribution > percentage)[0][0] + 1
        # 前m个特征值
        self.feature_values = self.eig_values[:m]
        # 前m个特征向量，矩阵形状为 (n_features, m)
        self.feature_vector_mat = self.eig_vector_mat[:, :m]  # type:np.ndarray

        # 创建成分统计表格
        data = {'特征值': self.eig_values,
                '方差贡献度(%)': self.contribution*100,
                '累计贡献度(%)': self.cum_contribution*100}
        self.component_df = pd.DataFrame(
            data, range(1, len(self.eig_values) + 1))
        self.component_df.index.name = '成分'

        # 载荷矩阵
        self.prin_comp_load_mat = np.sqrt(
            self.feature_values) * self.feature_vector_mat
        # 主成分决策矩阵(主成分得分矩阵), 每一列代表一种主成分的值
        self.F_score_mat = self.standard_data.values @ self.feature_vector_mat
        # 因子得分矩阵
        self.factor_score_mat = self.F_score_mat / \
            np.sqrt(self.feature_values)
        # 将 m 维主成分加权平均成一维综合成分
        self.F = self.F_score_mat @ self.contribution[:m]

        # 根据综合成分进行排名
        self.F_score_df = pd.DataFrame()
        for i in range(m):
            # self.standard_data[f'因子Factor{i+1}得分'] = self.factor_score_mat[:, i]
            self.F_score_df[f'主成分F{i+1}得分'] = self.F_score_mat[:, i]
        self.F_score_df['综合得分'] = self.F
        ranking = self.F.argsort()[::-1].argsort() + 1
        self.F_score_df['排名'] = ranking
        return self.F_score_mat, self.F

    def analysis(self, percentage=0.85, excel_path: str = '', is_plot: bool = True):
        """ 分析主成分分析结果

        Parameters
        ----------
        percentage : float
            确定所取的主成分个数 m 的累计主成分贡献度阈值

        excel_path : str
            写入分析结果的excel文件路径，如果为空则不写入excel

        is_plot : bool
            是否绘制碎石图
        """
        # 主成分分析
        self.getPrincipalComponent(percentage)
        # 生成主成分表达式
        print('各个主成分表达式：')
        for i in range(self.prin_comp_num):
            expr = f'F{i+1}='
            for j in range(self.n_features):
                num = self.feature_vector_mat[j, i]
                op = ' + ' if num > 0 else ' - '
                op = '   ' if j == 0 and num > 0 else op
                expr += f'{op}{round(abs(num), 4)}·ZX{j+1}'
            print(expr)

        # 分析结果写入excel
        if excel_path:
            with pd.ExcelWriter(excel_path) as f:
                self.origin_data.to_excel(f, '原始数据')
                self.standard_data.to_excel(f, '标准化数据')
                self.F_score_df.sort_values('排名').to_excel(f, '主成分得分及排名')
                self.correlation_df.to_excel(f, '相关系数')
                self.component_df.to_excel(f, '总方差解释')
                prin_comp_load_df = pd.DataFrame(
                    self.prin_comp_load_mat, self.origin_data.columns,
                    [f'主成分F{i}' for i in range(1, self.prin_comp_num+1)])
                prin_comp_load_df.to_excel(f, '成分(因子载荷)矩阵')
                feature_vector_df = pd.DataFrame(self.eig_vector_mat, [
                                                 f'ZX{i}' for i in range(1, self.n_features + 1)],
                                                 [f'F{i}' for i in range(1, self.n_features + 1)])
                feature_vector_df.to_excel(f, '特征向量(成分系数)表')

        # 绘制碎石图
        if is_plot:
            # 隐藏副刻度线
            plt.rcParams['xtick.minor.visible'] = False
            fig, ax = plt.subplots(num='主成分分析')
            n = np.arange(1, len(self.eig_values) + 1, dtype='int')
            ax.plot(n, self.eig_values, marker='o')
            plt.xticks(n, [str(i) for i in n])
            ax.set(xlabel=r'$Component$', ylabel=r'$Eigen\ Value$',
                   title=r'$Scree\ Plot$')

    def linearRegression(self, y) -> LinearRegression:
        """ 主成分回归

        Parameters
        ----------
        y : 1-D array_like
            长度为样本数的向量
        """
        y = np.array(y).reshape(-1, 1)
        self.model = LinearRegression()
        # 先对主成分进行回归
        self.model.fit(self.F_score_mat, y)
        # 转换为标准化数据的回归系数，形状为 (1, n_features)
        coeff = self.model.coef_ @ self.feature_vector_mat.T
        # 将标准化变量的回归方程还原为原始变量的回归方程
        mean = self.origin_data.mean().values
        std = self.origin_data.std().values
        self.model.intercept_ = (self.model.intercept_ - coeff.dot(mean/std))[0]
        self.model.coef_ = (coeff / std).flatten()
        # 打印回归方程表达式
        expr = f'y = {self.model.intercept_:.4f}'
        for i in range(self.model.coef_.shape[0]):
            num = self.model.coef_[i]
            op = ' + ' if num > 0 else ' - '
            expr += f'{op}{round(abs(num), 4)}·x{i+1}'
        print('回归方程为：\n', expr)
        return self.model


if __name__ == '__main__':
    plt.style.use('matlab')
    pca = PCA('resource\\Hald水泥数据.xlsx')
    pca.analysis(0.98, 'resource\\分析结果.xlsx')
    y = [78.5, 74.3, 104.3, 87.6, 95.9, 109.2,
         102.7, 72.5, 93.1, 115.9, 83.8, 113.3, 109.4]
    pca.linearRegression(y)
    # plt.show()
