# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from math import factorial

INF = float('inf')


class QueuingModel:
    """ 排队模型 """

    MM1 = 0
    MMS = 1
    MM1N = 2
    MMSN = 3

    def __init__(self, lamda, mu, s: int = 1, N=INF, model_type: int = 0):
        """创建排队论模型

        Parameters
        ----------
        lamda : float
            单位时间平均到达人数

        mu : float
            单位时间服务完成人数

        s : int
            服务台数量

        N : int
            系统的最大容量

        model_type : int
            模型类型, 有以下几种::

                - QueuingModel.MM1   : M/M/1/∞ 模型
                - QueuingModel.MMS   : M/M/s/∞ 模型
                - QQueuingModel.MM1N : M/M/1/N 模型
                - QQueuingModel.MMSN : M/M/s/N 模型

        """
        if not 0 <= model_type <= 3:
            raise Exception('模型类型错误')
        self.__getQueuingInfo_func = {
            0: self.__getMM1QueuingInfo,
            1: self.__getMMSQueuingInfo,
            3: self.__getMMSNQueuingInfo}
        self.model_type = model_type
        self.lamda = lamda
        self.mu = mu
        self.s = s
        self.N = N

    def getQueuingInfo(self) -> dict:
        """ 计算排队模型的信息 """
        info = self.__getQueuingInfo_func[self.model_type]()
        return info

    def __getMM1QueuingInfo(self) -> dict:
        """ 计算 M/M/1/∞ 排队模型信息 """
        if self.lamda >= self.mu:
            raise Exception('M/M/1/∞ 模型下λ必须小于μ')
        pho = self.lamda / self.mu  # 服务强度
        # 系统人数为空的概率
        p0 = 1 - pho
        # 顾客的等待概率
        c = 1 - p0
        # 稳定情况下系统人数为n的概率
        def pn(n): return (1 - pho) * pho
        # 平均队长（包括服务人数）
        Ls = self.lamda / (self.mu - self.lamda)
        # 平均排队长
        Lq = self.lamda ** 2 / (self.mu * (self.mu - self.lamda))
        # 平均逗留时间
        Ws = Ls / self.lamda
        # 平均排队时间
        Wq = Lq / self.lamda
        self.queuing_info = {'phos': pho, 'c': c, 'p0': p0, 'pn': pn,
                             'Ls': Ls, 'Lq': Lq, 'Ws': Ws, 'Wq': Wq}
        self.printQueuingInfo(self.queuing_info)
        return self.queuing_info

    def __getMMSQueuingInfo(self):
        """ 计算M/M/s/∞ 排队模型信息 """
        s = self.s
        pho = self.lamda / self.mu
        phos = pho / s
        if not 0 < phos < 1:
            raise Exception('ρs必须小于1')

        p0 = 1 / (sum(pho ** n / factorial(n)
                      for n in range(s)) + pho ** s / (factorial(s) * (1 - phos)))
        # 概率分布函数

        def pn(n): return pho**n*p0 / \
            factorial(n) if n <= s else pho ** n * \
            p0 / (factorial(s) * s ** (n - s))
        # 等待概率
        c = pho ** s * p0 / (factorial(s) * (1 - phos))
        Lq = p0 * phos * pho ** s / (factorial(s) * (1 - phos) ** 2)
        Ls = Lq + pho
        Wq = Lq / self.lamda
        Ws = Ls / self.lamda
        self.queuing_info = {'phos': phos, 'c': c, 'p0': p0, 'pn': pn,
                             'Ls': Ls, 'Lq': Lq, 'Ws': Ws, 'Wq': Wq}
        self.printQueuingInfo(self.queuing_info)
        return self.queuing_info

    def __getMMSNQueuingInfo(self):
        """ 计算 M/M/s/N 排队模型信息 """
        if self.N == INF:
            raise Exception('M/M/s/N模型下系统最大容量N不能为∞')
        s = self.s
        pho = self.lamda / self.mu
        phos = pho / s
        c = 0
        p0 = 1/(sum(pho**n/factorial(n)
                    for n in range(0, s)) + pho ** s / (factorial(s) * (1 - phos)))

        def pn(n): return pho**n*p0 / \
            factorial(n) if n <= self.s else pho ** n * \
            p0 / (factorial(s) * s ** (n - s))

        Lq = p0 * phos * (s * phos) ** s * (1 - s - (self.N - s + 1) * phos **
                                            self.N - s * (1 - phos)) / (factorial(s) * (1 - phos) ** 2)
        Ls = Lq + s * phos * (1 - pn(self.N))
        Wq = Lq / (self.lamda * (1 - pn(self.N)))
        Ws = Wq + 1 / self.mu
        self.queuing_info = {'phos': phos, 'c': c, 'p0': p0, 'pn': pn,
                             'Ls': Ls, 'Lq': Lq, 'Ws': Ws, 'Wq': Wq}
        self.printQueuingInfo(self.queuing_info)
        return self.queuing_info

    def printQueuingInfo(self, info: dict):
        """ 打印排队信息 """
        print(
            f"{'平均队长：':<11}{info['Ls']:.2f}\n"
            f"{'平均排队长：':<10}{info['Lq']:.2f}\n"
            f"{'平均等待时间：':<9}{info['Wq']:.2f}\n"
            f"{'平均逗留时间：':<9}{info['Ws']:.2f}\n"
            f"系统空闲的概率：{info['p0']:.3f}\n"
            f"顾客等待的概率：{info['c']:.3f}"
        )


if __name__ == "__main__":
    queuing_model = QueuingModel(6, 0.5, 8, 8, QueuingModel.MMSN)
    queuing_model.getQueuingInfo()
