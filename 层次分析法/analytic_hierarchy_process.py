# coding:utf-8

import numpy as np
import numpy.linalg as LA


class AHP:
    """ 层次分析法 """
    THEORETICAL_METHOD = 0
    SQUARE_ROOT_METHOD = 1
    SUM_PRODUCT_METHOD = 2

    def __init__(self, judgementMatrix):
        self.setJudgetmentMat(judgementMatrix)
        self.__getWeightVectorFunc_dict = {
            0: self.__theoreticalMethod,
            1: self.__squareRootMethod,
            2: self.__sumProductMethod}
        self.isConsistent = None  # type:bool
        self.weightVector = None  # type:np.ndarray
        self.maxEigenValue = 0    # type:np.float64
        self.CI = None            # type:int
        self.CR = None            # type:int

    def getWeightVector(self, method: int = 0) -> np.ndarray:
        """ 计算最大特征根和权重向量

        Parameters
        ----------
        method : 求取权重向量所用的方法，具体有::

            - AHP.THEORETICAL_METHOD : 解析解
            - AHP.SQUARE_ROOT_METHOD : 方根法
            - AHP.SUM_PRODUCT_METHOD : 和积法
        """

        if method < 0 or method > 2:
            raise Exception('求解方法错误')

        self.__getWeightVectorFunc_dict[method]()
        print('最大特征值：', self.maxEigenValue)
        print('归一化权重向量：', self.weightVector)

        # 一致性检验
        self.checkConsist()
        return self.weightVector

    def __theoreticalMethod(self):
        """ 理论法求解权重向量 """
        # 计算特征值和特征向量矩阵(每一列是一个特征向量)
        eigenValues, eigenVectors = LA.eig(self.judgementMatrix)
        # 最大特征值的索引
        index = np.argmax(eigenValues)
        # 最大特征值和归一化的权重向量
        self.maxEigenValue = eigenValues[index].real.round(4)
        self.weightVector = eigenVectors[:, index].real
        self.weightVector = (self.weightVector /
                             np.sum(self.weightVector)).round(4)

    def __squareRootMethod(self):
        """ 方根法求解权重向量 """
        # 计算判断矩阵每一行元素的积并取n次方根
        mat = self.judgementMatrix.prod(axis=1)  # type:np.ndarray
        mat = mat ** (1 / self.order)
        # 向量归一化得到权重向量
        self.weightVector = (mat / mat.sum()).round(4)  # type:np.ndarray
        # 计算最大特征根
        self.maxEigenValue = sum(
            (self.judgementMatrix @ self.weightVector) / self.weightVector) / self.order
        self.maxEigenValue = np.round(self.maxEigenValue, 4)

    def __sumProductMethod(self):
        """ 和积法求解特征向量 """
        # 对判断矩阵按列归一化再按行求和
        mat = self.judgementMatrix / self.judgementMatrix.sum(axis=0)
        mat = mat.sum(axis=1)
        # 向量归一化得到权重向量
        self.weightVector = (mat / mat.sum()).round(4)
        # 计算最大特征根
        self.maxEigenValue = sum(
            (self.judgementMatrix @ self.weightVector) / self.weightVector) / self.order
        self.maxEigenValue = np.round(self.maxEigenValue, 4)

    def checkConsist(self):
        """ 检查一致性 """
        self.CI = (self.maxEigenValue - self.order) / (self.order - 1)
        self.CR = self.CI / self.RI
        print(f'CR = {self.CR}')
        self.isConsistent = self.CR < 0.1
        if self.isConsistent:
            print('通过一致性检验')
        else:
            print('未能通过一致性检验')
        print('=='*30)

    def setJudgetmentMat(self, judgementMatrix):
        """ 设置判断矩阵 """
        self.judgementMatrix = np.array(judgementMatrix)  # type:np.ndarray
        self.order = len(self.judgementMatrix)  # type:int
        self.RI = [0, 0, 0.58, 0.90, 1.12, 1.24,
                   1.32, 1.41, 1.45][self.order]


if __name__ == "__main__":
    # 准则层判断矩阵
    A = np.array([[1, 1/2, 4, 3, 3],
                  [2, 1, 7, 5, 5],
                  [1/4, 1/7, 1, 1/2, 1/3],
                  [1/3, 1/5, 2, 1, 1],
                  [1 / 3, 1 / 5, 3, 1, 1]])

    ahp = AHP(A)
    weightVector_1 = ahp.getWeightVector(method=1)  # type:np.ndarray
    # 方案层判断矩阵
    B1 = [[1, 2, 5],
          [1/2, 1, 2],
          [1/5, 1/2, 1]]

    B2 = [[1, 1/3, 1/8],
          [3, 1, 1/3],
          [8, 3, 1]]

    B3 = [[1, 1, 3],
          [1, 1, 3],
          [1/3, 1/3, 1]]

    B4 = [[1, 3, 4],
          [1/3, 1, 1],
          [1 / 4, 1, 1]]

    B5 = [[1, 1, 1/4],
          [1, 1, 1/4],
          [4, 4, 1]]

    B_list = [B1, B2, B3, B4, B5]
    weightMat = np.zeros((3, len(B_list)))
    # 与准则无关的方案列表
    for i, B in enumerate(B_list):
        ahp.setJudgetmentMat(B)
        # 矩阵的每一列都是一个权重矩阵
        weightMat[:, i] = ahp.getWeightVector(method=1)
    # 层次总排序
    finalWeightVector = weightMat @ weightVector_1
    print(finalWeightVector)
