# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    Lines = open(filename).readlines()
    for line in Lines:
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def showDataSet(dataMat, labelMat):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] == 1:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus).T
    data_minus_np = np.array(data_minus).T

    # data_plus_np.plot(kind="scatter",x="data_plus_np[0]",y="data_plus_np[1]",alpha=0.1)
    plt.scatter(data_plus_np[0], data_plus_np[1], alpha=0.5)
    plt.scatter(data_minus_np[0], data_minus_np[1], alpha=0.5)
    plt.show()


class optStruct:
    """
    数据结构 维护需要操作的数据之
    Parameters：
		dataMatIn - 数据矩阵
		classLabels - 数据标签
		C - 松弛变量
		toler - 容错率
		kTup - 包含核函数信息的元组,第一个参数存放核函数类别，第二个参数存放必要的核函数需要用到的参数
    """

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def kernelTrans(X, A, kTup):
    """
    核函数 将数据转换为更高维度的空间
    :param X: 数据矩阵
    :param A: 单个数据的向量
    :param kTup: 包含核函数信息的元组
    :return: K 计算的核
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))  # 初始化列向量
    ##根据键值选择相应核函数
    # lin 线性核函数
    # rbf 径向基核函数
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-2 * kTup[1] ** 2))
    else:
        raise NameError('核函数无法识别')
    return K


def calcEk(oS, k):
    """
    计算误差
    :param oS: 数据结构
    :param k: 标记为k的数据
    :return: Ek 标记为k的数据误差
    """
    fxk = float((np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b))
    Ek = fxk - float(oS.labelMat[k])
    return Ek


def selectjrand(i, m):
    """
    随机选择alpha_j的索引值
    :param i: alpha_i的索引值
    :param m: alpha参数个数
    :return: j：alpha_j的索引
    """
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def selectJ(i, oS, Ei):
    """
    内循环启发式2
    :param i:数据i 的索引
    :param oS:数据结构
    :param Ei:
    :return:
    j，maxK 数据j maxK的索引
    Ej 标号为j的数据误差
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    # 返回误差不为0的数据的索引值
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:  # 若存在不是0 的误差
        for k in validEcacheList:
            if k == i:  # 不计算i 节省时间
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE - maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectjrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    """
    计算Ek 更新误差缓存
    :param oS:
    :param k:
    :return:
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def clippedAlpha(aj, H, L):
    """
    修 alpha_j
    :param aj:
    :param H:
    :param L:
    :return:aj
    """
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


def innerL(i, oS):
    """
    优化的SMO算法
    :param i:
    :param oS:
    :return:
    1 --有任意一对alpha值发生变化
    0  没有任意一对alpha值发生变化或变化太小
    """
    # 1  计算Ei
    Ei = calcEk(oS, i)
    # 优化alpha 设置一定的容错
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 内循环启发式2选择Aj
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的alpha数值 采用深层复制
        alphaI_old = oS.alphas[i].copy()
        alphaJ_old = oS.alphas[j].copy()
        # 2 计算上下边界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 3 计算eta
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
            # 步骤4：更新alpha_j
            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            # 步骤5：修剪alpha_j
            oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
            # 更新Ej至误差缓存
            updateEk(oS, j)
            if (abs(oS.alphas[j] - alphaJold) < 0.00001):
                print("alpha_j变化太小")
                return 0
            # 步骤6：更新alpha_i
            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
            # 更新Ei至误差缓存
            updateEk(oS, i)
            # 步骤7：更新b_1和b_2
            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[i, j]
            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[j, j]
            # 步骤8：根据b_1和b_2更新b
            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
                oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin',0)):
    """
    完整的线性SMO算法
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :param kTup: 包含核函数信息的元组
    :return:
    oS.b
    oS.alphas
    """
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).T,C,toler,kTup)
    iter=0
    entireSet=True
    alphaPairsChanged=0
    while(iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
                print("全样本遍历 第{:d}次迭代 样本{:d} alpha优化次数{:d}".format(iter,i,alphaPairsChanged))
            iter+=1
        else:
            nonBoundIs=np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print("非边界样本遍历 第{:d}次迭代 样本{:d} alpha优化次数{:d}".format(iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet=False
        elif (alphaPairsChanged==0):
            entireSet=True
        print("迭代次数{:d}".format(iter))
    return oS.b,oS.alphas

def testRBF(k1=1.3):
    """

    :param k1: 使用高斯核函数 表示到达率 下降至0的速度参数
    :return:
    """
    dataArr,labelArr=loadDataSet("/home/szu/PycharmProjects/machineLearning/SVM/textRBF")
    b,alphas=smoP(dataArr,labelArr,200,0.0001,100,('rbf',k1))
    datMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).T
    svmInd=np.nonzero(alphas.A>0)[0]
    SVMs=datMat[svmInd]
    labelSWV=labelMat[svmInd]
    print("支持向量个数{:d}".format(np.shape(SVMs)[0]))




if __name__ == "__main__":
    dataArr, labelArr = loadDataSet('/home/szu/PycharmProjects/machineLearning/SVM/textRBF')  # 加载训练集
    showDataSet(dataArr, labelArr)
