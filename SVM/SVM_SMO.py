# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import time


# 首先构建一个仅包含init方法的optStruct类，将其作为一个数据结构来使用，方便我们对于重要数据的维护
class optStruct:
    """
    数据结构，维护所有需要操作的值
    Parameters：
        dataMatIn - 数据矩阵
        classLabels - 数据标签
        C - 松弛变量
        toler - 容错率

    """

    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.eCache = np.mat(np.zeros((self.m, 2)))


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    Lines = open(filename).readlines()
    for line in Lines:
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def calcEk(oS, k):
    """

    计算误差
    :param oS: 数据结构
    :param k: 标记为k的数据
    :return: Ek
    """
    fxk = float((np.multiply(oS.alphas, oS.labelMat)).T * (oS.X * oS.X[k, :].T) + oS.b)
    Ek = fxk - float(oS.labelMat[k])
    return Ek


def selectJrand(i, m):
    """
    挑选alpha_j的索引值
    :param i:
    :param m:
    :return:
    """
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def selectJ(i, oS, Ei):
    """

    :param i:
    :param oS:
    :param Ei:
    :return:
    j,maxk
    Ej
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 更新误差缓存
    # 返回误差不为0的数据的索引值
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:  # 如果有不为0的误差
        for k in validEcacheList:  # 遍历，找到最大的Ek
            if k == i:  # 不计算i
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
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
    修剪aj
    :param aj:
    :param H:
    :param L:
    :return:
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
    1-有任意一对alpha值发生变化
    0-没有任意一对alpha值发生变化或者变化太小
    """
    # 1 计算误差Ei
    Ei = calcEk(oS, i)
    # 优化alpha 设置一定的容错率
    if ((oS.labelMat[i] * Ei) < -oS.tol) and (oS.alphas[i] < oS.tol) or (oS.labelMat[i] * Ei > oS.tol) and (
            oS.alphas[i] > 0):
        # 启用内循环 更新aj 计算Ej
        j, Ej = selectJ(i, oS, Ei)
        alphaI_old = oS.alphas[i].copy()
        alphaJ_old = oS.alphas[j].copy()
        # 2 计算上下边界LH
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0  # ?
        # 3 eta
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        # 4 alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 5 修剪alpha_j
        oS.alphas[j] = clippedAlpha(oS.alphas[j], H, L)
        # 更新Ej到缓存误差
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJ_old) < 0.00001):
            print("alphaJ变化太小")
            return 0
        # 6 alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJ_old - oS.alphas[j])
        #  Ei
        updateEk(oS, i)
        # 7 b1 b2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaI_old) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJ_old) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaI_old) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                oS.alphas[j] - alphaJ_old) * oS.X[j, :] * oS.X[j, :].T
        # 8 b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter):
    """
    完整的线性SMO算法
    :param dataMatIn:
    :param classLabels:
    :param C:
    :param toler:
    :param maxIter:
    :return:
    oS.b - SMO算法计算的b
    oS.alphas - SMO算法计算的alphas
    """
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).T, C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0  # 优化次数
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("全样本遍历：第{:d}次迭代 样本{:d} alpha优化次数{:d}".format(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("非边界样本遍历：第{:d}次迭代 样本{:d} alpha优化次数{:d}".format(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("迭代次数{:d}".format(iter))
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def showClassifer(dataMat, classLabels, w, b):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus).T
    data_minus_np = np.array(data_minus).T
    plt.scatter(data_plus_np[0], data_plus_np[1], s=30, alpha=0.5)
    plt.scatter(data_minus_np[0], data_minus_np[1], s=30, alpha=0.5)

    x1 = max(dataMat)[0]
    x2 = min(dataMat)[1]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if alpha > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


if __name__ == "__main__":
    start = time.clock()
    dataArr, classLabels = loadDataSet("/home/szu/PycharmProjects/machineLearning/SVM/textSet")
    b, alphas = smoP(dataArr, classLabels, 0.6, 0.001, 50)
    w = calcWs(alphas, dataArr, classLabels)
    end = time.clock()
    showClassifer(dataArr, classLabels, w, b)
    # 中间写上代码块

    print('Running time: %s Seconds' % (end - start))
