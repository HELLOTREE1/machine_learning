# -*- coding: utf-8 -*-
"""
训练单层决策树
"""
import numpy as np
import matplotlib.pyplot as plt

def loadSimpleData():
    dataMat=np.matrix([
        [1., 2.1],
        [1.5, 1.6],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    classLabels=[1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat,classLabels

def showData(dataMat,labelMat):
    data_plus=[]
    data_minus=[]
    for i in range(len(dataMat)):
        if labelMat[i]==1:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np=np.array(data_plus).T
    data_minus_np=np.array(data_minus).T

    # data_plus_np.plot(kind="scatter",x="data_plus_np[0]",y="data_plus_np[1]",alpha=0.1)
    plt.scatter(data_plus_np[0],data_plus_np[1],alpha=0.5)
    plt.scatter(data_minus_np[0],data_minus_np[1],alpha=0.5)
    plt.show()

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    """
    单层决策树分类函数
    :param dataMatrix:
    :param dimen: 第dimen列 也是第几个特征
    :param threshVal: 阈值
    :param threshIneq: 标志
    :return:
    resultArr分类结果
    """
    resultArr=np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq=='less':
        resultArr[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        resultArr[dataMatrix[:,dimen]>threshVal]=1.0
    return resultArr


def buildStump(dataArr,classLabels,D):
    """
        找到数据集上的单层最佳决策树
    :param dataArr:
    :param classLabels:
    :param D: 样本权重
    :return:
    bestStump-最佳单层决策树信息
    minError 最小误差
    bestClassEst-最佳分类结果
    """
    dataMatrix=np.mat(dataArr)
    labelMatrix=np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError=float('inf')
    for i in range(n):
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=abs(rangeMax-rangeMin)/numSteps#计算步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['less','greater']:
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                err






if __name__=="__main__":
    dataArr,classLabels=loadSimpleData()
    showData(dataArr,classLabels)