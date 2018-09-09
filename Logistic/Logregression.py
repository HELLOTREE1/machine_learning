# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np

import matplotlib.font_manager as fm

font = fm.FontProperties(fname='/home/szu/program_font/msyh.ttf')

# plt.rcParams['font.sans-serif'] = ['Arial']
"""
函数说明:加载数据

"""


def loadDataSet():
    dataMat = []
    labelMat = []
    fileLines = open('/home/szu/PycharmProjects/machineLearning/Logistic/testSet').readlines()
    for line in fileLines:
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

"""
绘制数据
"""
def plotDataSet():
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    num = np.shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(num):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=0.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('Y')
    plt.legend(['正样本', '负样本'], loc='upper right', prop=font, fontsize=10)
    plt.grid()  # 添加网格
    plt.show()

def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr=np.array(dataMat)
    n=np.shape(dataMat)[0]
    xcord1=[]
    ycord1=[]
    xcord2=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=0.5)
    ax.scatter(xcord2,ycord2,s=20,c='green',alpha=0.5)

    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]

    ax.plot(x,y)
    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
梯度上升算法
parameter：
dataMatIn--数据集
dataLabels--数据标签
returns：
weights.getA()--求得权重数组（最有参数）
"""
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)										#转换成numpy的mat
    labelMat = np.mat(classLabels).T					#转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)											#返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01														#移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500														#最大迭代次数
    weights = np.ones((n,1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)								#梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.T * error
        weights_array = np.append(weights_array,weights)
    weights_array = weights_array.reshape(maxCycles,n)
    return weights.getA(),weights_array
"""
传统的梯度下降法 每次更新需要所有样本参与到计算中，算法复杂
随机梯度下降法：对于每一次更新参数，不必遍历所有的训练集合，仅仅使用了一个数据，来变换一个参数。
一次只用一个样本点 去更新回归系数（最有参数）
所以采用随机梯度上升算法
"""
import random
from numpy import *

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    dataMatrix=array(dataMatrix)
    m,n = np.shape(dataMatrix)												#返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)   													#参数初始化

    weights_array = np.array([])#存储每次更新的回归系数										#存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.001									#降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))			#随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))					#选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h 								#计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  	#更新回归系数
            weights_array = np.append(weights_array,weights,axis=0) 		#添加回归系数到数组中
            del(dataIndex[randIndex]) 										#删除已经使用的样本
    weights_array = weights_array.reshape(numIter*m,n) 						#改变维度
    # return weights
    return weights,weights_array

"""
绘制回归系数和迭代次数的关系
parameter：
weights_array1--
weights_array2

"""
def plotWeights(weights_array1,weights_array2):
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1', FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.show()
    savefig("/home/szu/PycharmProjects/machineLearning/Logistic/迭代次数于回归系数.jpg")
if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)
