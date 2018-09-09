# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import random
from numpy import *
from sklearn.linear_model import LogisticRegression
"""
sigmoid函数
"""

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

"""
改进的梯度上升算发
"""


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # dataMatrix = array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights
"""

分类器函数
"""
def classsifyLogistic(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def testHorse():
    train_lines = open('/home/szu/PycharmProjects/machineLearning/Logistic/horseColicTraining').readlines()
    test_lines = open('/home/szu/PycharmProjects/machineLearning/Logistic/horseColicTest').readlines()
    trainSet = []
    train_labelsSet = []
    testSet=[]
    test_labelsSet=[]
    for line in train_lines:
        currantLine = line.strip().split('\t')  # \t 空格
        lineArr=np.array(currantLine[:-1]).astype(dtype='float')

        trainSet.append(lineArr)
        train_labelsSet.append(float(currantLine[-1]))
    # trainSet=trainSet[1:]
    # trainSet.dtype=float32
    # trainWeights = stocGradAscent1(np.array(trainSet), train_labelsSet, 500)
    # errorCount = 0
    # numTestLine = len(test_lines)

    for line in test_lines:
        currantLine = line.strip().split('\t')
        lineArr = np.array(currantLine[:-1]).astype(dtype='float')
        testSet.append(lineArr)
        test_labelsSet.append(float(currantLine[-1]))
        # testArr = currantLine[:-1]
        # #分类
        # if int(classsifyLogistic(np.array(lineArr), trainWeights)) != int(currantLine[-1]):
        #     errorCount += 1
    ##sklearn
    classifier=LogisticRegression(solver='saga',max_iter=5000).fit(trainSet,train_labelsSet)
    test_accurcy=classifier.score(testSet,test_labelsSet)*100
    # errorRate = (float(errorCount) / numTestLine) * 100
    print("测试集准确率：{:.2f}".format(test_accurcy))


if __name__ == "__main__":
    testHorse()
