# -*- coding: utf-8 -*-
from math import log
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

"""
函数说明：创建测试集
年龄：0代表青年，1代表中年，2代表老年；
有工作：0代表否，1代表是；
有自己的房子：0代表否，1代表是；
信贷情况：0代表一般，1代表好，2代表非常好；
类别(是否给贷款)：no代表否，yes代表是。

"""


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet, labels


"""
函数说明：计算香农熵
parameter：dataSet
returns：shannonEnt--香农熵
看最终结果被分为几类
H=-n个求和(p(xi)*log2p(xi))
有数据得到的：是香农经验熵
"""


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}  # 保存每个标签出现的次数的字典
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


"""
函数说明：选择各个特征的子集
parameters：
    dataSet
    axis--划分数据集的特征
    value--返回特征的值
假如axis=0，value=1，则返回

[1, 0, 0, 0, 'no'],
[1, 0, 0, 1, 'no'],
[1, 1, 1, 1, 'yes'],
[1, 0, 1, 2, 'yes'],
[1, 0, 1, 2, 'yes'],

"""


def splitDataSet(dataSet, axis, value):
    returnDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            returnDataSet.append(featVec)
    return returnDataSet


"""

函数说明：选择最优特征
parameters：dataSet
returns：bestFeature--信息增益最大的特征索引值
"""


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量(最后一个是类别，不是特征)
    baseEnt = calcShannonEnt(dataSet)
    bestInfoGain = 0.0  # 初始化信息增益
    bestFeature = -1  # 最佳特征的索引
    for i in range(numFeatures):
        # 获取dataSet第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set{}集合，元素不可重复
        newEnt = 0.0  # 初始化经验条件熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 打印每个特征的信息增益
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEnt += prob * calcShannonEnt(subDataSet)  # 计算经验条件熵
        infoGain = baseEnt - newEnt
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


"""
函数说明;统计classList中出现次数最多的类标签
投票表决
parameters：classList--类标签列表
returns：sortedClassCount[0][0]

"""


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 根据字典值降序排序
    return sortedClassCount[0][0]


"""
函数说明：创建决策树
parameters：
    dataSet-训练数据集
    labels-分类标签
    featLabels-存储的最优特征标签
returns：
    myTree-决策树
    
递归创建决策树时，递归有两个终止条件：
第一个停止条件是所有的类标签完全相同，则直接返回该类标签；
第二个停止条件是使用完了所有特征，仍然不能将数据划分仅包含唯一类别的分组，即决策树构建失败，特征不够用。
此时说明数据纬度不够，由于第二个停止条件无法简单地返回唯一的类标签，这里挑选出现数量最多的类别作为返回值。   

"""


def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]  # 取出分类标签
    if classList.count(classList[0]) == len(classList):  # 如果划分类别完成，则结束.第一个停止条件
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  ##遍历完所有特征时返回出现次数最多的类标签。第二个停止条件
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 最优特征
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    # print(dataSet)
    # print(calcShannonEnt(dataSet))
    # print("最佳特征索引值：" + str(chooseBestFeatureToSplit(dataSet)))
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
