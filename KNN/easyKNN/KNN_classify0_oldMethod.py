#-*-coding:utf-8-*-
import numpy as np
import operator

"""
KNN分类器算法
parameters：

    inX测试集输入
    dataSet：训练集
    labels：分类标签
    k：knn算法参数，选择距离最小的k个点
returns：
    sortedClassCount[0][0]:分类结果
"""
def classify0(inX,dataSet,labels,k):
    #numpy shape[0]返回行数
    dataSetSize=dataSet.shape[0]
    # tile函数：在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # np.argsort(x, axis=0) 按列排序 np.argsort(x, axis=1) 按行排序
    # np.argsort(x) 按升序排列 np.argsort(-x) 按降序排列
    # x[np.argsort(x)] #通过索引值排序后的数组
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]