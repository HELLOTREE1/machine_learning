# -*- coding: utf-8 -*-
import numpy as np



"""
函数说明:创建实验样本

"""
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],                #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec
"""
创建词汇表
"""
def createVocabList(dataSet):
    vocabSet=set([])#创建一个空的不重复列表
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)
"""
根据vocablist 将inputSet向量化 向量的每个元素为1或0
Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词集模型
"""
def setOfWord2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word %s is not in my vocablist"% word)
    return returnVec

"""
朴素贝叶斯训练函数
parameters:
 trainMatrix--训练文档矩阵，文档向量
 trainCategory--训练文档标签，classVec
returns：
 p0Vect - 侮辱类的条件概率数组
    p1Vect - 非侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率

"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)#训练文档数目
    numWords=len(trainMatrix[0])#文档的词条数目
    #文档属于侮辱类的概率
    PAbusive=sum(trainCategory)/float(numTrainDocs)
    #分别统计文当中侮辱类词条 和 费侮辱类词条
    p0num=np.ones(numWords);p1num=np.ones(numWords)
    #分母
    p0denom=2.0;p1denom=2.0
    for i in range(numTrainDocs):
        print("trainMatrix[i]:\n",trainMatrix[i])
        if trainCategory[i]==1:
            p1num+=trainMatrix[i]
            print("P1num",p1num)
            p1denom+=sum(trainMatrix[i])
        else:
            p0num+=trainMatrix[i]
            p0denom+=sum(trainMatrix[i])
    p0Vect=p0num/p0denom
    p1Vect=p1num/p1denom
    return p0Vect,p1Vect,PAbusive

"""
分类
Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 侮辱类的条件概率数组
    p1Vec -非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
"""
from functools import reduce
def classifNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1-pClass1)
    print("p0",p0)
    print("p1",p1)
    if p1>p0:
        return 1
    else:
        return 0
"""
测试函数
"""
def testNB0():
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    print("myVocabList:\n", myVocabList)

    trainMAt = []
    for postingDoc in postingList:
        trainMAt.append(setOfWord2Vec(myVocabList, postingDoc))

    P0V, P1V, PAV = trainNB0(trainMAt, classVec)

    testDoc=['love','my','dalmation','fool','stupid']
    thisDoc=np.array(setOfWord2Vec(myVocabList,testDoc))
    if classifNB(thisDoc,P0V,P1V,PAV):
        print("侮辱",testDoc)
    else:
        print("nono",testDoc)

if __name__ == '__main__':
    testNB0()