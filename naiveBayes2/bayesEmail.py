# -*- coding: utf-8 -*-
import numpy as np
import random
import re

def textParse(bigString): #将字符串转换为字符列表
    listOfTokens=re.split('\w+',bigString) #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) >2]  #除了单个字母，其它单词变成小写

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)

    return list(vocabSet)
"""
建立词汇表
"""
def setOfWord2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("%s not in my vocablist" % word)
    return returnVec
"""
朴素贝叶斯 训练器
"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    #垃圾文档概率
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p1Num=np.ones(numWords);p0Num=np.ones(numWords)
    p0Denom=2.0;p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=np.log(p1Num/p1Denom)
    p0Vect=np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

"""
分类器
"""
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0



def testBayesEmail():
    docList=[];classList=[]
    for i in range(1,26):
        wordList=textParse(open('/home/szu/PycharmProjects/machineLearning/naiveBayes2/email/spam/%d.txt' % i, 'rb').read().decode(
                'GBK'))
        docList.append(wordList)
        classList.append(1)
        wordList1 = textParse(open('/home/szu/PycharmProjects/machineLearning/naiveBayes2/email/ham/%d.txt' % i, 'rb').read().decode(
                'GBK')) # 读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList1)
        classList.append(0)  # 标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)  # 创建词汇表，不重复
#创建存储训练集的索引值的列表和测试集索引值
    trainingSet=list(range(50));testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])#加测试集
        del(trainingSet[randIndex])#在训练集中删除测试集索引

    trainMat=[];trainClass=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClass))

    erroCount=0
    for docIndex in testSet:
        wordVector=setOfWord2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            erroCount+=1
            print("错误的测试集：",docList[docIndex])
    print("错误率：{:.2f}".format(float(erroCount)/len(testSet)*100))

if __name__=="__main__":
    testBayesEmail()