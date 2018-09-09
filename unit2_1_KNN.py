#-*-coding:utf-8-*-
from numpy import *
import operator#操作符
# import iteritems
def createDataSet():
    group=array([1.0,1,1],[1.0,1.0],[0,0],[0,0.1])
    labels=['A','A','B','B']
    return group,labels

#实际计算
#计算训练集中点与当前点的距离
#按照距离递增次序排序
#选取与当前点距离最小的k个点
#确定前k个点所在类别的出现频率
#返回前k个点出现频率最高的类别作为当前点的预测分类

def classify0(inputX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]#读取数据第一维的长度
    distances=(((tile(inputX,(dataSetSize,1))-dataSet)**2).sum(axis=1))**0.5
    sortedDistIndicies=distances.argsort()
    #排序函数 sort函数是list列表中的函数，而sorted可以对list或者iterator进行排序
    # sort函数排序时会影响列表本身，sorted不会
    # argsort函数返回的是数组值从小到大的索引值
    # np.argsort(x, axis=0) 按列排序 np.argsort(x, axis=1) 按行排序
    # np.argsort(x) 按升序排列 np.argsort(-x) 按降序排列
    # x[np.argsort(x)] #通过索引值排序后的数组
    # sorted(iterable，cmp，key，reverse）
    # 参数：iterable可以是list或者iterator；
    # cmp是带两个参数的比较函数；
    # key
    # 是带一个参数的函数；
    # reverse为False或者True；

    classCount={}                               #选择距离最小的k个点
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]

        # dict.get(key, default=None) 通过get获取键值对的key
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1

    sortedClassCount=sorted(classCount,iteritems(),key=operator.itemgetter(1),reverse=True)#大到小的排序
    return sortedClassCount[0][0]
#每个样本数据占一行，1000行，3中特征
def file2materix(filename):
    fr=open(filename)
    fileLines=fr.readlines()
    numLines=len(fileLines)
    returnMat=zeros((numLines,3))#特征3种
    classLabelVector=[]
    index=0
    for line in fileLines:          #解析文件数据到列表
        line=line.strip()#str.strip([chars]);移除字符串头尾指定的字符charts，默认空格
        listFromLines=line.split('\t')
        returnMat[index,:]=listFromLines[0:3]
        classLabelVector.append(int(listFromLines[-1]))
        index += 1
    return returnMat,classLabelVector
