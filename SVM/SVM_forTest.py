# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
import time

"""
读取数据
"""
def loadDataSet(filename):
    dataMat=[]
    labelMat=[]
    Lines=open(filename).readlines()
    for line in Lines:
        lineArr=line.strip().split("\t")
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat
# 数据可视化

import matplotlib.font_manager as fm
font = fm.FontProperties(fname='/home/szu/program_font/msyh.ttf')
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
    plt.savefig('showData.png')

# 选择合适的ij
def selectJ(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

# 修剪alpha
def clippedAlpha(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

"""
SMO算法
parameters:
dataMat
labelMat
C-松弛变量
toler-容错率
maxInter-最大迭代次数
"""
def smoSimple(dataMat,labelMat,C,toler,maxInter):
    dataMatrix=np.mat(dataMat)
    labelMatrix=np.mat(labelMat).T
    #初始化b
    b=0
    #统计dataMatrix的维度
    m,n=np.shape(dataMatrix)
    #初始化alpha
    alphas=np.mat(np.zeros((m,1)))
    #初始化迭代次数
    iter_num=0
    while(iter_num<maxInter):
        alphaPairsChanged = 0  # flag：记录alpha是否被优化
        for i in range(m):

            #1 计算Ei
            fxi=float((np.multiply(alphas,labelMatrix)).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei=fxi-float(labelMatrix[i])
            # 优化alpha，更设定一定的容错率。
            # 若误差较大，且alpha没有到边界
            if ((labelMatrix[i]*Ei <-toler) and (alphas[i]<C)) or ((labelMatrix[i]*Ei >toler) and (alphas[i]>0)):
                #随机选择j
                j=selectJ(i,m)
                #1  计算Ej
                # fxj=float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                fxj = float((np.multiply(alphas, labelMatrix)).T * (dataMatrix * dataMatrix[j, :].T)) + b

                Ej=fxj-float(labelMatrix[j])
                #保存更新前的aplpha值，使用深拷贝
                alphaI_old=alphas[i].copy()
                alphaJ_old=alphas[j].copy()
                #2  计算上下界
                if (labelMatrix[i]!=labelMatrix[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[j]+alphas[i]-C)
                    H=min(C,alphas[j]+alphas[i])
                if L==H:
                    print("L==H")
                    continue
                #3 计算学习率(和笔记本上的推到变化负号)
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0:
                    print("eta>=0")
                    continue
                #4 更新aj
                alphas[j]=alphas[j]-labelMatrix[j]*(Ei-Ej)/eta
                #5 修建aj
                alphas[j]=clippedAlpha(alphas[j],H,L)
                #6 更新ai
                alphas[i]+=labelMatrix[i]*labelMatrix[j]*(alphaJ_old-alphas[j])
                #7 更新b1 b2
                b1=b-Ei-labelMatrix[i]*(alphas[i]-alphaI_old)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMatrix[j]*(alphas[j]-alphaJ_old)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2=b-Ej-labelMatrix[i]*(alphas[i]-alphaI_old)*dataMatrix[j,:]*dataMatrix[i,:].T-labelMatrix[j]*(alphas[j]-alphaJ_old)*dataMatrix[j,:]*dataMatrix[j,:].T
                #8 根据b1 b2 更新b
                if (alphas[i]>0) and (alphas[i]<C):
                    b=b1
                elif (alphas[j]>0) and(alphas[j]<C):
                    b=b2
                else:
                    b=(b1+b2)/2.0

                #统计优化次数
                alphaPairsChanged+=1
                print("第{:d}次迭代 样本{:d} alpha优化次数%{:d}".format(iter_num,i,alphaPairsChanged))
        #更新迭代次数
        if (alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数{:d}".format(iter_num))
    return b,alphas

#计算w
#w=alpha*y*x
def getW(dataMat,labelMat,alphas):
    alphas,labelMat,dataMat=np.array(alphas),np.array(labelMat),np.array(dataMat)
    w=np.dot((np.tile(labelMat.reshape(1,-1).T,(1,2))*dataMat).T,alphas)
    return w.tolist()

def showClassifer(dataMat,w,b):
    data_plus=[]
    data_minus=[]
    for i in range(len(dataMat)):
        if labelMat[i]>0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np=np.array(data_plus).T
    data_minus_np=np.array(data_minus).T

    plt.scatter(data_plus_np[0],data_plus_np[1],s=30,alpha=0.5)
    plt.scatter(data_minus_np[0],data_minus_np[1],s=30,alpha=0.5)

    #绘制直线
    x1=max(dataMat)[0]
    x2=min(dataMat)[0]
    a1,a2=w
    b=float(b)
    a1=float(a1[0])
    a2=float(a2[0])
    y1,y2=(-b-a1*x1)/a2,(-b-a1*x2)/a2
    plt.plot([x1,x2],[y1,y2])
    #找出超平面上的点
    for i,alpha in enumerate(alphas):
        if abs(alpha)>0:
            x,y=dataMat[i]
            plt.scatter([x],[y],s=150,c='none',alpha=0.5,linewidths=1.5,edgecolors='red')
    plt.show()

if __name__=="__main__":
    start = time.clock()

    dataMat,labelMat=loadDataSet("/home/szu/PycharmProjects/machineLearning/SVM/textSet")
    b,alphas=smoSimple(dataMat, labelMat, 0.6,0.001,40)
    w=getW(dataMat,labelMat,alphas)
    # 中间写上代码块
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))
    showClassifer(dataMat,w,b)
    showData(dataMat,labelMat)