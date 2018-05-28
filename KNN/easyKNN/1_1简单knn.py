# -*-coding:utf-8-*-
import numpy as np

# from KNN_classify0 import classify0
from KNN_classify0_newMethod import classify0
"""
创建数据集
parameters:
无
returns：
group-数据集
labels-分类标签
"""


def createDataSet():
    # 6组二维特征
    group = np.array([[1, 109], [5, 89], [108, 5], [115, 8], [2, 98], [1, 100]])
    # 特征标签
    labels = ['爱情片', '爱情片', '动作片', '动作片', '爱情片', '爱情片']
    return group, labels


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()

    test=[3,120]
    test_class=classify0(test,group,labels,3)
    print(test_class)
    # print(group)
    # print(labels)
