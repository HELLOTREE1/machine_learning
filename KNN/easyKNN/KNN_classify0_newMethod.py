#-*-coding:utf-8-*-
import numpy as np
import operator
import collections

def classify0(inx, dataset, labels, k):
	# 计算距离
    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
	dist = np.sum((inx - dataset)**2, axis=1)**0.5
	# k个最近的标签
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	# 出现次数最多的标签即为最终类别
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label