# -*- coding: utf-8 -*-
"""
数据说明;
一共有24组数据，数据的Labels依次是age、prescript、astigmatic、tearRate、class，
也就是第一列是年龄，第二列是症状，第三列是是否散光，第四列是眼泪数量，第五列是最终的分类标签。
ecisionTreeClassifier这个函数
criterion：特征选择标准，可选参数，默认是gini，可以设置为entropy
    ID3算法使用的是entropy，CART算法使用的则是gini。
splitter：特征划分点选择标准，可选参数，默认是best，可以设置为random
    默认的”best”适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐”random”。
max_features：划分时考虑的最大特征数，可选参数，默认是None如果样本特征数不多，比如小于50，我们用默认的”None”就可以
    如果max_features是整型的数，则考虑max_features个特征；
    如果max_features是浮点型的数，则考虑int(max_features * n_features)个特征；
    如果max_features设为auto，那么max_features = sqrt(n_features)；
    如果max_features设为sqrt，那么max_featrues = sqrt(n_features)，跟auto一样；
    如果max_features设为log2，那么max_features = log2(n_features)；
    如果max_features设为None，那么max_features = n_features，也就是所有特征都用。

"""

from sklearn import tree
import pandas as pd
import pydotplus
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
if __name__ == '__main__':
    with open("lenses.txt", 'r') as file:
        lenses = [inst.strip().split('\t') for inst in file.readlines()]
    lenses_target = []
    for each in lenses:  # 提取每组数据的类别，保存在列表里
        lenses_target.append(each[-1])

    # 对string类型的数据序列化,原始数据->字典->pandas数据
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []  # 保存lenses数据的临时列表
    lenses_dict = {}  # 保存lenses数据的字典，用于生成pandas
    for each_label in lensesLabels:  # 提取信息，生成字典
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    print(lenses_dict)

    # 数据序列化
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
#编码
    le=LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col]=le.fit_transform(lenses_pd[col])
    print(lenses_pd)

    classify=tree.DecisionTreeClassifier(max_depth=4)
    # 使用数据，构建决策树
    lenses=classify.fit(lenses_pd.values.tolist(),lenses_target)
    dot_data=StringIO()
    tree.export_graphviz(classify, out_file=dot_data,  # 绘制决策树
                         feature_names=lenses_pd.keys(),
                         class_names=classify.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")

    print(classify.predict([[1,1,1,0]]))