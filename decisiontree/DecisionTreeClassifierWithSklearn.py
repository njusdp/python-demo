#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:16:47 2018

@author: sundaoping
"""

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
import os
import pydot
import pydotplus
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image 


#iris以鸢尾花的特征作为数据来源，常用在分类操作中。该数据集由3种不同类型的鸢尾花的50个样本数据构成。
#其中的一个种类与另外两个种类是线性可分离的，后两个种类是非线性可分离的。
#该数据集包含了5个属性：
#& Sepal.Length（花萼长度），单位是cm;
#& Sepal.Width（花萼宽度），单位是cm;
#& Petal.Length（花瓣长度），单位是cm;
#& Petal.Width（花瓣宽度），单位是cm;
#& 种类：Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），以及Iris Virginica（维吉尼亚鸢尾）。
iris =load_iris()

clff = tree.DecisionTreeClassifier()
clff.fit(iris.data,iris.target)

dot_data = StringIO()
tree.export_graphviz(clff,out_file=dot_data,feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
    
Image(graph.create_png()) 

print(clff.predict(iris.data[:1, :]))
print(clff.predict_proba(iris.data[45:60, :]))


#为方便画图，用两个连续特征训练&预测
# 仍然使用自带的iris数据
iris = load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# 训练模型，限制树的最大深度4
clf3 = tree.DecisionTreeClassifier(max_depth=4)
#拟合模型
clf3.fit(X, y)


# 画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf3.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()

feature_names2 = [iris.feature_names[0],iris.feature_names[2]]
dot_data = tree.export_graphviz(clf3, out_file=None, 
                         feature_names=feature_names2,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
