#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:02:08 2018

@author: sundaoping
"""

"""
功能：回归决策树
说明：人为设置函数模型为每隔5个点引入噪音的离散的sin(x)，我们利用决策树回归拟合这些数据
作者：唐天泽
博客：http://blog.csdn.net/u010837794/article/details/76596063
日期：2017-08-03
"""

"""
导入项目所需的包
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
# 使用交叉验证的方法，把数据集分为训练集合测试集
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def creat_data(n):
    '''
    产生用于回归问题的数据集
    :param n:  数据集容量
    :return: 返回一个元组，元素依次为：训练样本集、测试样本集、训练样本集对应的值、测试样本集对应的值
    '''
    np.random.seed(0)
    X = 5 * np.random.rand(n, 1)
    y = np.sin(X).ravel()
    noise_num=(int)(n/5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num)) # 每第5个样本，就在该样本的值上添加噪音
    X_train, X_test, y_train, y_test=train_test_split(X, y,test_size=0.25,random_state=1)
    return X_train, X_test, y_train, y_test # 拆分原始数据集为训练集和测试集，其中测试集大小为元素数据集大小的 1/4


# 使用DecisionTreeRegressor考察线性回归决策树的预测能力
def test_DecisionTreeRegressor(X_train, X_test, y_train, y_test):
    # 选择模型
    cls = DecisionTreeRegressor()

    # 把数据交给模型训练
    cls.fit(X_train, y_train)

    print("Training score:%f" % (cls.score(X_train, y_train)))
    print("Testing score:%f" % (cls.score(X_test, y_test)))

    """绘图"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = np.arange(0.0, 5.0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    X = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]  # X为array([[ 0.  ],[ 0.01],...,[4.99]]
    Y = cls.predict(X)

    # 离散点
    ax.scatter(X_train, y_train, label="train sample", c='g')
    ax.scatter(X_test, y_test, label="test sample", c='r')

    # 连续点
    ax.plot(X, Y, label="predict_value", linewidth=2, alpha=0.5)

    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha=0.5)
    plt.show()

if __name__=='__main__':
    X_train,X_test,y_train,y_test=creat_data(50) # 产生用于回归问题的数据集
    test_DecisionTreeRegressor(X_train,X_test,y_train,y_test) # 调用 test_DecisionTreeRegressor