#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 19:50:14 2018

@author: sundaoping

数据准备
特征提取
模型准备
训练模型
使用模型


今天这个菜真好吃！ ->吃饭
嗨！今天天气不错！ ->打招呼
今天很开心，明天见！ ->再见

"""

#数据准备
list_sen=['今天这个菜真好吃！','嗨！今天天气不错！','今天很开心，明天见！']


#特征提取   这里我们将每一个字作为一个特征，1/（字出现的总次数）作为权值

dict_voc = dict()
for s in list_sen:
    for w in s:
        if w in dict_voc.keys():
            dict_voc[w] += 1
        else:
            dict_voc[w] = 1
            

#for ch in dict_voc:
#    print(ch, dict_voc[ch])
    #print(dict_voc[ch])
    

features_one = dict() # 吃饭
features_two = dict() # 打招呼
features_thr = dict() # 再见

for w in list_sen[0]:
    if w in  features_one.keys():
        features_one[w] += 1
    else:
        features_one[w] = 1
        
for w in list_sen[1]:
    if w in  features_two.keys():
        features_two[w] += 1
    else:
        features_two[w] = 1  
        
        
for w in list_sen[2]:
    if w in  features_thr.keys():
        features_thr[w] += 1
    else:
        features_thr[w] = 1         

for ch1 in features_one:
    print(ch1, features_one[ch1])

print()

for ch2 in features_two:
    print(ch2, features_two[ch2])
print()

for ch3 in features_thr:
    print(ch3, features_thr[ch3])