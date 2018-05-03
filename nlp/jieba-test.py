#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:03:50 2018

@author: sundaoping

ref : https://github.com/fxsjy/jieba
ref : http://tech.int.nuomi.com/article/nlp
"""

import jieba as jieba
import jieba.posseg as pseg # 词性标注
import codecs # 由于python中默认的编码是ascii，如果直接使用open方法得到文件对象然后进行文件的读写，都将无法使用包含中文字符（以及其他非ascii码字符），因此建议使用utf-8编码。
import re


#demo
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))



jieba.add_word('布达佩斯七季酒店')  #添加专有名词
seg_list = pseg.cut('小明预订了布达佩斯七季酒店的房间呀')


file = codecs.open('./wiki.seg.before.txt', 'r', encoding='utf-8')
lines = [line.strip('\n') for line in file] 


file_content = ''.join(lines);

file_content2 = re.sub("[\s+\.\!\/_,$%^*(+\"\'\[\]]+|[+——！，．。？、~@#￥%……&*（）\d；]+", "",file_content)
                
#print(file_content2)
#print(lines)

fw = codecs.open('wiki.seg.strip.txt','w','utf-8')
fw.write(file_content2)

seg_list = jieba.cut(file_content2)

#seg_content = ' '.join(seg_list)

    
    



seg_words = []
stop_words = []
words = codecs.open('./stopwords.txt', 'r', encoding='utf8') #加载停用词 “了”
stop_words = [i.strip() for i in words]
for word in seg_list:
    #print(' ' + word)
    if word not in stop_words:
       seg_words.append(word)
        #词性过滤
       # if i.flag != 'y':  #过滤语气词
        #    print(i.word,i.flag)

fw2 = codecs.open('wiki.seg.seg.txt', 'w', 'utf-8')
fw2.write(' ' .join(seg_words))