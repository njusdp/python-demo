#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:34:23 2018

@author: sundaoping
决策树cart算法
ref : https://blog.csdn.net/wyb_009/article/details/8972652
"""
from math import log

my_data=[['beijing','yes','yes', 28,'yes'],     #A
        ['hebei',   'no', 'no',  23,'no'],      #B
        ['henan',  'yes', 'no',  24,'yes'],     #C
        ['hubei',  'no' , 'yes', 28,'no'],      #D
        ['hunan',  'yes', 'yes', 39,'no']]      #E

#1.定义结点：每个节点必须表明自己代表的那个考察因素，一个成立或者不成立的判断条件，一个指向条件成立后的要执行下一个节点，
#一个指向条件不成立后要判断的下一个节点。如果是末级叶子节点，还需要一个结果节点，来指明要不要参加非诚勿扰。
class decisionnode:
    
    
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col
        self.value=value
        self.results=results
        self.tb=tb
        self.fb=fb


#2.输入数据分组：我们考虑因素有四个，祖籍，房，车和年龄。祖籍是字符型，年龄是整数型，是否有车有房是布尔型；
#假设我们根据是否有房分组，可以将用户分为[A,C,E]和 [B,D]，
#如果根据是否有车分组，可以将用户分为[A,D,E]和 [B,C]；
#如果根据年龄是否大于等于39分组，可分为 [E]和[A,B,C,D];
#根据年龄是否大于等于28分组，可分为[A,D,E]和 [B,C]; 
#根据是否是北京人，分为[A]和[B,C,D,E]...
def divideset(rows,column,value):
    
    split_function=None
    if isinstance(value,int) or isinstance(value,float):
       split_function=lambda row:row[column]>=value
    else:
       split_function=lambda row:row[column]==value
   
    # Divide the rows into two sets and return them
    set1=[row for row in rows if split_function(row)]
    set2=[row for row in rows if not split_function(row)]
    return (set1,set2)

#3.第三步：拆分数据的评价模型：
#找出合适的因素，使根据该因素divideset后生成的两组数据集的混杂程度尽可能的小,
#先定义个一个小工具函数，这个函数返回rows里各种可能结果和结果的出现的次数；
def uniquecounts(rows):
    results = {}
    for row in rows:
        if row[4] not in results:
            results[row[4]] = 1
        else:
            results[row[4]] += 1;
    return results
        
    #基尼不纯度      
def giniimpurity(rows):
  total=len(rows)
  counts=uniquecounts(rows)
  imp=0
  for k1 in counts:
    p1=float(counts[k1])/total
    for k2 in counts:
      if k1==k2: continue
      p2=float(counts[k2])/total
      imp+=p1*p2
  return imp
  

#熵：代表集合的无序程度，也就是集合的混杂程度：
def entropy(rows):
   log2=lambda x:log(x)/log(2)  
   results=uniquecounts(rows)
   # Now calculate the entropy
   ent=0.0
   for r in results.keys():
      p=float(results[r])/len(rows)
      ent=ent-p*log2(p)
   return ent
   

#4.构建决策树，构建决策树的过程，就是找出一个因素，根据这个因素对数据分组，使得分组后数据的熵比根据其他因素分组后的熵都要低。
def buildtree(rows,scoref=entropy):
  if len(rows)==0: return decisionnode()
  current_score=scoref(rows)

  # Set up some variables to track the best criteria
  best_gain=0.0
  best_criteria=None
  best_sets=None
  
  column_count=len(rows[0])-1
  for col in range(0,column_count):
    # Generate the list of different values in
    # this column
    column_values={}
    for row in rows:
       column_values[row[col]]=1
    # Now try dividing the rows up for each value
    # in this column
    for value in column_values.keys():
      (set1,set2)=divideset(rows,col,value)
      
      # Information gain
      p=float(len(set1))/len(rows)
      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)
      if gain>best_gain and len(set1)>0 and len(set2)>0:
        best_gain=gain
        best_criteria=(col,value)
        best_sets=(set1,set2)
  # Create the sub branches   
  if best_gain>0:
    trueBranch=buildtree(best_sets[0])
    falseBranch=buildtree(best_sets[1])
    return decisionnode(col=best_criteria[0],value=best_criteria[1],
                        tb=trueBranch,fb=falseBranch)
  else:
    return decisionnode(results=uniquecounts(rows))

#5.对新数据进行判定
#classfiy以buildtree返回的结果为参数，沿着树向下遍历。每次调用后，函数会根据调用结果来判断是否达到分支的末端。
#如果尚未达到末端，他会对观测数据做出评估，以确认列数据是否与参考值匹配。如果匹配，则会再次在True分支上调用classify，
#否则会在false分支上调用classify函数。当函数到达一个包含结果信息的节点时，就会对观测数据给出结果了。
def classify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    branch=None
    if isinstance(v,int) or isinstance(v,float):
      if v>=tree.value: branch=tree.tb
      else: branch=tree.fb
    else:
      if v==tree.value: branch=tree.tb
      else: branch=tree.fb
    return classify(observation,branch)

#6.剪枝
#先构建好整棵树，然后尝试消除多余的节点，这就是剪枝。剪枝的过程就是对具有相同父节点的一组节点进行检查，
#判断如果将其合并，熵的增加量是否会小于某个阈值。如果确实如此，则这些叶子节点被合并成一个单一的节点
def prune(tree,mingain):
  # If the branches aren't leaves, then prune them
  if tree.tb.results==None:
    prune(tree.tb,mingain)
  if tree.fb.results==None:
    prune(tree.fb,mingain)
    
  # If both the subbranches are now leaves, see if they
  # should merged
  if tree.tb.results!=None and tree.fb.results!=None:
    # Build a combined dataset
    tb,fb=[],[]
    for v,c in tree.tb.results.items():
      tb+=[[v]]*c
    for v,c in tree.fb.results.items():
      fb+=[[v]]*c
    
    # Test the reduction in entropy
    delta=entropy(tb+fb)-(entropy(tb)+entropy(fb)/2)

    if delta<mingain:
      # Merge the branches
      tree.tb,tree.fb=None,None
      tree.results=uniquecounts(tb+fb) 
      
#7.缺失值的处理
#我们的训练数据只有五个省的用户，如果判断一个祖籍是安徽的用户呢？安徽不在我们的训练集类，决策树也就不会有安慰的信息。如果我们缺失了一些信息，
#而这个信息是确定分支走向所必须的，那我们可以选择两个分支都走。不过不是平均地统计各分支对应的结果值，而是对其加权统计。
def mdclassify(observation,tree):
  if tree.results!=None:
    return tree.results
  else:
    v=observation[tree.col]
    if v==None:
      tr,fr=mdclassify(observation,tree.tb),mdclassify(observation,tree.fb)
      tcount=sum(tr.values())
      fcount=sum(fr.values())
      tw=float(tcount)/(tcount+fcount)
      fw=float(fcount)/(tcount+fcount)
      result={}
      for k,v in tr.items(): result[k]=v*tw
      for k,v in fr.items(): result[k]=v*fw      
      return result
    else:
      if isinstance(v,int) or isinstance(v,float):
        if v>=tree.value: branch=tree.tb
        else: branch=tree.fb
      else:
        if v==tree.value: branch=tree.tb
        else: branch=tree.fb
      return mdclassify(observation,branch)

tree=buildtree(my_data)  
print(mdclassify(['beijing', 'yes', 'yes', 22], tree)  )
print(mdclassify(['beijing', 'no', 'yes', 30], tree)  )
print(mdclassify(['beijing', 'no', 'yes', 20], tree))
print(mdclassify(['beijing', 'yes', 'no', 40], tree)  )