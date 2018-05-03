#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:13:07 2018

@author: sundaoping
"""


import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


model = Word2Vec.load('./wiki.zh.text.model')
word = model.most_similar('台风')
for t in word:
    print(t[0],t[1])
    print()