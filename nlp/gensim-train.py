#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:29:47 2018

@author: sundaoping
"""

import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
fdir = './'
inp = fdir + 'wiki.seg.seg.txt'
outp1 = fdir + 'wiki.zh.text.model'
outp2 = fdir + 'wiki.zh.text.vector'
model = Word2Vec(LineSentence(inp), size = 400, window =5, min_count=5, workers=multiprocessing.cpu_count())
model.save(outp1)
model.wv.save_word2vec_format(outp2, binary=False)

