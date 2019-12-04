#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 
# @Desc  :

'''
from gensim.models import KeyedVectors
import numpy as np

model1 = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\chapter4\\simi\\node2vec.txt', binary=False)
print('loading model done')
# model2 = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\chapter4\\simi\\lda', binary=False)
# print('loading model done')
# model3 = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\chapter4\\simi\\d2v', binary=False)
# print('loading model done')

result1 = open(r'E:\\deal\\pubmed\\chapter4\\simi\\train-simi-node', 'w')
result2 = open(r'E:\\deal\\pubmed\\chapter4\\simi\\test-simi-node', 'w')

with open('E:\\deal\\pubmed\\chapter4\\train', 'r') as f:
    for line in f:
        p1 = line.strip().split(' ')[0]
        p2 = line.strip().split(' ')[2]
        lsi = model1.similarity(p1, p2)
        # lda = model2.similarity(p1, p2)
        # d2v = model3.similarity(p1, p2)
        result1.write(p1+' '+p2+'\t'+str(lsi)+'\n')
result1.close()

with open('E:\\deal\\pubmed\\chapter4\\test', 'r') as f:
    for line in f:
        p1 = line.strip().split(' ')[0]
        p2 = line.strip().split(' ')[2]
        lsi = model1.similarity(p1, p2)
        # lda = model2.similarity(p1, p2)
        # d2v = model3.similarity(p1, p2)
        result2.write(p1+' '+p2+'\t'+str(lsi)+'\n')
result2.close()
'''

pid = {}
with open('E:\\deal\\pubmed\\chapter4\\test-id', 'r') as f:
    for line in f:
        pid[line.strip()] = 1

simi = {}
simichar = {}
siminode = {}



with open('E:\\deal\\pubmed\\chapter4\\simi\\simi', 'r') as f:
    for line in f:
        p1p2 = line.strip().split('\t')[0]
        p1 = p1p2.split(' ')[0]
        simi1 = line.strip().split('\t')[1]
        if p1 in pid:
            simi[p1p2] = simi1
with open('E:\\deal\\pubmed\\chapter4\\simi\\simi-char', 'r') as f:
    for line in f:
        p1p2 = line.strip().split('\t')[0]
        p1 = p1p2.split(' ')[0]
        simi2 = line.strip().split('\t')[1]
        if p1 in pid:
            simichar[p1p2] = simi2
with open('E:\\deal\\pubmed\\chapter4\\simi\\simi-node', 'r') as f:
    for line in f:
        p1p2 = line.strip().split('\t')[0]
        p1 = p1p2.split(' ')[0]
        simi3 = line.strip().split('\t')[1]
        if p1 in pid:
            siminode[p1p2] = simi3

result = open(r'E:\\deal\\pubmed\\chapter4\\simi\\test-simi', 'w')
for i in simi:
    if i in simichar and i in siminode:
        result.write(i+'\t'+simi[i]+' '+simichar[i]+' '+siminode[i]+'\n')
    else:
        print(i)
result.close()