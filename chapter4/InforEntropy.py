#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 
# @Desc  :

import math
from gensim.models import KeyedVectors
from sklearn import preprocessing


def normalizations(vec):
    vec1 = min_max_scaler.fit_transform(vec.reshape(-1, 1))
    for index, value in enumerate(vec1):
        if value == 0:
            vec1[index] = 1e-10
    return vec1


model1 = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\chapter4\\simi\\lsi', binary=False)
print('loading model done')
model2 = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\chapter4\\simi\\lda', binary=False)
print('loading model done')
model3 = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\chapter4\\simi\\d2v', binary=False)
print('loading model done')

result = open(r'E:\\deal\\pubmed\\chapter4\\diver\\test-entropy', 'w')
min_max_scaler = preprocessing.MinMaxScaler()
with open('E:\\deal\\pubmed\\chapter4\\simi\\test-simi', 'r') as f:
    for line in f:
        print("count .....")
        pp = line.strip().split('\t')[0]
        p1 = pp.split(' ')[0]
        p2 = pp.split(' ')[1]
        v11 = normalizations(model1[p1])
        v12 = normalizations(model1[p2])
        v21 = normalizations(model2[p1])
        v22 = normalizations(model2[p2])
        v31 = normalizations(model3[p1])
        v32 = normalizations(model3[p2])
        inf1 = 0
        inf2 = 0
        inf3 = 0
        re1 = 0
        re2 = 0
        re3 = 0
        for vec in v12:
            inf1 = inf1-vec*math.log(vec, 2)
        for vec in v22:
            inf2 = inf2-vec*math.log(vec, 2)
        for vec in v32:
            inf3 = inf3-vec*math.log(vec, 2)
        for v1, v2 in zip(v11, v12):
            re1 = re1 + v1*math.log(v1/v2, 2)
        for v1, v2 in zip(v21, v22):
            re2 = re2 + v1*math.log(v1/v2, 2)
        for v1, v2 in zip(v31, v32):
            re3 = re3 + v1*math.log(v1/v2, 2)
        result.write(p1+" "+p2+"\t"+str(inf1[0])+" "+str(inf2[0])+" "+
                     str(inf3[0])+" "+str(re1[0])+" "+str(re2[0])+" "+str(re3[0])+"\n")
result.close()

