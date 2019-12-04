#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 2019/3/6
# @Desc  : baseline for whole time period recommendation

from gensim.models import KeyedVectors
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# iddd = {}
# with open("E:\\deal\\pubmed\\4-10\\2-test-1000-id", 'r') as f:
#     for line in f:
#         iddd[line.strip()] = 1
#
#
# outp = open('E:\\deal\\pubmed\\4-10\\2-test-1000-lda', 'w')
#
# with open("E:\\deal\\pubmed\\content\\ldda1", 'r') as f:
#     for line in f:
#         x = line.strip().split(' ')[0]
#         y = line.strip()[len(x)+1:]
#         if x in iddd:
#             outp.write(x+'\t'+y+'\n')
# outp.close()

year = {}
with open("E:\\deal\\pubmed\\time\\id-year-new-new.txt", 'r') as f:
    for line in f:
        pid = line.strip().split('\t')[0]
        py = line.strip().split('\t')[1]
        year[pid] = py

print 'loading model'
# model = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\content\\ldda1', binary=False)
model = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\vector\\doc2vec-keyedformat-2-1000', binary=False)
# model = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\chapter3\\glove-fasttext\\test-fasttext-subword-mean-doc2vec-format', binary=False)
# E:\\deal\\pubmed\\chapter3\\lsi-lda-d2v\\test-lda-doc2vec-format


result1 = open(r'E:\\deal\\pubmed\\chapter3\\lsi-lda-d2v\\baseline-doc2vec-1', 'w')

# chapter3\\glove-fasttext\\test-fasttext-subword-mean
with open('E:\\deal\\pubmed\\4-10\\2-test-1000', 'r') as f:
    for line in f:
        resulto = {}
        ids = line.strip().split('\t')[0]
        vec = line.strip().split('\t')[1].split(' ')
        vec = map(eval, vec)
        test_vector = np.asarray(vec)
        puby = year[ids]
        result = model.wv.similar_by_vector(test_vector, topn=58430, restrict_vocab=None)
        for res in result:
            if str(res[0]) == ids:
                continue
            else:
                pid = res[0]
                simi = res[1]
                pyear = year[str(pid)]
                if int(pyear) > int(puby):
                    continue
                else:
                    resulto[pid] = simi
                if len(resulto) == 500:
                    break
        if len(resulto) == 500:
            ress = sorted(resulto.items(), key=lambda x: x[1], reverse=True)
            for indx, i in enumerate(ress):
                result1.write(str(ids)+' Q0 '+str(i[0])+' '+str(indx+1) + ' ' + str(i[1])+' x\n')
        else:
            print 'no'
    result1.close()

