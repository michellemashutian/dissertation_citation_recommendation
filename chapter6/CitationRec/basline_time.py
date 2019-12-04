#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 2019/3/6
# @Desc  : baseline for whole time period recommendation

import math
from gensim.models import KeyedVectors
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# iddd = set()
# with open("E:\\deal\\pubmed\\vector\\test-ci-id.txt", 'r') as f:
#     for line in f:
#         iddd.add(line.strip())


weightss = {}
with open("E:\\deal\\pubmed\\vector\\doi-citations-pagerank.txt", 'r') as ff:
    for liness in ff:
        paperid = liness.strip().split('\t')[0]
        pager = liness.strip().split('\t')[1]
        weightss[paperid] = pager

def weight(yeara, yearb):
    #basline 2
    # return math.pow(math.e, -(int(yeara)-int(yearb)))
    # baseline 1
    if yeara == yearb:
        w = math.pow(0.8, 7)
    elif int(yeara) - int(yearb) > 20:
        w = math.pow(0.8, 20)
    else:
        w = math.pow(0.8, int(yeara)-int(yearb))
    return w


year = {}
with open("E:\\deal\\pubmed\\time\\id-year-new.txt", 'r') as f:
    for line in f:
        pid = line.strip().split('\t')[0]
        py = line.strip().split('\t')[1]
        year[pid] = py


print 'loading model'
model = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\vector\\doc2vec-keyedformat-2-1000', binary=False)


result1 = open(r'E:\\deal\\pubmed\\result\\base-pagerank', 'w')

with open('E:\\deal\\pubmed\\4-10\\2-test-1000', 'r') as f:
    for line in f:
        resulto = {}
        ids = line.strip().split('\t')[0]
        vec = line.strip().split('\t')[1].split(' ')
        vec = map(eval, vec)
        test_vector = np.asarray(vec)
        puby = year[ids]
        result = model.wv.similar_by_vector(test_vector, topn=80000, restrict_vocab=None)
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
                    # resulto[pid] = simi * weight(puby, pyear)
                    resulto[pid] = simi * float(weightss[pid]) * (math.pow(math.e, 5))
                if len(resulto) == 500:
                    break
        if len(resulto) == 500:
            ress = sorted(resulto.items(), key=lambda x: x[1], reverse=True)
            for indx, i in enumerate(ress):
                result1.write(str(ids)+' Q0 '+str(i[0])+' '+str(indx+1) + ' ' + str(i[1])+' pre-w\n')
        else:
            print 'wrong'
    result1.close()

