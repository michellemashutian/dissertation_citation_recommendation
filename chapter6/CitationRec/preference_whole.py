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


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    # s = x
    return s

k = 10
preference = {}
with open("E:\\deal\\pubmed\\result\\result-sa-new\\sa-"+str(k)+"-pre", 'r') as f:
    for line in f:
        idsss = line.strip().split('\t')[0]
        veccc = line.strip().split('\t')[1]
        preference[idsss] = veccc

year = {}
with open("E:\\deal\\pubmed\\time\\id-year-new.txt", 'r') as f:
    for line in f:
        pid = line.strip().split('\t')[0]
        py = line.strip().split('\t')[1]
        year[pid] = py

print 'loading model'
model = KeyedVectors.load_word2vec_format('E:\\deal\\pubmed\\vector\\doc2vec-keyedformat-2-1000', binary=False)


result1 = open(r'E:\\deal\\pubmed\\result\\result-sa-new\\sa-'+str(k)+'-output', 'w')

with open('E:\\deal\\pubmed\\4-10\\2-test-1000', 'r') as f:
    for line in f:
        resulto = {}
        ids = line.strip().split('\t')[0]
        vec = line.strip().split('\t')[1].split(' ')
        vec = map(eval, vec)
        test_vector = np.asarray(vec)
        puby = year[ids]
        if ids in preference:
            prefs = preference[ids].split(" ")
            result = model.wv.similar_by_vector(test_vector, topn=57465, restrict_vocab=None)
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
                        if int(pyear) < 1996:
                            resulto[pid] = simi * sigmoid(float(prefs[0]))
                        elif (int(pyear) < 2001) and (int(pyear) > 1995):
                            resulto[pid] = simi * sigmoid(float(prefs[1]))
                        elif (int(pyear) < 2004) and (int(pyear) > 2000):
                            resulto[pid] = simi * sigmoid(float(prefs[2]))
                        elif (int(pyear) < 2006) and (int(pyear) > 2003):
                            resulto[pid] = simi * sigmoid(float(prefs[3]))
                        elif (int(pyear) < 2008) and (int(pyear) > 2005):
                            resulto[pid] = simi * sigmoid(float(prefs[4]))
                        elif (int(pyear) < 2010) and (int(pyear) > 2007):
                            resulto[pid] = simi * sigmoid(float(prefs[5]))
                        elif int(pyear) > 2009:
                            resulto[pid] = simi * sigmoid(float(prefs[6]))
                        else:
                            print 'yyy', pyear
                            resulto[pid] = 0
                    if len(resulto) == 500:
                        break
            if len(resulto) == 500:
                ress = sorted(resulto.items(), key=lambda x: x[1], reverse=True)
                for indx, i in enumerate(ress):
                    result1.write(str(ids)+' Q0 '+str(i[0])+' '+str(indx+1) + ' ' + str(i[1])+' w\n')
            else:
                print len(resulto)
                print 'wrong'
    result1.close()

