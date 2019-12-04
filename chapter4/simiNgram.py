#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 
# @Desc  :

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 2019/3/6
# @Desc  : simi

from strsimpy.levenshtein import Levenshtein
from strsimpy.metric_lcs import MetricLCS
from strsimpy.ngram import NGram
from strsimpy.jaccard import Jaccard
import numpy as np

import time
import random
def bigram_overlap(sstring, tstring):
    s = sstring.strip().split(" ")
    t = tstring.strip().split(" ")

    if len(s) == 0 or len(t) == 0:
        return 0

    bigrams_s = list(zip(s[:-1], s[1:]))
    bigrams_t = list(zip(t[:-1], t[1:]))

    bigrams_s = set(bigrams_s)
    if len(bigrams_s) == 0:
        return 0

    bigrams_t = set(bigrams_t)

    overlap = 0
    for bigram in bigrams_s:
        overlap += 1 if bigram in bigrams_t else 0
    return overlap / len(bigrams_s)

def jaccard(sstring, tstring):
    s = sstring.strip().split(" ")
    t = tstring.strip().split(" ")

    if len(s) == 0 or len(t) == 0:
        return 0

    grams_s = list(set(s))
    grams_t = list(set(t))

    temp = 0
    for i in grams_s:
        if i in grams_t:
            temp = temp + 1
    return float(temp/(len(grams_t)+len(grams_s)-temp))

docs = {}
with open("E:\\deal\\pubmed\\9-12\\abstract-p.txt", 'r') as f:
    for lines in f:
        line = lines.strip().split("\t")
        if len(line) == 2:
            docid = line[0]
            doc = line[1]
            docs[docid] = doc

result1 = open(r'E:\\deal\\pubmed\\chapter4\\simi\\train-simi-char', 'w')
result2 = open(r'E:\\deal\\pubmed\\chapter4\\simi\\test-simi-char', 'w')

with open('E:\\deal\\pubmed\\chapter4\\train', 'r') as f:
    for line in f:
        print('train...')
        p1 = line.strip().split(' ')[0]
        p2 = line.strip().split(' ')[2]
        ngram = bigram_overlap(docs[p1], docs[p2])
        ja = jaccard(docs[p1], docs[p2])

        result1.write(p1+' '+p2+'\t'+str(ngram)+' '+str(ja)+'\n')
result1.close()

with open('E:\\deal\\pubmed\\chapter4\\test', 'r') as f:
    for line in f:
        print('test...')
        p1 = line.strip().split(' ')[0]
        p2 = line.strip().split(' ')[2]
        ngram = bigram_overlap(docs[p1], docs[p2])
        ja = jaccard(docs[p1], docs[p2])
        result2.write(p1 + ' ' + p2 + '\t' + str(ngram) + ' ' + str(ja) + '\n')
result2.close()