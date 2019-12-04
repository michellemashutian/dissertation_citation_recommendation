#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 
# @Desc  :


result = open(r'E:\\deal\\pubmed\\chapter4\\auth\\train-auth', 'w')

inout = {}
pr = {}
hub = {}
auth = {}

with open('E:\\deal\\pubmed\\chapter4\\auth\\query-in-out', 'r') as f:
    for line in f:
        p = line.strip().split('\t')[0]
        d = line.strip().split('\t')[1]
        inout[p] = d
with open('E:\\deal\\pubmed\\chapter4\\auth\\pagerank', 'r') as f:
    for line in f:
        p = line.strip().split('\t')[0]
        d = line.strip().split('\t')[1]
        pr[p] = d
with open('E:\\deal\\pubmed\\chapter4\\auth\\hits-hub', 'r') as f:
    for line in f:
        p = line.strip().split('\t')[0]
        d = line.strip().split('\t')[1]
        hub[p] = d
with open('E:\\deal\\pubmed\\chapter4\\auth\\hits-auth', 'r') as f:
    for line in f:
        p = line.strip().split('\t')[0]
        d = line.strip().split('\t')[1]
        auth[p] = d

with open('E:\\deal\\pubmed\\chapter4\\tim\\train-time', 'r') as f:
    for line in f:
        p = line.strip().split('\t')[0]
        p1 = p.strip().split(' ')[0]
        p2 = p.strip().split(' ')[1]
        result.write(str(p1)+" "+str(p2)+"\t"+inout[p1]+" "+inout[p2]+" "+
                     pr[p1]+" "+pr[p2]+" "+hub[p1]+" "+hub[p2]+" "+auth[p1]+" "+auth[p2]+"\n")
result.close()
