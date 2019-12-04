#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 
# @Desc  : calculate similarity

result1 = open(r'E:\\deal\\pubmed\\chapter4\\auth\\query-in-out', 'w')

indict = {}
outdict = {}
indict1 = {}
outdict1 = {}

pid = {}

with open('E:\\deal\\pubmed\\chapter4\\query', 'r') as f:
    for line in f:
        p1 = line.strip().split(' ')[0]
        p2 = line.strip().split(' ')[2]
        c = line.strip().split(' ')[3]
        pid[p1] = 1
        pid[p2] = 1

        if p1 in outdict:
            a = outdict[p1]
            outdict[p1] = int(a) + int(c)
        else:
            outdict[p1] = int(c)
        if p2 in indict:
            a = indict[p2]
            indict[p2] = int(a) + int(c)
        else:
            indict[p2] = int(c)

        if p1 in outdict1:
            a = outdict1[p1]
            outdict1[p1] = a + 1
        else:
            outdict1[p1] = 1
        if p2 in indict1:
            a = indict1[p2]
            indict1[p2] = a + 1
        else:
            indict1[p2] = 1

for p in pid:
    ind = 0
    oud = 0
    if p in outdict:
        oud = outdict[p]
    if p in indict:
        ind = indict[p]
    ind1 = 0
    oud1 = 0
    if p in outdict1:
        oud1 = outdict1[p]
    if p in indict1:
        ind1 = indict1[p]
    result1.write(str(p) + '\t' + str(ind) + ' ' + str(oud) + ' ' + str(ind1) + ' ' + str(oud1) + '\n')
result1.close()