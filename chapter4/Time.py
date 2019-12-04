#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Shutian
# @Date  : 
# @Desc  : calculate time metrics
import math

def weight1(yeara, yearb):
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

def weight2(yeara, yearb):
    #basline 2
    return math.pow(math.e, -(int(yeara)-int(yearb)))

year = {}
with open("E:\\deal\\pubmed\\time\\id-year-new.txt", 'r') as f:
    for line in f:
        pid = line.strip().split('\t')[0]
        py = line.strip().split('\t')[1]
        year[pid] = py

result = open(r'E:\\deal\\pubmed\\chapter4\\tim\\train-time', 'w')
with open('E:\\deal\\pubmed\\chapter4\\simi\\train-simi', 'r') as f:
    for line in f:
        print("count .....")
        pp = line.strip().split('\t')[0]
        p1 = pp.split(' ')[0]
        p2 = pp.split(' ')[1]
        time1 = year[p2]
        time0 = year[p1]
        time2 = int(year[p1])-int(year[p2])
        time3 = weight1(int(year[p1]), int(year[p2]))
        time4 = weight2(int(year[p1]), int(year[p2]))
        result.write(p1+" "+p2+"\t"+str(time0)+" "+str(time1)+" "+str(time2)+" "+str(time3)+" "+str(time4)+"\n")
result.close()

