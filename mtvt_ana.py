# -*- coding: utf-8 -*-  
from __future__ import print_function
import csv
import sys


pList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
sList = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3]
traceMtvtDict = {'prn_1': [26, 34, 38], 'ts_0': [14, 48, 79], 'usr_0': [72, 77, 80], 
'wdev_0': [20, 63, 86], 'proj_3': [24, 36, 67], 'prxy_0': [41, 49, 67], 
'hm_0': [31, 38, 52], 'src2_0': [36, 39, 48]}


class dLookup():

    """ 
    input key=trace, value=[(p,s,hr)]
    dLookup key = (trace, p, s), value = hr
    """
    def __init__(self, d):
        self.dLookup = {}
        for key in d.keys():
            l = d[key]
            for item in l:
                (p, s, hr) = item
                self.dLookup[(key, p, s)] = hr
    
    # find hr+-hitRange内的(p, s)
    # p=1 属于 defaults
    # return (defaults, allcons)
    def find_default(self, trace, hr, hitRange):
        defaults = []
        allcons = []
        for key in self.dLookup.keys():
            (pTrace, p, s) = key
            if pTrace==trace:
                phr = self.dLookup[key]
                if phr >= hr-hitRange and phr <= hr+hitRange:
                    if p==1:
                        defaults.append((p,s))
                    allcons.append((p,s))
        if defaults==[]:
            maxp = 0
            for (p,s) in allcons:
                if p>maxp:
                    maxp = p
            for (p,s) in allcons:
                if p==maxp:
                    defaults.append((p,s))
        return (defaults, allcons)

    def find_hr(self, trace, p, s):
        return self.dLookup[(trace, p, s)]
            
def load_mtvt_file_wsh(filename):
    d = {}
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            trace = row[0]
            write = int(row[5])
            p = float(row[1])
            size = int(row[3])
            hr = 100*float(row[4])
            key = trace
            value = (size, p, write, hr)
            if key in d and value not in d[key]:
                d[key].append(value)
            else:
                d[key] = [value]
    csvfile.close()
    return d

# return d, dict, key = trace, value = (p, size, hr)
def load_mtvt_file(filename):
    d = {}
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # print row
            trace = row[0]
            p = float(row[1])
            size = float(row[2])
            # value = int(1000*float(row[4]))
            hr = int(100*float(row[4]))
            key = trace
            value = (p, size, hr)
            if key in d and value not in d[key]:
                d[key].append(value)
            else:
                d[key] = [value]
    csvfile.close()
    return d



# whether p1<=p2 and s1<=s2, hr1<=hr2
def check_ps_principle(filename, d): 
    result = []
    rate = 0
    
    traces = d.keys()
    traces.sort()
    for trace in traces:
        s = 0
        r = 0
        l = d[trace]
        for i in range(0, len(l)):
            for j in range(i+1, len(l)):
                a = l[i]
                b = l[j]
                if a[0] <= b[0]:
                    if a[1] <= b[1]:
                        s += 1
                        if a[2] <= b[2]:
                            r += 1
                elif a[1] >= b[1]:
                    s += 1
                    if a[2] >= b[2]:
                        r += 1
        result.append((r, s, 1.0*r/s))
        rate += 1.0*r/s
    print(result)
    print(rate/len(traces))
 

# return the best config as（cost, p, s)
def get_config(trace, dLookup, default, hr, hitRange, nrround, matrix):
    (p, s) = default
    minCost = p*s
    minConfig = (p,s)
    preRound = 0
    while preRound < nrround:
        print("preRound", preRound, minConfig)
        sign = False
        par = nrround-preRound
        pindex = pList.index(minConfig[0])
        sindex = sList.index(minConfig[1])
        print(pindex, sindex, par)
        (pstart, pend) = get_range(pindex, matrix, par, pList, pList.index(p))
        (sstart, send) = get_range(sindex, matrix, par, sList, len(sList))
        print("p", pstart, pend, pList[pstart], pList[pend-1])
        print("s", sstart, send, sList[sstart], sList[send-1])
        for a in range(0, matrix):
            for b in range(0, matrix):
                i = min(pstart + a*par, len(pList)-1)
                j = min(sstart + b*par, len(sList)-1)
                # if pList[i] > p:
                #     # print("debug", pList[i], p)
                #     continue
                if minCost > pList[i]*sList[j]:
                    newhr = dLookup.find_hr(trace, pList[i], sList[j])
                    print(minCost, pList[i], sList[j], newhr)
                    if newhr >= hr-hitRange and newhr <= hr+hitRange:
                        minCost = pList[i]*sList[j]
                        minConfig = (pList[i], sList[j])
                        sign = True
                        preRound = 0
        # can't find better one
        if not sign:
            preRound += 1
    return (minCost, minConfig[0], minConfig[1])


def get_range(i, matrix, par, l, maxIndex):
    start = i
    end = i+1
    n = 1
    while(n<matrix):
        if start > 0:
            start = max(0, start-par)
            n += 1
        if end < maxIndex:
            end = min(maxIndex, end+par)
            n += 1
    # print(start, end)
    return (start, end)

def lookup_matrix(dLookup):
    p = input("p:")
    s = input("s:")
    trace = input("trace:")
    matrix = input("#matrix:")
    pindex = pList.index(p)
    sindex = sList.index(s)
    (pstart, pend) = get_range(pindex, matrix, 1, pList)
    (sstart, send) = get_range(sindex, matrix, 1, sList)
    
    # pstart = max(0, int(pindex-(matrix-1)/2))
    # sstart = max(0, int(sindex-(matrix-1)/2))
    # for i in range(pstart, min(pstart+matrix, len(pList))):
    #     if (i == pstart):
    #         print("\t", sList[sstart:min(sstart+matrix, len(sList))])
    #     print(pList[i], end="\t")
    #     for j in range(sstart, min(sstart+matrix, len(sList))):
    #         print(dLookup.find_hr(trace, pList[i], sList[j]), end=",")
    #     print()
    for i in range(pstart, pend):
        if i == pstart:
            print("\t", sList[sstart:send])
        print(pList[i], end="\t")
        for j in range(sstart, send):
            print(dLookup.find_hr(trace, pList[i], sList[j]), end=",")
        print()


# def original_pmuls():
    
#     filename = "mtvt_result.csv"
#     d = load_mtvt_file(filename)    
#     hitRange = 1
#     dLookup = dLookup(d)
#     indexList = []
#     costList = []
#     pSmallList = []
#     recordList = []
#     for trace in traceMtvtDict:
#         print(trace)
#         l = traceMtvtDict[trace]
#         for hr in l:
#             (defaults, allcons) = dLookup.find_default(trace, hr, hitRange)
#             print("defaults", defaults)
#             print("allcons", allcons)
#             templ = []        
#             for cons in allcons:
#                 (p, s) = cons
#                 templ.append((p*s, p, s))
#             templ.sort()
#             print("templ", templ)
#             print("hr", hr)
#             for default in defaults:
#                 (p, s) = default
#                 # p is too small
#                 if p < 0.5:
#                     pSmallList.append((trace, p, s))
#                     continue
#                 bestCon = get_config(trace, dLookup, default, hr, hitRange, 3, 3)
#                 index = templ.index(bestCon)
#                 print(bestCon, index)
#                 indexList.append(index)
#                 recordList.append((trace, default, hr, bestCon, index))
#                 costList.append(bestCon[0]/(default[0]*default[1]))
#                 # sys.exit(0)
#     print(indexList)
#     print(recordList)
#     print(indexList.count(0), indexList.count(1), len(indexList))
#     print(1.0*sum(indexList)/len(indexList))
#     print(costList)
#     print(sum(costList)/len(costList))
#     print(pSmallList)


def get_cost():
    pass

def new_sorwrite():
    filename = "mtvt_result.csv"
    d = load_mtvt_file_wsh(filename) 
    del d["prn_1"]
    hitRange = 1
    baseline = (0, 0)
    # 7%, 10%, 20%
    for sizeIdx in [3, 6, 8]: 
        result = []
        for trace in d:
            tracePotential = []
            value = d[trace]
            size = value[sizeIdx][0]
            for item in value:
                (psize, p, pwrite, phr) = item
                if psize == size and p == 1:
                    hr = phr
                    baseline[0] += psize
                    baseline[1] += pwrite
                    tracePotential.append((psize, pwrite))
                    break
            for item in value:
                (psize, p, pwrite, phr) = item
                if abs(hr-item[-1])<=hitRange:                    
                    tracePotential.append((psize, pwrite))
            result.append(tracePotential)
    basecost = 

            
