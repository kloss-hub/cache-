#coding=utf-8
from __future__ import print_function
import operator 
import time
import sys
import math
import random
from deal_file import parse_line
from deal_file import load_lines
from cache_algorithm import PLRU

danwei = 10**7
g = 0.014/3600/danwei
# add trace workflow:
# add ucln
# add path

# run overflow:
# 

uclnDict = {"netsfs":47949, "mix":195423,
# "mds_0":	769376,
"hm_0":	488986,
"prn_0":	985474,
"proj_0":	462462,
"prxy_0":	79483,
"rsrch_0":	20855,
# "src1_0":	31656826,
# "usr_1":	171538746,
"src2_0":	104953,
# "stg_0":	1608935,
# "ts_0":	131140,
# "wdev_0":	52111,
# "web_0":	1887115,

"probuild": 1970,
"bs24":	4155,
"bs39":	12190,
"dev5.3.7.19":	1873854,
"dev5.3.11.19":	1081695,
"live6.31":	30393264,
"live3.31":	6920465,
"mfs253":	2222973,
"mfs101":	1408158,
"msc116":	216158,
"msc651":	128895,
"dad807":	294190,
"dap812":	224019,

"prn_1" :	19343730 ,
# "proj_2" : 	107472935,
"proj_3" : 	1383543,
# "prxy_1" : 	196870,
# "src1_1" : 	30780233,
# "src2_1" : 	5019544,
"usr_0" : 	554527,
# "web_2" :	17610218,

"fileserver":	3840461,
"mongo":	287694,
"webproxy":	378100,

"web_3":	58776,
"rsrch_2":	174674,
"wdev_0":	52111,
"mds_0":	769376,
"ts_0":	131140,
"web_1":	962441,
"stg_0":	1608935,
"hm_0":	488986,
"web_0":	1887115,
"src2_2":	5325043,
"proj_3":	1383543,
"usr_0":	554527,
"src2_1":	5019544,
"stg_1":	20865229,
"mds_1":	22014981,
"proj_4":	32411052,
"prn_1":	19343730,
"web_2":	17610218,
"usr_2":	99264060,
"prxy_1":	196870,
"src1_0":	31656826,
"proj_1":	183030933,
"proj_2":	107472935,
"src1_1":	30780233,
"usr_1":	171538746,
"usr_10":	34659971,
"usr_20":	31968051,
"prxy_10":	33713	

}
pathDirCam = "/home/trace/ms-cambridge/"
pathDictHome = "/root/bn"
pathDictMtc = "/home/trace/ms-cambridge/part/"

pathDict = {
"bs24":	"/mnt/raid5/trace/MS-production/BuildServer/Traces/24.hour.BuildServer.11-28-2007.07-24-PM.trace.csv.req",
"bs39":	"/mnt/raid5/trace/MS-production/BuildServer/Traces/24.hour.BuildServer.11-28-2007.07-39-PM.trace.csv.req",
"dev5.3.7.19":	"/home/bn/python/DevelopmentToolsRelease/Traces/DevDivRelease.03-05-2008.07-19-PM.trace.csv.req",
"dev5.3.11.19":	"/home/bn/python/DevelopmentToolsRelease/Traces/DevDivRelease.03-05-2008.11-19-PM.trace.csv.req",
"live6.31":	"/home/bn/python/LiveMapsBackEnd/Traces/LiveMapsBE.02-21-2008.06-31-PM.trace.csv.csv.req",
"live3.31":	"/home/bn/python/LiveMapsBackEnd/Traces/LiveMapsBE.02-21-2008.03-31-PM.trace.csv.csv.req",
"mfs253": "/home/bn/python/MSNStorageFileServer/Traces/MSNFS.2008-03-10.02-53.trace.csv.csv.req",
"mfs101": "/home/bn/python/MSNStorageFileServer/Traces/MSNFS.2008-03-10.01-01.trace.csv.csv.req",
"msc116": "/home/bn/python/MSNStorageCFS/Traces/CFS.2008-03-10.01-16.trace.csv.csv.req",
"msc651": "/home/bn/python/MSNStorageCFS/Traces/CFS.2008-03-10.06-51.trace.csv.csv.req",
"dad807": "/home/bn/python/DisplayAdsDataServer/Traces/DisplayAdsDataServer.2008-03-08.08-07.trace.csv.csv.req",
"dap812": "/home/bn/python/DisplayAdsPayload/Traces/DisplayAdsPayload.2008-03-08.08-12.trace.csv.csv.req",
"netsfs" : "filebench-netsfs.req",
"mix" : "mix-trace.req",
"probuild" : "production-build00-1-4K.req"
}



def getPath(traceID, typeID):
    if typeID == "home":
        return pathDictHome + pathDict[traceID]
    if typeID == "cam":
        return pathDirCam + traceID + ".csv.req"
    if typeID == "production":
        return pathDict[traceID]
    if typeID == "filebench":
        return "/home/chai/go_filebench-1.4.8.fsl.0.8/workloads/" + traceID + "/test1_short.txt.req"
    if typeID == "mtc":
        return pathDictMtc + traceID + "_1.req"

PERIODNUM = 10
PERIODLEN = 10 ** 5
logFilename = "./mtvt_result.csv"
# SIZERATE = 0.1

# def parse_line(line, mode):
#     # print("line", line)
#     items = line.strip().split(' ')
#     # print("items", items)
#     if mode=="mtc":
#         time = int(items[0])
#         rw = int(items[1])
#         blkid = int(items[3])
#         return (time, rw, blkid)
#     else:
#         rw = int(items[0])
#         blkid = int(items[2])
#     return (rw, blkid)

# mode = 'w', deal with write reqs
# mode = 'r', ignore write reqs
def load_file(traceID, typeID, sizerate=0.1, p=1, mode='w'):
    readReq = 0
    size = int(sizerate*uclnDict[traceID])
    ssd = PLRU(size, p)
    fin = open(getPath(traceID, typeID), 'r')
    lines = fin.readlines()
    req = 0
    myupdate = 0
    minTime = None
    maxTime = None
    print("load file finished")
    for line in lines:
        items = line.strip().split(' ')        
        reqtype = int(items[0])
        blockid = int(items[2])
        if mode == 'r':
            if reqtype == 1:			
                ssd.delete_cache(blockid)
            else:	
                readReq += 1	
                if readReq % 1000000 == 0:
                    print(readReq)
                req += 1
                ssd.is_hit(blockid)				
                ssd.update_cache(blockid)
        else:
            req += 1
            hit = ssd.is_hit(blockid)
            if reqtype == 0:
                readReq += 1
            if reqtype == 1 and hit:
                ssd.add_update()
                myupdate += 1
            ssd.update_cache(blockid)
            if not hit and ssd.get_top_n(1)==[blockid]:
                myupdate += 1
    fin.close()
    print(traceID, "size", size, p, "myupdate", myupdate)
    print("total hit rate", 1.0*ssd.hit/req, ssd.update)
    logFile = open(logFilename, "a")
    print(traceID, p, sizerate, size, 1.0*ssd.hit/req, ssd.update, req, round(1.0*readReq/req,3), minTime, maxTime, sep=',', file=logFile)
    logFile.close()

# calculate hit ratios of every period
def load_file_time(traceID, typeID, sizerate=0.1, p=1):
    readReq = 0
    # print(traceID)
    historyHit = 0
    historyReq = 0
    size = int(sizerate*uclnDict[traceID])
    # print(sizerate*uclnDict[traceID], size)
    ssd = PLRU(size, p)
    fin = open(getPath(traceID, typeID), 'r')
    lines = fin.readlines()
    print("load file finished")
    logFile = open(logFilename, "a")
    print(traceID, p, sizerate, size, sep=",", end=",", file=logFile)
    for line in lines:
        items = line.split(' ')
        reqtype = int(items[0])
        blockid = int(items[2])
        if reqtype == 1:            
            ssd.delete_cache(blockid)
        else:       
            if readReq % PERIODLEN == 0 and readReq!=0:
                localHitRatio = 1.0*(ssd.hit-historyHit)/(readReq-historyReq)
                historyHit = ssd.hit
                historyReq = readReq
                print(localHitRatio, sep=",", end=",", file=logFile)
            readReq += 1
            ssd.is_hit(blockid)               
            ssd.update_cache(blockid)
    fin.close()
    print("size", size, p)
    print("total hit rate", 1.0*ssd.hit/readReq, ssd.update)
    
    print(1.0*ssd.hit/readReq, ssd.update, 1.0*ssd.hit/ssd.update, sep=',', file=logFile)
    logFile.close()

def load_metadata_dict(filename="metadata.csv"):
    fin = open(filename, 'r')
    lines = fin.readlines()
    for line in lines:
        line = line.strip().split(',')
        traceID = line[0]
        ucln = int(line[9])
        uclnDict[traceID] = ucln


def mtc_test_size_p(path, traceID, totalTimeLength, timeStart, sizerate, p):
    lines = []
    uclnDict = {}
    req = 0
    readReq = 0

    
    load_lines(path, traceID, totalTimeLength, timeStart, lines, uclnDict)
    size = int(sizerate*len(uclnDict))
    # 稀疏周期，size过小
    if size <= 100:
        return 
    print("size=", size)
    ssd = PLRU(size, p)
    for line in lines:
        (time, rw, blockid) = parse_line(line, "gen")
        req += 1
        hit = ssd.is_hit(blockid)
        if rw == 0:
            readReq += 1
        if rw == 1 and hit:
            ssd.add_update()
        ssd.update_cache(blockid)
    
    print(traceID, "size", size, p)
    print("total hit rate", 1.0*ssd.hit/req, "update", ssd.update)
    global g

    if ssd.update > 1.0*size * g * totalTimeLength:
        cost = 1.0*ssd.update/g/totalTimeLength
    else:
        cost = size
    print(ssd.update, size, g*totalTimeLength, 1.0*size * g * totalTimeLength, 1.0*ssd.update/size, cost)
    logFile = open(logFilename, "a")
    print(traceID, timeStart/totalTimeLength, totalTimeLength/danwei, 
        sizerate, size, p, 1.0*ssd.hit/req, ssd.update, req, 
        round(1.0*readReq/req,3), cost, sep=',', file=logFile)
    logFile.close()
    

uclnDict = {
    "prxy_0": 25331,
"usr_1": 319484 ,
"web_0": 4567 
}
# load_metadata_dict()
# # print(uclnDict)
# # print(len(uclnDict))
# l = uclnDict.items()
# l.sort(key=lambda x:(x[1],x[0]))
# for (trace, ucln) in l:
#     print(trace)
#     load_file(trace, "cam", 0.1, 1, 'w')
# traceList = ["prxy_0", "usr_1", "web_0"]
# traceList = ["web_1", "wdev_0", "mds_0", "src2_0", "rsrch_0", "ts_0", "stg_0", "proj_3", 
# "web_0", "src2_1", "usr_1", "src1_2", "src2_2", "prn_0", "stg_1", 
# "prxy_0", "mds_1", "proj_0", "proj_4", "prn_1", "web_2"] 
spList = [(2,1)]
# pList = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# sList = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3]
traceList = ["ts_0", "prn_1", "hm_0", "stg_1", "wdev_0",
"src1_2", "rsrch_0"]
starts = [23, 13, 20, 25, 17, 23, 4]
path = "/home/trace/ms-cambridge/part/"
totalTimeLength = 5*3600*danwei
timeStart = 0
for i in range(0,7):
    traceID = traceList[i]

    for (sizerate, p) in spList:   
        timeStart = starts[i]*totalTimeLength
        start = time.clock()
        mtc_test_size_p(path, traceID, totalTimeLength, timeStart, sizerate, p)
        end = time.clock()
        print(traceID, sizerate, p, "consumed ", end-start, "s")
		# sys.exit(-1) 
