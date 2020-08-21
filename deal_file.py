#coding=utf-8
from __future__ import print_function
import math
import mtc_data_structure
import time
import sys
import os
path = "/home/trace/ms-cambridge/part/"
# path = "./"
danwei = 10**7

# 用来处理和文件读取相关的通用操作
# 参数：
# path=文件路径
# traceID=文件id
# totalTimeLength=需要的总时间长度, 单位是10^-7 second
# timeStart=开始时间, 单位同样是10^-7
# lines用来存放所有的req lines
# uclnDict是所有的ucln，key是blockid，value没用
# 假设timeStart和totalTimeLength都是1h的整数倍

# 读取cambridge.py切割好的长度为1h的文件
# 文件每行的格式为timestamp, rw, line[2], req，中间用空格分割
# 最后两行分别为元数据和uclnDict

def load_lines(path, traceID, totalTimeLength, timeStart, lines, uclnDict):
    start = time.clock()
    traceFileLength = 3600*danwei
    # ul=1, timeStart=0/3600, fileStart=0/1, totalTimeLength=3600/7200, fileEnd=1/2/2/3
    print(traceFileLength, timeStart, timeStart / traceFileLength)
    fileStart = timeStart / traceFileLength
    fileEnd = (timeStart+totalTimeLength) / traceFileLength
    nrfile = totalTimeLength/traceFileLength
    print("traceID", traceID, "fileStart", fileStart, "fileEnd", fileEnd, "nrfile", nrfile)
    #traceID ts_0 fileStart 115.0 fileEnd 120.0 nrfile 5.0  不是很理解这些参数的意义
    print(fileStart, fileEnd)
    for i in range(int(fileStart), int(fileEnd)):
        filename = path + traceID + "_" + str(i+1) + ".req"
        print("filename=",filename)
        fp = open(filename, "r")
        l = fp.readlines()      
        lines.extend(l[:-2])
        d = eval(l[-1])
        uclnDict.update(d)
        # print("test", traceID, len(d), len(uclnDict))
        fp.close()
    print("load lines consumed", time.clock()-start)

# 功能：把读到的一行转变成需要的内容
# mode = "gen", 是整个程序Input的req文件
# mode = "get", 是程序中间生成的mix文件，第一个元素是trace
def parse_line(line, mode):
    # print("line", line)
    items = line.strip().split(' ')
    # print("items", items)
    if mode=="gen":
        time = int(items[0])
        rw = int(items[1])
        blkid = int(items[3])
        return (time, rw, blkid)
    # traceID, time, rw, blkid
    traceID = items[0]
    time = int(items[1])
    rw = int(items[2])
    blkid = int(items[3])
    return (traceID, time, rw, blkid)