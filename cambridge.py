from __future__ import print_function
from cache_algorithm import PLRU
import sys
import os
import time

uclnDict = {"usr_1": 172674631, 
"prxy_0": 230657,
"web_0": 1915690}


log = "C:/Users/hp/Desktop/all/cache/metadata.csv"
logFile = open(log, "a")
danwei = 10**7
block_size = 4096


def get_time(line):
    line = line.strip().split(',')
    time = int(float(line[0]))
    return time

# given lines and the filename to write
# output req with timestamp, the format is timestamp, rw, line[2], req
# output metadata and also the uclndict at last
def handle_lines(filename, lines, timeZero):
    print("enter handle_lines", filename)
    start = time.clock()
    flag=False
    readcount=0
    writecount=0   
    readsize = 0
    writesize = 0
    totalDict = {}
    readDict = {}
    writeDict = {}
    nrreq = 0
    print("hallo？？？")
    outfile = open(filename, 'w')
    for line in lines:
        print(line)
        line = line.strip().split(',')
        timestamp = int(line[0])-timeZero
        block_id = int((float(line[4]))/block_size)           #
        block_end = int((float(line[4])+float(line[5])-1)/block_size)
        # if count % 100000 == 0:
        #     print(count)
        if line[3]=='Write':
            rw = 1
            writecount += 1
            writesize+=block_end-block_id+1
        elif line[3]=='Read':
            rw = 0
            readcount+=1
            readsize+=block_end-block_id+1
        else:
            rw = 2
        for i in range(block_id,block_end+1):
            nrreq += 1
            print('{0} {1} {2} {3}'.format(timestamp,rw,line[2],i),file=outfile)
            # print>>outfile, '{0} {1} {2}'.format(rw,line[2],i)
            totalDict[i] = True            
            if rw == 0:
                readDict[i] = True
            elif rw == 1:
                writeDict[i] = True
    print(len(totalDict), len(readDict), len(writeDict), nrreq, readcount, writecount, file=outfile)
    print(totalDict, file=outfile)
    outfile.close()
    end = time.clock()
    print("consumed", end-start, "s", filename)

# function: transfer the original .csv req file into several .req files 
#           including time message

# Parameters:
# fileid example : "web_0"
# filename : including the file path
# time length : the time length of each result file, unit is second 
#               value should be 3600
#               For example, time length = 1h, each result file includes 1hr traces

# output : 
# name is fileid + _ + number, starts from 1, "web_0_1.req"
# each line of result file is 
# timeStamp, rw, original line[2], blockID
# The last two lines of result file are
# len(totalDict), len(readDict), len(writeDict), nrreq, readcount, writecount
# totalDict
def handle_csv_time_partition(fileid, filename, timeLength=3600):
    infile = open(filename, 'r')
    lines = infile.readlines()
    # cut the lines
    line = lines[0]
    timeStart =  get_time(line)
    timeZero = timeStart
    idxStart = 0
    timeEnd = timeStart + timeLength*danwei
    print(len(lines), get_time(lines[-1]))
    nrpar = 0
    for i in range(0, len(lines)):
        time = get_time(lines[i])
        # if i%100000==0:
        #     print(i, time/danwei)
        if time >= timeEnd:
            nrpar += 1
            pfn = "C:/Users/hp/Desktop/all/cache/" + fileid + "_" + str(nrpar) + ".req"
            handle_lines(pfn, lines[idxStart:i], timeZero)
            idxStart = i
            timeStart += timeLength*danwei
            timeEnd += timeLength*danwei
            # print(i, timeStart/danwei, timeEnd/danwei)
    pfn = "C:/Users/hp/Desktop/all/cache/" + fileid + "_" + str(nrpar+1) + ".req"
    handle_lines(pfn, lines[idxStart:], timeZero)
    infile.close()

# order starts from 0
# only deal with continuous case 
# in warm mode, try not to use negative order or too big order
# it will take long time to really replace the cache
def handle_csv_time(fileid, filename, order, time, pattern, ssd, fileIdx):
    print("enter handle_csv_time", fileid, order, time, fileIdx)
    flag=False
    readcount=1
    writecount=1   
    readsize = 1
    writesize = 1
    # outname = filename +'.req'
    count = fileIdx
    totalDict = {}
    readDict = {}
    writeDict = {}
    lba=[[sys.maxsize,0], [sys.maxsize,0], [sys.maxsize,0]]
    infile = open(filename, 'r')
    # outfile = open(outname, 'w')
    nrreq = 0
    lines = infile.readlines()
    if pattern=="warm":
        if fileIdx == 0:
            timeStart = -1
            timeEnd = 10**25
        else:
            line = lines[fileIdx]
            timeBase = get_time(line)
            timeStart = timeBase + order * time
            timeEnd = timeStart + time
        if ssd == None:
            ssd = PLRU(int(0.1 * uclnDict[fileid]), 1)
            oldhit = -1
            oldupdate = -1
        else:
            oldhit = ssd.hit
            oldupdate = ssd.update
    else:
        if order < 0  and fileIdx == 0:
            line = lines[-1]
        else:
            line = lines[fileIdx]
        line = line.strip().split(',')
        timeBase = int(line[0])
        if fileIdx > 0:
            timeStart = timeBase
        else:
            timeStart = timeBase + order * time
        timeEnd = timeStart + time
    print("time", timeBase/danwei, timeStart/danwei, timeEnd/danwei)
    for line in lines[fileIdx:]:
        count += 1
        line = line.strip().split(',')
        timestamp = int(line[0])
        # print(timestamp, timestamp < timeStart, timestamp > timeEnd)
        if timestamp < timeStart:
            continue
        elif timestamp > timeEnd:
            break
        # first coming
        if not flag:
            lineStart = count
            flag = True
        block_id = int((float(line[4]))/block_size)
        block_end = int((float(line[4])+float(line[5])-1)/block_size)
        # if count % 100000 == 0:
        #     print(count)
        if line[3]=='Write':
            rw = 1
            writecount += 1
            writesize+=block_end-block_id+1
        elif line[3]=='Read':
            rw = 0
            readcount+=1
            readsize+=block_end-block_id+1
        else:
            rw = 2
        if lba[2][0] > block_id:
            lba[2][0] = block_id
        if lba[2][1] < block_end:
            lba[2][1] = block_end
        if lba[rw][0] > block_id:
            lba[rw][0] = block_id
        if lba[rw][1] < block_end:
            lba[rw][1] = block_end
        for i in range(block_id,block_end+1):
            # print('{0} {1} {2}'.format(rw,line[2],i),file=outfile)
            # print>>outfile, '{0} {1} {2}'.format(rw,line[2],i)
            totalDict[i] = True
            
            if rw == 0:
                readDict[i] = True
            elif rw == 1:
                writeDict[i] = True
            if pattern == "warm":
                nrreq += 1
                hit = ssd.is_hit(i)
                if line[3]=='Write' and hit:
                    ssd.add_update()
                ssd.update_cache(i)
        if pattern == "warm" and oldhit<0 and ssd.is_full():
            timeBase = timestamp
            timeStart = timeBase + order * time
            timeEnd = timeStart + time
            nrreq = 0
            ssd.oldhit = ssd.hit
            ssd.oldupdate = ssd.update
            readsize = 0
            writesize = 0
            flag = False
    # print("read write", readcount, writecount, readcount/writecount, readsize, writesize, readsize/writesize, sep=',')
    # print("ucln", len(totalDict), len(readDict), len(writeDict), 
        # lba[2][1]-lba[2][0]+1, lba[0][1]-lba[0][0]+1, lba[1][1]-lba[1][0]+1, sep=',')
    lineStop = count
    print(fileid, pattern, order, time/danwei,  
        readsize, writesize, round(1.0*readsize/(readsize+writesize), 2), 
        sep=',', end=',', file=logFile)
    infile.close()
    # outfile.close()
    if pattern!="warm":
        nrreq = 0
        assert len(totalDict)>0
        ssd = PLRU(int(0.1 * len(totalDict)), 1)
        infile = open(filename, 'r')
        # outfile = open(outname, 'w')
        lines = infile.readlines()
        for line in lines[lineStart-1: lineStop]:
            line = line.strip().split(',')
            timestamp = int(line[0])
            # if timestamp < timeStart:
            #     continue
            # elif timestamp > timeEnd:
            #     break
            block_id = int((float(line[4]))/block_size)
            block_end = int((float(line[4])+float(line[5])-1)/block_size)
            if count % 100000 == 0:
                print(count)
            for req in range(block_id, block_end+1):
                nrreq += 1
                hit = ssd.is_hit(req)
                if line[3]=='Write' and hit:
                    ssd.add_update()
                ssd.update_cache(req)
        hit = ssd.hit
        update = ssd.update
    else:
        hit = ssd.hit - oldhit
        update = ssd.update - oldupdate

    print(fileid, hit, nrreq, 1.0*hit/nrreq, update)
    print(nrreq, ssd.size, hit, 1.0*hit/nrreq, update, sep=',', file=logFile)
    logFile.flush()
    return (ssd, count)

def handle_csv(fileid, filename):
    block_size = 4096
    flag=False
    readcount=0
    writecount=0   
    readsize = 0
    writesize = 0 
    outname = filename +'.req'
    count = 0
    totalDict = {}
    readDict = {}
    writeDict = {}
    lba=[[sys.maxsize,0], [sys.maxsize,0], [sys.maxsize,0]]
    infile = open(filename, 'r')
    outfile = open(outname, 'w')

    for line in infile.readlines():
        count += 1
        line = line.strip().split(',')
        block_id = int((float(line[4]))/block_size)
        block_end = int((float(line[4])+float(line[5])-1)/block_size)
        if count % 100000 == 0:
            print(count)
        if line[3]=='Write':
            rw = 1
            writecount += 1
            writesize+=block_end-block_id+1
        elif line[3]=='Read':
            rw = 0
            readcount+=1
            readsize+=block_end-block_id+1
        else:
            rw = 2
        if lba[2][0] > block_id:
            lba[2][0] = block_id
        if lba[2][1] < block_end:
            lba[2][1] = block_end
        if lba[rw][0] > block_id:
            lba[rw][0] = block_id
        if lba[rw][1] < block_end:
            lba[rw][1] = block_end
        for i in range(block_id,block_end+1):
            print('{0} {1} {2}'.format(rw,line[2],i),file=outfile)
            # print>>outfile, '{0} {1} {2}'.format(rw,line[2],i)
            totalDict[i] = True
            
            if rw == 0:
                readDict[i] = True
            elif rw == 1:
                writeDict[i] = True

    print("read write", readcount, writecount, readcount/writecount, readsize, writesize, readsize/writesize, sep=',')
    print("ucln", len(totalDict), len(readDict), len(writeDict), 
        lba[2][1]-lba[2][0]+1, lba[0][1]-lba[0][0]+1, lba[1][1]-lba[1][0]+1, sep=',')
    print(fileid, readcount+writecount, readcount, writecount, round(readcount/writecount, 2), 
        readsize+writesize, readsize, writesize, round(readsize/writesize, 2), 
        len(totalDict), len(readDict), len(writeDict), 
        lba[2][1]-lba[2][0]+1, lba[0][1]-lba[0][0]+1, lba[1][1]-lba[1][0]+1, sep=',', file=logFile)
    # l = [readcount, writecount, readcount/writecount, readsize, writesize, readsize/writesize]
    # print "read write",
    # for item in l:
    # 	print item,
    # print

    # l = [len(totalDict), len(readDict), len(writeDict), lba[2][1]-lba[2][0]+1, lba[0][1]-lba[0][0]+1, lba[1][1]-lba[1][0]+1]
    # print "ucln",
    # for item in l:
    # 	print item,
    # print

    # l = [fileid, readcount+writecount, readcount, writecount, round(readcount/writecount, 2),  
    # readsize+writesize, readsize, writesize, round(readsize/writesize, 2), len(totalDict), len(readDict), 
    # len(writeDict), lba[2][1]-lba[2][0]+1, lba[0][1]-lba[0][0]+1, lba[1][1]-lba[1][0]+1]
    # for item in l:
    # 	print >> logFile, item,
    # print
    infile.close()
    outfile.close()

def handle_req(fileid, filename):
    fin = open(filename, 'r')
    lines = fin.readlines()
    totalDict = {}
    readDict = {}
    writeDict = {}
    lba=[[sys.maxsize,0], [sys.maxsize,0], [sys.maxsize,0]]
    readcount = writecount = 0
    for line in lines:
        items = line.split(' ')
        reqtype = int(items[0])
        block = int(items[2])
        if reqtype == 1:            
            writecount += 1   
            writeDict[block] = True         
        else:       
            readcount += 1
            readDict[block] = True
        totalDict[block] = True
        if lba[2][0] > block:
            lba[2][0] = block
        if lba[2][1] < block+1:
            lba[2][1] = block+1
        if lba[reqtype][0] > block:
            lba[reqtype][0] = block
        if lba[reqtype][1] < block+1:
            lba[reqtype][1] = block+1
    print("read write", readcount, writecount, readcount/writecount, -1, -1, -1, sep=',')
    print("ucln", len(totalDict), len(readDict), len(writeDict), 
        lba[2][1]-lba[2][0]+1, lba[0][1]-lba[0][0]+1, lba[1][1]-lba[1][0]+1, sep=',')
    print(fileid, readcount+writecount, readcount, writecount, round(readcount/writecount, 2), 
        -1, -1, -1, -1, 
        len(totalDict), len(readDict), len(writeDict), 
        lba[2][1]-lba[2][0]+1, lba[0][1]-lba[0][0]+1, lba[1][1]-lba[1][0]+1, sep=',', file=logFile)
    fin.close()

# l2 = ["web_1", "wdev_0", "mds_0", "src2_0", "rsrch_0", "ts_0", "stg_0", "proj_3", 
# "hm_0", "web_0", "src2_1", "usr_1", "src1_2", "src2_2", "prn_0", "stg_1", 
# "prxy_0", "mds_1", "proj_0", "proj_4", "prn_1", "web_2"] 
# l = ["prxy_0", "hm_0", "proj_3", "ts_0", "src2_0", "usr_0",  "wdev_0", "prn_1" ]
traces = ["ts_0", "prn_1", "hm_0", "stg_1", "wdev_0",
"src1_2", "rsrch_0"]
# for root, dirs, files in os.walk('/home/trace/ms-cambridge'):  
# for i in l:
# mode = "warm"
# for i in l:
#     # need to write a method to copy the context of ssd!
#     ssd_stored = None
#     fileIdx_stored = 0
#     # 1s, 1h, 1day
#     for timeLength in [60*danwei, 3600*danwei, 24*3600*danwei]:
#         if mode=="warm":
#             if ssd_stored != None:
#                 ssd = PLRU(1,1)
#                 ssd.copy(ssd_stored)
#             else:
#                 ssd = None
#             fileIdx = fileIdx_stored
#         else:
#             ssd = None
#             fileIdx = 0
#         for order in range(0, 5):
#             start = time.clock()
#             (ssd, fileIdx) = handle_csv_time(i, "/home/trace/ms-cambridge/" + i + ".csv", 
#                 order, timeLength, mode, ssd, fileIdx)
#             if ssd_stored==None:
#                 ssd_stored = PLRU(1,1)
#                 ssd_stored.copy(ssd)
#                 fileIdx_stored = fileIdx
#     # handle_csv_time(i, "/home/trace/ms-cambridge/" + i + ".csv", 0, 3600*10000000)            
#             end = time.clock()
#             print(i, order, timeLength/danwei, start, end, "consumed ", end-start, "s")
traces = ["ts_0", "hm_0",  "wdev_0", "rsrch_0"]      #路径
for i in traces:

    handle_csv_time_partition(i, "C:/Users/hp/Desktop/all/cache/" + i + ".csv", 3600)                 
                 
# handle_req("probuild", "/home/trace/" + "production-build00-1-4K.req")
# logFile.close()