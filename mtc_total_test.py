#coding=utf-8
from __future__ import print_function
import math
import mtc_data_structure
import time
import sys
import os
from deal_file import load_lines
from deal_file import parse_line

path = "/home/trace/ms-cambridge/part/"  
#++
# 模拟磁盘与ssd文件放置目录
diskpath = "/home/trace/disk/virtual-disk"
ssdDict={}  #ssd映射表
# 磁盘块大小以及模拟磁盘的开始块
blockSize=4096
trace_block_bigin={}
#模拟写入的数据,blockSize字节
tempbuffer='a'*blockSize
free_list={}        #各个trace的空闲位字典
total_free_list=[]  #全部的空闲位

danwei = 10**7
lifespanMonths=36
# g = 10^-7s内一个块允许的写入次数
g = 0.014/3600/danwei
# 摩尔系数，按照摩尔定律18个月翻一倍，计算出来的1个月涨多少，到时候向下取整
molc = math.exp(math.log(2)/18)
traceFileName = None
version=4.6

# The program is used to simulate the total test for multi-tenent cache
# input: the output of handle_csv_time_partition of cambridge.py
#        (which means the req files are 1hr each)
# output: the result of original algorithm and advanced algorithm


# Function: 
# 每次调用把某个trace所有相关的lines都load到程序中

# Parameters:
# trace = fileid, "web_0"
# ttl = total time length, unit is 10^-7 second
# s = start, 10^-7
# lines用来存放所有的req lines
# uclnDict是所有的ucln，key是blockid，value没用

# assume that s|ttl%3600=0

# 功能：混合trace，输出到mix file中
# 目前的程序仍然是所有trace数据都存放到了内存，再输出到文件，再读入。
# 有点蠢。后期如果trace太多，内存不够，可以考虑把代码改成一秒一秒读，输出到文件，再读入

# unitLength the unit length you mix the traces, 10^-7 second
#      例如，ul=1s，说明各个trace file第一秒的req被连接在一起，然后是第2s的
#      1s内的req是不会再被切割的
#      ul设定原则是，必须倍数于后面cache内的优化操作。例如cache调整大小最小单位是0.5s
#      那仍然用1s去混合trace，就会不准
# assume that unit length is 1s and do not need to mix inside a unit
#++
# 创建具有block_num块的文件(模拟磁盘与ssd)
def mkdir(path, block_num):
    with open(path,'wb') as f:
        f.seek(block_num*blockSize-1)
        f.write(b'\x00')
        f.close()		

#读取文件对应偏移量的块 返回buffer
def read_block(path, block_offset):
    with open(path,'r') as f:
        f.seek(block_offset*blockSize)
        buffer=f.read(blockSize)
        f.close()
    return buffer

# 将buffer写入文件对应偏移量的块
def write_block(path, block_offset, buffer):
    with open(path,'w') as f:
        f.seek(block_offset*blockSize)
        f.write(buffer)
        f.close()

# 读取并返回trace中block_id对应的block
def read_disk(trace, block_id):
    return read_block(diskpath+trace,block_id-trace_block_bigin[trace])

# 写入trace中block_id对应的block
def write_disk(trace, block_id, buffer):
    write_block(diskpath+trace,block_id-trace_block_bigin[trace],buffer)

# 读取并返回ssd中对应的块
def read_ssd(ssd_bid):
    return read_block(diskpath+'ssd',ssd_bid)

#写入ssd中对应的块
def write_ssd(ssd_bid, buffer):
    write_block(diskpath+'ssd',ssd_bid,buffer)


# 生成混合请求
#++ 同时计算每个用户的模拟磁盘大小
def generate_reqs(traces, totalTimeLength, unitLength, starts):
    print("traces", traces)
    lines = []
    idxlist = []
    uclnDict = []
    global traceFileName
    traceFileName = path + "mix" + "_" + str(time.process_time()) + ".req"
    logfile = open(traceFileName, 'w')
    #混合文件
    for trace in traces:  #初始化
        lines.append([])
        idxlist.append(0)
        uclnDict.append({})
    for i in range(len(traces)):  
        start = starts[i]
        timeUnitEnd = start + unitLength
        timeEnd = start + totalTimeLength       #这两步是否没有意义呢
        load_lines(path, traces[i], totalTimeLength, start, lines[i], uclnDict[i])
#++     计算每个trace的所需磁盘大小并构建对应文件模拟磁盘
        bid = []
        for line in lines[i]:
            items = line.strip().split(' ')
            bid.append(int(items[3]))                    #记录下所有的bid
        mkdir(diskpath + traces[i],max(bid)-min(bid)+1)  #创建文件，模拟磁盘
        trace_block_bigin[traces[i]]=min(bid)            #记录下磁盘的开始块
#
    timeUnitEnd = unitLength
    timeEnd = totalTimeLength
    while True:
        for i in range(len(traces)):
            while True:
                if idxlist[i] >= len(lines[i]):
                    break
                (mytime, rw, blkid) = parse_line(lines[i][idxlist[i]], "gen")
                mytime -= starts[i]
                if mytime > timeUnitEnd:
                    break
                print(traces[i], mytime, rw, blkid, file=logfile)
                idxlist[i] += 1
        # print("time", time, "timeUnitEnd", timeUnitEnd)
        timeUnitEnd += unitLength
        if timeUnitEnd > timeEnd:
            break
        sign = False
        for i in range(len(traces)):
            if idxlist[i] < len(lines[i]):
                sign = True
                break
        if not sign:
            break
    for i in range(len(traces)):
        print(traces[i], len(lines[i]), idxlist[i])
    logfile.close()
    return uclnDict
        
def get_reqs(traces):
    global traceFileName
    #traceFileName = path + "mix" + "_" + str(time.clock()) + ".req" in generate_reqs
    fp = open(traceFileName, 'r')
    lines = fp.readlines()
    return lines

# 返回"cache"或者"base"模式下的总写入量
def get_total_write(cacheDict, mode):
    w = 0
    for trace in cacheDict:
        if mode == "cache":
            w += cacheDict[trace].cache.get_update()
        elif mode == "base":
            w += cacheDict[trace].baseline.get_update()
        else:
            w += cacheDict[trace].baseline2.get_update()
    return w

# 返回"cache"或者"base"模式下的总空间
def get_total_size(cacheDict, mode):
    s = 0
    for trace in cacheDict:
        if mode == "cache":
            s += cacheDict[trace].cache.size
        elif mode == "base":
            s += cacheDict[trace].baseline.size
        else:
            s += cacheDict[trace].baseline2.size
    return s

# 输出所有结果
def print_result(traces, device, cacheDict, time, starts, periodLength, sizeRate, policy, runTime):

    logfile = "./total_result.csv"
    fp = open(logfile, 'a')
    print(version, time/danwei, periodLength/danwei, sizeRate, policy, sep=',', file=fp)
    for trace in traces:
        print(trace, sep=',', end=',', file=fp)
    print(runTime, file=fp)
    write = get_total_write(cacheDict, "base")
    size = get_total_size(cacheDict, "base")
    cost = device.get_cost(write, time, size)
    print("base", write, size, cost, sep=',', file=fp)
    write = get_total_write(cacheDict, "base2")
    size = get_total_size(cacheDict, "base2")
    cost1 = device.get_cost(write, time, size)
    print("base2", write, size, cost1, cost1/cost, sep=',', file=fp)
    write = get_total_write(cacheDict, "cache")
    size = get_total_size(cacheDict, "cache")
    cost2 = device.get_cost(write, time, device.size)
    print("cache", write, size, cost2, cost2/cost, sep=',',  file=fp)

    for i in range(len(traces)):
        trace = traces[i]
        start = starts[i]/totalTimeLength
        base = 1.0*cacheDict[trace].baseline.get_hit()/cacheDict[trace].req
        base2 = 1.0*cacheDict[trace].baseline2.get_hit()/cacheDict[trace].req
        cache = 1.0*cacheDict[trace].cache.get_hit()/cacheDict[trace].req
        print(trace, start, base,  base2, cache, (base-cache<=cacheDict[trace].policy["throt"]), cacheDict[trace].req, sep=',', file=fp)
        # print(cacheDict[trace].baseline.get_parameters())
        (size, p, update, hit) = cacheDict[trace].baseline.get_parameters()
        print("base", size, p, update, hit, sep=',', file=fp)
        (size, p, update, hit) = cacheDict[trace].baseline2.get_parameters()
        print("base2", size, p, update, hit, sep=',', file=fp)
        (size, p, update, hit) = cacheDict[trace].cache.get_parameters()
        print("cache", size, p, update, hit, sep=',', file=fp)
    fp.close()

# 记录过程中的重要参数，输出调参过程
def record_process(watchDict, cacheDict):
    # print("before", watchDict)
    for trace in cacheDict.keys():
        
        l = []
        cache = cacheDict[trace]
        templ = [cache.baseline, cache.baseline2, cache.cache]
        for i in range(len(templ)):
            item = templ[i]
            paras = item.get_parameters()
            (size, p, update, hit) = paras
            if i==0:
                update += cache.lastBaseUpdate
            if i==2:
                update += cache.lastCacheUpdate
            paras = (size, p, update, hit)
            l.append(paras)
        if trace not in watchDict:
            watchDict[trace] = []
        watchDict[trace].append(l)
    # print("after", watchDict)

def get_cost(write, time, size):
    unitWrite = 1.0*write/size/time
    # print(unitWrite, unitWrite>self.g)
    # 写入量超出额定写入量
    if unitWrite > g:
        cost = 0
        lifespan = lifespanMonths/(unitWrite/g)
        for i in range(int(math.ceil(lifespanMonths/lifespan))):
            tcost = size * (molc**int(lifespan*i))
            cost += tcost
            # print("debug4.6", "tcost=", tcost, "\ti=", i)
    else:
        cost = size
    return cost

def print_watch(watchDict, cacheDict, time):
    logfile = "./total_result.csv"
    fp = open(logfile, 'a')
    l = ["base", "base2", "cache"]
    print(watchDict)
    for trace in cacheDict.keys():
        print("Trace=", trace)
        # print(watchDict[trace])
        print(trace, file=fp)
        lastupdate = [0,0,0]
        for item in watchDict[trace]:
            for i in range(3):
                (size, p, update, hit) = item[i]
                # if i==1:
                myupdate = update-lastupdate[i]
                lastupdate[i] = update
                # else:
                #     myupdate = update
                assert myupdate>=0
                # print(lastupdate, myupdate, update)
                cost = get_cost(myupdate, time, size)
                print(l[i], size, p, hit, myupdate, cost, sep=",", file=fp)
    fp.close()

#++ 将evicted_bid从ssd中驱逐，并返回驱逐的块在ssd中的id
def evicte_in_ssd(trace,evicted_bid):
    sbid=ssdDict[trace][evicted_bid]['ssd_bid']
    if ssdDict[trace][evicted_bid]['needwb']==1:   #需要写回磁盘
        buffer=read_ssd(sbid)
        write_disk(trace,evicted_bid,buffer)        #写
    ssdDict[trace].pop(evicted_bid)                 #从映射表中删除
    return sbid    

#++ 更改配置，需要对有数据的块和空闲块分别处理
def change_ssd_config(trace , del_list , freenode):
    for del_bid in del_list:    #需要删除的节点
        sbid=evicte_in_ssd(trace,del_bid)
        total_free_list.append(sbid)
        freenode = freenode-1
    if freenode > 0:
        for i in range(freenode):
            total_free_list.append(free_list[trace].pop())    #删除空闲节点
    if freenode < 0:
        for i in range(-freenode):
            free_list[trace].append(total_free_list.pop())    #添加空闲节点


def process(traces, starts, totalTimeLength, unitLength, bsizeRate, csizeRate, policy):
    global g
    uclnDict = generate_reqs(traces, totalTimeLength, unitLength, starts)
    for i in range(len(traces)):
        print(traces[i], "ucln=", len(uclnDict[i]))
    # init
    cacheDict = {}
    p = (1, round(bsizeRate/csizeRate, 1))
    dimdm = {}
    print(">>??")         
    for i in range(len(traces)):  #有和trace相同数量个cache    
        trace = traces[i]
        cache = mtc_data_structure.Cache(trace, bsizeRate, csizeRate, len(uclnDict[i]), p, policy)
        cacheDict[trace] = cache     #构建对应的cache,不同的只有ucln，即cache将访问的cache空间大小
        # print(trace, cacheDict[trace].cache.get_size())
        dimdm[trace] = policy["interval"]   #时间间隔    
#++ 初始化ssdDict
        ssdDict[trace]={} 
#   
    # g=tbw/lifespan/capacity
    # size = get_total_size(cacheDict, "base")
    # g是1单位时间(10^-7s)内每个块的基准写入次数
    # k1没有放进来，假设是1，租用1B1单位时间的价格为1
    device = mtc_data_structure.Device(get_total_size(cacheDict, "cache"), g, cacheDict)
#++ 模拟ssd  分配初始free_list
    mkdir(diskpath+"ssd",get_total_size(cacheDict, "cache"))
    # 根据cache的大小按顺序分配
    tempbidbegin=0
    # for trace in traces:
    #     free_list[trace]={}
    #     for bid in range(tempbidbegin,cacheDict[trace].cache.size):
    #         free_list[trace][bid]=0         #以ssd中的bid为key,value里的0代表数据为clean 1代表数据为
    #     tempbidbegin+=cacheDict[trace].cache.size    
    for trace in traces:
        free_list[trace]=[]
        for ssd_bid in range(tempbidbegin,cacheDict[trace].cache.size):
            free_list[trace].append(ssd_bid)
        tempbidbegin += cacheDict[trace].cache.size
#
    # size,g,cachedict
    # print(csizeRate, bsizeRate, size, int(csizeRate/bsizeRate)*size, device.size)
    periodStart = 0
    periodLength = 60*danwei
    reqs = get_reqs(traces)   
    print("Reqs = ", len(reqs))  #得到混合情况下的所有请求
    timestart = time.clock()   
    debugCount = 0   
    # 遍历每个req，进行处理
    for req in reqs:
        (trace, mytime, rw, blkid) = parse_line(req, "get")  #从mix文件中解读出请求
        # mytime += i*totalTimeLength
        # hit = cacheDict[trace].cache.get_hit()                #对应cache的hit值，应该没用到
        (needInmediateM, hit, update, evicted) = cacheDict[trace].do_req(rw, blkid)   #处理请求，返回是否需要立刻更新
        
#++    请求处理后需要真正对cache（或磁盘）进行读写操作
        if update:        #cache不命中，需要写入cache
            if (evicted==None):   #不需要驱逐，说明cache中有空闲位，需要占用新的空闲位
                if (len(free_list[trace])<1):
                    print("error: no free space")
                sbid=free_list[trace].pop() #从尾巴空闲的ssd
            else: #需要驱逐
                sbid=evicte_in_ssd(trace,evicted)
            #加载到ssd中
            buffer=read_disk(trace,blkid)
            write_ssd(sbid,buffer)  
            # 映射表中添加项
            ssdDict[trace][blkid]={}
            ssdDict[trace][blkid]['ssd_bid']=sbid
            ssdDict[trace][blkid]['needwb']=0      #还未修改，此时不需要写回
            if rw==1:   #写操作
                write_ssd(sbid,tempbuffer)
                ssdDict[trace][blkid]['needwb']=1
            else:        #读操作 Q:这里还需要再读一次吗
                ssd_read(sbid)
        else:   #cache命中或直接读取磁盘
            if hit: #cache命中
                sbid=ssdDict[trace][blkid]['ssd_bid']#从映射表中得到数据
                if rw==1:   #写操作
                    write_ssd(sbid,tempbuffer)
                    ssdDict[trace][blkid]['needwb']=1
                else:        #读操作
                    ssd_read(sbid)
            else:   #不借助cache，直接读写磁盘
                if rw==1:   #写操作
                    write_disk(trace,blkid,tempbuffer)
                else:        #读操作
                    read_disk(trace,blkid)
#
        # hit不足，触发【更新操作】
        if needInmediateM and mytime-dimdm[trace]>=policy["interval"]:
            dimdm[trace] = mytime
            schemel = cacheDict[trace].get_hit_scheme()          #能够提高命中率的更改列表deltas与deltap
            temp = device.try_modify(schemel)         #返回修改方案改变的s和p (deltas, deltap)
            # device空间不足（即所有更改方案都不合适），需要强制更新全体缓存配置
            if temp == None:
                debugCount+=1
                if debugCount % 100 == 0:
                    print("mydebugCount", debugCount)
                mytrace = trace
                potentials = []
                # potentials里面  去掉需要更改的trace
                # 其实从逻辑上，是可以把schemel插入到potentials中的
                # 只是考虑到schemel中可能有一些非sample的情况，update是错的，就没放
                # 而且get_best_config是以write为第一优先级
                # 但是schemel的优化是以命中率为第一优先级
                for trace in traces:
                    if trace==mytrace:                        
                        continue
                    l = cacheDict[trace].get_potential()
                    potentials.append(l)
                # print("len of potentials=", len(potentials))
                for scheme in schemel:    #计划
                    (dlts, dltp, thit) = scheme
                    # print(scheme)
                    tsize = cacheDict[mytrace].cache.size+dlts
                    (result, availSize) = device.get_best_config(potentials, tsize)
                    # print("len resurlt=", len(result))
                    assert(len(result)<=len(traces)-1)
                    if result==None or len(result)==0:
                        continue
                    sign = False
                    device.usedSize = cacheDict[mytrace].cache.size   #只计算mytrace的size?
                    for i in range(len(traces)):
                        if traces[i] == mytrace:
                            sign = True
                            continue   
                        if sign:
                            item = result[i-1]
                        else:
                            item = result[i]                    
                        device.try_modify([(item[0], item[1], None)])   #self.usedSize += deltas(item[0])
                        # try_modify在尝试的同时已经将device的usedsize修改了
#++                     更改所有cache的配置
                        (dellist, freenode)=cacheDict[traces[i]].change_config(item[0], item[1])
                        

#
                        # print(item, deltas, deltap)
                    # 把剩余的size【都】分给hit不足的cache Q:是否就是保证了该device的usedsize一定等于device的size呢
                    for i in range(len(schemel)):
                        (tsize, tp, thit) = schemel[i]
                        availSize = device.size-device.usedSize   #可以放在循环外？
                        # print("293 availSize=", availSize)
                        schemel[i] = (availSize, tp, thit)
                    temp = device.try_modify(schemel)
                    (deltas, deltap) = temp
                    s = deltas + cacheDict[trace].cache.get_size()
                    p = deltap + cacheDict[trace].cache.get_p()
                    # print(temp)
                    # print(s, p)
#++     更改配置
                    (dellist, freenode)=cacheDict[trace].change_config(s, p)
#

                    break
            # device空间足够，直接更改该trace的配置
            # Q:这种情况存在吗？不是每次都会将device所有的size都分配给所有的cache吗
            else:
                (deltas, deltap) = temp
                s = deltas + cacheDict[trace].cache.get_size()
                p = deltap + cacheDict[trace].cache.get_p()
                # 在前面try_modify已经调用过
#++     更改配置
                (dellist, freenode)=cacheDict[trace].change_config(s, p) 

#

            # print("hit不足需要更新", "trace=", trace, cacheDict[trace].req,
             #    "baseline=", cacheDict[trace].baseline.get_hit(),
             #    "cache=", cacheDict[trace].cache.get_hit(),
             # "deltas=", deltas, "deltap=", deltap, "s=", s, "p=", p, sep="\t") 
            # print("test error needInmediateM")
            # sys.exit(0)
        # 一个周期结束，修改所有cache配置
        if mytime - periodStart >= periodLength:
            print("one period stop!", mytime/periodLength)
            if policy["watch"][0]:               #policy["watch"]是什么
                if policy["watch"][1] < mytime:
                    break
                elif int(mytime/periodLength)%policy["watch"][-1]==0:
                    record_process(policy["watch"][2], cacheDict)
            periodStart = mytime
            potentials = []      
            for trace in traces:    #potentials：所有trace的potential的【候选集】
                # print("输出", trace, "的候选集：")
                l = cacheDict[trace].get_potential()
                # for tempPotential in l:
                #     tempPotential.print_sample()
                if len(l) == 0:
                    print("debug 候选集为空")
                    break                
                potentials.append(l)
            # print("len(potentials)=", len(potentials))
            if len(potentials) < len(traces):    #这个长度对比的目的是什么呢？
                result = []
            else:
                (result, tsize) = device.get_best_config(potentials, 0) # 返回最好的配置
                assert(len(result)<=len(traces))
            #没有更好的候选集？
            if len(result) == 0:             
                continue                    #直接处理下一条req    
            # 防止修改方案失效
            device.usedSize = 0
            for i in range(len(traces)):
                # print("i=", i, ",trace=", traces[i], result[i].get_size(), result[i].get_p())
                #尝试各个修改方案
                temp = device.try_modify([(result[i][0], result[i][1], None)])
                assert temp!=None        #保证能够找到合适的修改方案，否则退出程序（？）
#++             更改配置
                (dellist, freenode)=cacheDict[traces[i]].change_config(result[i][0], result[i][1])
#
                cacheDict[traces[i]].init_samples()
            print("after config", device.usedSize)
    if policy["watch"][0]:
        print_watch(policy["watch"][2], cacheDict, policy["watch"][-1]*periodLength)
    for i in range(len(traces)):
        cacheDict[traces[i]].finish()
    runTime = time.clock()-timestart     #计算cache运行时间
    print("consumed", runTime, "s")
    (traces, device, cacheDict, totalTimeLength, starts, periodLength, (bsizeRate, csizeRate), policy, runTime)    
    os.remove(traceFileName)

# 因为允许不同trace有不同的开始，所以所有trace时间都对准为0
# traces = ["hm_0", "prn_1", "ts_0", "rsrch_0", "src1_2", 
# "src2_0", "web_0", "stg_1", "proj_0", "wdev_0",
#  "stg_0"]
# starts = [20, 13, 23, 4, 23, 
# 20, 4, 25, 20, 17,
#  0]
traces = ["ts_0", "hm_0",  "wdev_0", "rsrch_0"]      #路径
starts = [23, 13, 20, 25, 17, 23, 4]  #开始时间
totalTimeLength = 5*3600*danwei       #总时长
for i in range(len(starts)):
    starts[i] *= totalTimeLength

unitLength = 1*danwei
policy = {"nrsamples":3, "deltas":0.02, "deltap":0.1, "throt":0.01, 
"interval":int(1*danwei), "hitThrot":0.005, "watch":(True, 1200*danwei, {}, 10)}
#policy是什么

process(traces, starts, totalTimeLength, unitLength, float(sys.argv[1]), float(sys.argv[2]), policy)

# l = [(1669659,73707),
# (778871  ,110556),
# (739300  ,147409),
# (739995  ,184264),
# (678284  ,221117),
# (692262  ,257975),
# (800582  ,294825),
# (682596  ,331681),
# (698915  ,368532)]

# l2 = [
# (73707, 1669659),
# (110560,  1611477),
# (147416,  1563094),
# (184268,  1477185),
# (221123,  1464413),
# (257978,  1450944),
# (294833,  1434010),
# (331685,  1420296),
# (368542,  1413590)
# ]

# l3 = [
# (3360,  11424),
# (10081,  7886),
# (4032,  5920),
# (3360,  121),
# (10081,  121),
# (19283,  87),
# (3360,  122),
# (10081,  122),
# (16359,  78
# )(3360,  154
# )(10081,  154)
# (15057,  106)
# (3360,  197)
# (10081,  197)
# (14739,  123)
# (3360,  796)
# (10081,  796)
# (11645,  418)
# (3360,  254)
# (10081,  254)
# (15134,  148)
# (3360,  123)
# (10081,  123)
# (10363,  78
# )(3360,  671
# )(10081,  671),
# (8824,  394),
# ]

# costl = []
# # for (write, size) in l:
# for (size, write) in l2:
#     # print("write=", write, "size=", size)
#     device = mtc_data_structure.Device(size, g, {})
#     cost = device.get_cost(write, totalTimeLength, size)
    
#     costl.append(cost)
#     print(cost/costl[0])
# print(costl)
# print(costl[1]/costl[0], costl[2]/costl[0])

# for trace in traces:
#     for cache in cacheDict[trace].samples:
#         print(trace)
#         cache.print_sample()