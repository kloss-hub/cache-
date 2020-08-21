from __future__ import print_function
from cache_algorithm import PLRU
import time
import sys
import random
# baoning
# created 2019.04.25
# latest edited 2019.04.25

#function: no limit of total cache capacity
# every trace can be treated separately
# use sample ssd to find min cost

class Log(object):
    """docstring for Log"""
    def __init__(self, parameters):
        self.reqNum = 0
        self.periodNum = 0
        self.periodLength = parameters.periodLength 
        self.start = time.clock()
        # l is list of delta probs, example: [-0.1, 0.1, -0.2, 0.2]
        # self.l = parameters.l
        self.parameters = parameters
        self.pscost = 0
        self.cost = 0
        self.pastWrite = 0
        self.l = []

    def tick(self):
        self.reqNum += 1
        if self.reqNum % self.periodLength == 0:
            self.periodNum += 1
            return True
        return False

    def print(self, ssd1, ssd2):
        print(ssd1.get_hit(), ssd1.get_update())
        print(ssd2.get_hit(), ssd2.get_update())
        print(ssd1.get_size()*ssd1.get_p(), ssd1.get_size()*ssd1.get_update())
        print(self.cost)

    def record(self, cost, p, s, write):
        #给的cost是p*s，准确的cost是s*写入速率
        self.pscost += cost
        self.cost += s*(write-self.pastWrite)
        self.pastWrite = write
        self.l.append((self.periodNum, cost, p, s, self.pscost, self.cost))


class SamplePolicy(object):
    """docstring for samplePolicy"""
    def __init__(self, samplePolicy):
        (p, myround, matrix, minS, maxS, minP, maxP, parp, pars) = samplePolicy
        self.p = p
        self.myround = myround
        self.matrix = matrix
        self.maxS = maxS
        self.minS = minS
        self.minP = minP
        self.maxP = maxP
        # [0.3, 0.2, 0.1]
        self.parp = parp
        # [0.1, 0.05, 0.02]
        self.pars = pars

    def get_range(present, mymin, mymax, par, matrix):
        l = [present]
        n = 1
        start = present
        end = present
        while(n<matrix):
            if start > mymin:
                start = max(mymin, start-par)
                l.append(start)
                n += 1
            if end < mymax:
                end = min(mymax, end+par)
                l.append(end)
                n += 1    
        return l

    def getSamples(self, parameters):
        l = []
        presentP = parameters.presentP
        presentS = parameters.presentS
        
        plist = get_range(presentP, self.minP, 
            self.maxP, self.parp[-self.p+self.myround-1], self.matrix)
        slist = get_range(presentS, self.minS, self.maxS, 
            self.pars[-self.p+self.myround-1], self.matrix)
        cost = presentP * presentS

        for p in plist:
            for s in slist:
                if p*s < cost:
                    l.append((p,s))
        return l       

    def add_round(self):
        if self.myround < self.p:            
            self.myround += 1   
        else:
            self.myround = 1                                              

class SampleSSD(object):
    """docstring for SampleSSD"""
    def __init__(self, p, s, ssd):
        self.p = p
        self.s = s
        self.add = {}
        self.minus = {}
        self.update = ssd.get_update()
        self.hit = ssd.get_hit()

    def is_hit(self, req, rw, hit):
        myhit = hit
        if myhit:
            if req in self.minus:
                myhit = False
        else:
            if req in self.add:
                myhit = True
        if myhit:
            self.hit += 1
        return myhit

    def add_update(self):
        self.update += 1

    def update_cache(self, roll, req, rw, hit, update):
        myhit = self.is_hit(req, rw, hit)
        if myhit and rw==1:
            self.add_update()
        # in old ssd -> not in add
        # if in minus and updated in sample, del minus
        if hit:
            if req in self.minus and roll<self.p:
                del self.minus[req]
                self.add_update()
        # not in old ssd -> not in minus
        else:
            # in new ssd
            # if in add, del add
            # else if not updated in sample, add minus
            if update:
                if req in self.add:
                    del self.add[req]
                elif roll>=self.p:
                    self.minus[req] = True
            elif roll<self.p:
                self.add[req] = True
                self.add_update()

        # another way to program is to calculate inssd and insample
        # directly change add and minus

class SampleSSDs(object):
    """docstring for SampleSSDs"""
    def __init__(self, parameters, ssd):
        self.ssds = []
        self.samplePolicy = parameters.samplePolicy
        l = self.samplePolicy.getSamples(parameters)
        for (p,s) in l:
            self.ssds.append(SampleSSD(p,s,ssd))
    
    def update_cache(self, roll, req, rw, hit, update):
        for ssd in self.ssds:
            ssd.update_cache(roll, req, rw, hit, update)

    def modify_config(self, basehit, hitRange):
        l = []
        for ssd in self.ssds:
            r = 1.0*(ssd.hit-basehit)/basehit
            if r <= hitRange:
                l.append((ssd.p*ssd.s, ssd.p, ssd.s))
        if l==[]:
            self.samplePolicy.add_round()
        return l

class Parameters(object):
    """docstring for Parameter"""
    def __init__(self, size, p, periodLength, hitRange, samplePolicy):
        self.size = size
        self.p = p
        self.periodLength = periodLength
        self.hitRange = hitRange
        self.samplePolicy = SamplePolicy(samplePolicy)
        

    def present(self, size, p):
        self.presentS = size
        self.presentP = p

def load_trace(fin):
    reqs = []
    lines = fin.readlines()
    for line in lines:
        items = line.split(' ')
        reqtype = int(items[0])
        block = int(items[2])
        reqs.append((reqtype, block))
    return reqs

def update_cache(rw, req, ssd, roll):
    hit = ssd.is_hit(req)
    if rw==1 and hit:
        ssd.add_update()
    (evicted, update) = ssd.update_cache(req)
    if update==-1:
        update=False
    else:
        update=True
    return (hit, update)

def modify_config(samples, ssd, baseline, parameters, log):
    l = samples.modify_config(baseline.get_hit(), parameters.hitRange)
    if l != []:
        l.sort()
        (cost, p, s) = l[0]
        ssd.change_p(p)
        ssd.change_size(s)
        log.record(cost, p, s)
    else:
        log.record(None, ssd.p, None)


def do_single_trace(traceName, parameters):

    #load trace
    fin = open(traceName, 'r')
    reqs = load_trace(fin)
    fin.close()

    #initial
    baseline = PLRU(parameters.size, parameters.p)
    ssd = PLRU(parameters.size, parameters.p)
    samples = SampleSSDs(parameters, ssd)
    log = Log(parameters)

    for (rw, req) in reqs:
        random.seed()
        roll = random.random()
        #?what about write request?
        update_cache(rw, req, baseline, roll)
        (hit, update) = update_cache(rw, req, ssd, roll)
        samples.update_cache(roll, req, rw, hit, update)
        if log.tick():
            modify_config(samples, ssd, baseline, parameters, log)
    log.print(baseline, ssd)
