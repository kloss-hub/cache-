from __future__ import print_function
from mtc_test_size_p import PLRU
import time
import sys
# baoning
# created 2019.01.14
# latest edited 2019.01.14

#function: dynamically change cache size and update frequency
# but try not to change the hit ratio 

class Log(object):
	"""docstring for Log"""
	def __init__(self, parameters):
		self.reqNum = 0
		self.periodNum = 0
		self.periodLength = parameters.periodLength 
		self.start = time.clock()
		# l is list of delta probs, example: [-0.1, 0.1, -0.2, 0.2]
		self.l = parameters.l
		self.parameters = parameters
		self.cost = 0

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

	def record(self, mc):
		self.cost += mc
		

class Parameter(object):
	"""docstring for Parameter"""
	def __init__(self, size, p, evictQueue, shadowQueue, l, periodLength):
		self.size = size
		self.p = p
		self.evictQueue = evictQueue
		self.shadowQueue = shadowQueue
		self.l = l
		self.periodLength = periodLength
		

class DPLRU(PLRU):
	"""docstring for DPLRU"""
	def __init__(self, parameters):
		super(DPLRU, self).__init__(parameters.size, parameters.p)
		self.evictQueue = ([-1]*parameters.evictQueue, 0, 0)
		self.shadowQueue = ([-1]*parameters.shadowQueue, 0, 0)
		# 0 loc, 1/2/3=cache/eq/sq/nowhere, #is the relevant number of hits
		self.log = {}

	def is_hit(self, req):
		if req in self.log:
			loc = self.log[req][0]
			self.log[req][loc]+=1
		#if not in log, means not in cache/eq/sq, it will appear in cache/sq later
		return super(DPLRU, self).is_hit(req)


	def updatelog(self, req, loc):
		if req in self.log:
			self.log[req][0] = loc
			self.log[req][loc] = 0
		else:
			self.log[req] = [loc, 0, 0, 0]

	def deletelog(self, req, loc):
		if req in self.log:
			if self.log[req][0] == loc:
				del self.log[req]

	def get_req_hit(self, req, loc):
		# print(self.log, req, loc)
		return self.log[req][loc]

	def update_queue(self, req, sign):
		if sign == 2:
			queue = self.evictQueue
		elif sign == 3:
			queue = self.shadowQueue
		self.updatelog(req, sign)
		(l, head, tail) = queue
		old = l[tail]
		if old!=-1:
			# eprint(old, self.log, type(old), type(self.log))
			self.deletelog(old, sign)

		l[tail] = req
		tail = (tail + 1) % len(l)
		if head==tail:
			head = (head + 1) % len(l)
		
		if sign == 2:
			self.evictQueue = (l, head, tail)
		elif sign == 3:
			self.shadowQueue = (l, head, tail)
		# queue = (l, head, tail)
		# eprint(queue)
		# print(self.shadowQueue, self.evictQueue)

	def update_cache(self, req):
		(evict, update) = super(DPLRU, self).update_cache(req)
		if evict != None:
			self.update_queue(evict, 2)
		if update==-1:
			self.update_queue(req, 3)	
		elif update!=None:
			self.updatelog(req, 1)
		return (evict, update)	

	def change_size(self, n):
		l = super(DPLRU, self).change_size(n)
		for req in l:
			self.update_queue(req, 2)

	def get_log(self):
		return self.log

	def get_evictQueue(self):
		return self.evictQueue

	def get_shadowQueue(self):
		return self.shadowQueue

	def parseq(self, loc):
		if loc == 2:
			q = self.evictQueue
		elif loc == 3:
			q = self.shadowQueue
		else:
			q = self.get_top_n(self.size)
		# ll = []
		# (l, head, tail) = q
		# i = head
		# only no eviction of eq&sq, attention
		return q

			

def eprint(s):
	print(s)
		
def load_trace(fin):
	reqs = []
	lines = fin.readlines()
	for line in lines:
		items = line.split(' ')
		reqtype = int(items[0])
		block = int(items[2])
		reqs.append((reqtype, block))
	return reqs

def update_cache(req, ssd):
	hit = ssd.is_hit(req[1])
	if req[0] == 1 and hit:
		ssd.add_update()
	return ssd.update_cache(req[1])

def cal_p(ssd, loc, actualp, vp):
	h = 0
	em = int(1/actualp)
	en = int(1/vp)	
	q = ssd.parseq(loc)
	print("loc:\t", loc)
	print("queue:\t", q)

	if loc!=1:		
		if q[1] < q[2]:
			q = q[0][q[1]:q[2]]	
		else:
			q = q[0]

	for req in q:
		if req == -1:
			continue
		if actualp > vp: # reduce			
			h += min(en-em, ssd.get_req_hit(req, loc))	
		else: #add
			h += max(ssd.get_req_hit(req, loc)-en, 0)
	if actualp > vp: #reduce
		h = -h
	return h

# calculate the changing size with objective
# parameters: ssd, loc is for identifying location, but objective can also do it
# objective is an integer, means # of hits you need to add or reduce 
# upbound is the max value to change. It is common for data not to be hitted so the upbound will become the final result.
def cal_s(ssd, loc, objective, upbound):
	i = 0
	
	print("objective\t", objective)

	if objective == 0:
		return 0

	if objective > 0:
		q = ssd.parseq(2)
		(l, head, tail) = q
		
		while l[head]!=-1 and i<upbound:
			req = l[head]
			objective-=ssd.get_req_hit(req, 2)
			i += 1
			head = (head+1) % len(l)
			if objective<=0:
				return i
		return upbound

	# objective < 0
	q = ssd.get_tail_n(upbound)
	for req in q:
		objective += ssd.get_req_hit(req, 1)
		i += 1
		if objective>=0:
			return i
	return upbound
			
		

def modify_config(ssd, baseline, log):
	objective = baseline.get_hit() - ssd.get_hit()
	op = baseline.get_p()
	osize = baseline.get_size()
	actualp = ssd.get_p()
	actuals = ssd.get_size()
	mc = actualp * actuals
	config = (-1, -1)
	for i in log.l:
		vp = min(1, max(actualp + i, 0.2))
		print("vp", vp)
		if actualp > vp:
			# print(type(ssd))
			# reduce update p, calculate tail of LRU
			h = cal_p(ssd, 1, actualp, vp)

			s = cal_s(ssd, 2, objective-h, int(max(1, osize*0.1))) + actuals
		else:
			h = cal_p(ssd, 3, actualp, vp)
			s = cal_s(ssd, 1, objective-h, int(osize*0.1)) + actuals
		if mc > vp*s:
			mc = vp*s
			config = (vp, s)
		print("vp:", vp, "\th:", h, "\ts:", s, "\tcost:", vp*s, "\tmc:", mc)

	if vp!=actualp and s!=actuals:
		ssd.change_size(s)
		ssd.change_p(vp)
	log.record(mc)
	return config

def do_single_trace(traceName, parameters):

	#load trace
	fin = open(traceName, 'r')
	reqs = load_trace(fin)
	fin.close()

	#initial
	baseline = PLRU(parameters.size, parameters.p)
	ssd = DPLRU(parameters)
	log = Log(parameters)

	for req in reqs:
		if log.tick():
			modify_config(ssd, baseline, log)
		#?what about write request?
		
		debug = baseline.hit + baseline.update
		update_cache(req, baseline)
		update_cache(req, ssd)
		# if baseline.hit + baseline.update - debug == 0:
		# 	print(req, baseline.get_hit(), baseline.get_update(), debug)
		# 	eprint("error")
		# 	sys.exit(-1)

		# ssd.update_cache(req)
	log.print(baseline, ssd)
 
parameters = Parameter(2, 0.7, 2, 2, [-0.2, -0.1, 0.1, 0.2], 14) 
# {
# "size": 30,
# "p":	0.9,
# "evictQueue": 10,
# "shadowQueue": 10,
# "periodLength": 50,
# "l": 
# }
do_single_trace("test.req", parameters)

# fin = open("test.req", "r")
# reqs = load_trace(fin)
# fin.close()
# print(reqs)
# pd = Parameter(3, 0.8, 2, 2)
# ssd = DPLRU(pd)
# for req in reqs:
# 	a = update_cache(req, ssd)
# 	print(a)
# 	print(ssd.get_hit(), ssd.get_update(), ssd.get_log(), ssd.get_evictQueue(), ssd.get_shadowQueue())


###
# test req 
# 0 0 1
# 0 0 2
# 1 0 1
# 0 0 3
# 0 0 4
# 0 0 1
# 0 0 1
# 0 0 2
# 0 0 1
# 0 0 3
# 0 0 5
# 0 0 6
# 0 0 7
# 0 0 8
# 0 0 9
# 0 0 10
###