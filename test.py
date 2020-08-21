#coding=utf-8
from __future__ import print_function
from cache_algorithm import PLRU 
import mtc_data_structure
import copy
# l = [(0,0), (0,1), (0,3), (0,4), (0,3), (1,3), (0,5), (0,6), (0,0)]*1000
# policy = {"nrsamples":3, "deltas":0.02, "deltap":0.1, "throt":0.01}
# g = 0.014/3600/(10**7)
# cache = mtc_data_structure.Cache("test", 0.1, 30, 1, policy)
# device = mtc_data_structure.Device(10, g, {"test":cache})
# for i in range(len(l)):
#     (rw, req) = l[i]    
#     (needInmediateM, scheme1, scheme2) = cache.do_req(rw, req)
#     if needInmediateM:
#         (s, p) = device.try_modify(scheme1, scheme2)
#         cacheDict[trace].change_config(s, p) 
#         print("test error needInmediateM")
#     if (i+1)%1000 == 0:
#         potentials = []
#         trace = "test"        
#         potentials.append(cache.get_potential())
#         for p in potentials:
#             for potentail in p:                
#                 print(potentail.p, potentail.size)
#         result = device.get_best_config(potentials)        
#         print("len result=", len(result))
#         print("modified config=", result[0].p, result[0].size)
#         cache.change_config(result[0].size, result[0].p)
#         cache.init_samples()
        
# (size, p, update, hit) = cache.baseline.get_parameters()
# print("base", size, p, update, hit)
# (size, p, update, hit) = cache.cache.get_parameters()
# print("cache", size, p, update, hit, cache.req)
# for potentail in cache.samples:
#     potentail.print_sample()

# minRate = 1.0
# minL = []

# def parse(l, pl, base, cache, n):
#     # print(l)
#     # print(pl, base, cache, n)
#     if n==0:
#         rate = 1.0*cache/base
#         global minRate, minL
#         if rate < minRate:
#             minRate = rate
#             minL = []
#             for item in pl:
#                 minL.append(item)
            
#     else:
#         tpl = copy.copy(pl)
#         (trace, b, c) = l[0]
#         if len(l)-1 >= n:
#             parse(l[1:], tpl, base, cache, n)
#         tpl.append(l[0])
#         parse(l[1:], tpl, base+b, cache+c, n-1)
        
    

# l = [("hm_0", 711573, 383736), ("prn_1", 221592, 119171), ("proj_0", 276508, 177570),
# ("rsrch_0", 140764, 90332), ("src1_2", 226507, 138549), ("src2_0", 332033, 216572),
# ("stg_0", 141085, 98177), ("stg_1", 103682, 66594), ("ts_0", 161075, 84415),
# ("wdev_0", 104466, 68727), ("web_0", 197162, 125957)]

# parse(l, [], 0, 0, 5)
# print(minL, minRate)

update = input("update")
size = input("size")
g = 0.014/3600/10**7
totalTimeLength = 5*3600*10**7
if update > 1.0*size * g * totalTimeLength:
    cost = 1.0*update/g/totalTimeLength
else:
    cost = size
print(cost)