import sys
sys.path.insert(1, '../robustness')
import os

import MTL as MTL
import time
from numpy import genfromtxt
import numpy as np
import random as rand

mode = 'cpu_threaded'

Ar1 = 1.1
br1 = 150

Ar2 = np.array([[1],[1],[1]],dtype=np.float64)
br2 = np.array([4500],dtype=np.float64)

r1 = MTL.Predicate('comb',Ar1,br1)
r2 = MTL.Predicate('comb2',Ar2,br2)

# phi = '[](r1 /\ r2)';
root = r1
#root = MTL.Or(r1,r1)

i = 100000000
traces = {'comb': np.ones(i,dtype=np.float32),'comb2':np.array([[1,1,1]]*i,dtype=np.float64)}
# traces['comb2'][0] = [20000,1]
time_stamps = np.arange(1, i + 1,dtype=np.float32)
#traces['comb2'][0] = np.array([20,20,200])

times = []
t0 = time.time()
root.eval_interval(traces, time_stamps)
t1 = time.time()
print("Phi_1_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0,' | Robustness:',root.robustness)

#Change made to test docker update. 
