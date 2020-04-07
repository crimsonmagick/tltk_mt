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
br1 = 10

Ar2 = np.array([[1],[1],[1]],dtype=np.float64)
br2 = np.array([4500],dtype=np.float64)

r1 = MTL.Predicate('comb',Ar1,br1,process_type = mode)
r2 = MTL.Predicate('comb2',Ar2,br2)

# phi = '[](r1 /\ r2)';
root = r1
#root = MTL.Or(r1,r1)

i = 2
#traces = {'comb': np.ones(i,dtype=np.float32),'comb2':np.array([[1,1,1]]*i,dtype=np.float64)}
traces = {'comb': np.array([1,2,2,5,3],np.float32)}
# traces['comb2'][0] = [20000,1]
time_stamps = np.arange(1, i + 1,dtype=np.float32)
#traces['comb2'][0] = np.array([20,20,200])
root2 = MTL.Next(r1)
times = []
t0 = time.time()
print(root.eval_interval(traces, time_stamps))
print(np.array(root2.eval_interval(traces, time_stamps)))
t1 = time.time()
print("Phi_1_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0,' | Robustness:',root.robustness)

#Change made to test docker update. 
