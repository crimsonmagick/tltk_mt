import sys
sys.path.insert(1, '../robustness')
import os

import MTL as MTL
import time
from numpy import genfromtxt
import numpy as np

mode = 'cpu_threaded'

Ar1 = [1,0]
br1 = [150]

Ar2 = [0,1]
br2 = [4500]

r1 = MTL.Predicate('comb',Ar1,br1)
r2 = MTL.Predicate('comb',Ar2,br2)

# phi = '[](r1 /\ r2)';
root = MTL.And(MTL.Global(0,1000,r1,mode), MTL.Global(0,1000,r2,mode),mode)

i = 1000000
traces = {'comb': [[1,0]]*i}
time_stamps = np.arange(1, i + 1)

times = []
t0 = time.time()
root.eval_interval(traces, time_stamps)
t1 = time.time()
print("Phi_1_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)

#Change made to test docker update. 