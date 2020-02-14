import sys
sys.path.insert(1, '../robustness')
import os
import MTL as MTL 
import numpy as np
import time

start = 100000000
stop = start * 2
step = 1
mode = 'cpu'

for i in range(start,stop+1,step):
    Ar3 = 150
    br3 = 20

    

    pred3 = MTL.Predicate('data',Ar3,br3)
    #pred4 = MTL.Predicate('data',Ar4,br4)

    traces = {'data': np.array([1]*i,dtype=np.float32)}
    time_stamps = np.arange(1, i + 1)
    #root = MTL.Global(0,float("inf"),MTL.And(pred3,MTL.Finally(0,100,pred4,mode),mode),mode)
    root = pred3    
    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()
    print("Phi_1_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
