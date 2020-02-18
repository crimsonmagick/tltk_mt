import sys
sys.path.insert(1, '../robustness')
import os
import MTL as MTL 
import numpy as np
import time


start = 1000000
stop = 50000000
step = 10000000
mode = 'cpu_threaded'


for i in range(start,stop+1,step):
    Ar3 = [1, 0]
    br3 = [160]
    
    Ar4 = 1
    br4 = 4500
    

    pred3 = MTL.Predicate('data1',Ar3,br3)
    pred4 = MTL.Predicate('data2',Ar4,br4)

    #traces = {'data1': np.ones(i,dtype=np.float32),'data2': np.ones(i,dtype=np.float32)}
    traces = {'data1': [[1,1]]*i,'data2': np.ones(i,dtype=np.float32)}
    time_stamps = np.arange(1, i + 1,dtype=np.float32)

    #root = MTL.Not(MTL.And(MTL.Global(0,100,pred3),MTL.Global(0,100,pred4)))
    #root = MTL.Global(0,1000,pred3,'cpu_threaded')

    root = MTL.Global(0,1000,pred3,"cpu_threaded")

    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()
    print("Phi_1_higher_dim\t","| Mode:" ,mode,'\t| Samples:', i, '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
    del traces
    del time_stamps
