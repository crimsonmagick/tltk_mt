import sys
sys.path.insert(1, '../robustness')
import os
import MTL as MTL 
import numpy as np
import time

start = 10000000
stop = start * 2
step = start * 3
mode = 'cpu'

for i in range(start,stop+1,step):
    Ar3 = 150
    br3 = 20
    
    Ar4 = 400
    br4 = 130
    

    pred3 = MTL.Predicate('data1',Ar3,br3)
    pred4 = MTL.Predicate('data2',Ar4,br4)

    traces = {'data1': np.ones(i,dtype=np.float32),'data2': np.ones(i,dtype=np.float32)}
    time_stamps = np.arange(1, i + 1,dtype=np.float32)
    #root = MTL.Not(MTL.And(MTL.Finally(0,1000,pred3),MTL.Global(0,1000,pred4)))
    root = MTL.Global(0,1000,pred3,'cpu_threaded')
    #print("data generated")
    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()
    print("Phi_1_higher_dim\t","| Mode:" ,mode,'\t| Samples:', i, '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
    del traces
    del time_stamps
