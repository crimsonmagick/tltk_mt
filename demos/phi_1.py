import sys
sys.path.insert(1, '../robustness')
import os
import MTL as MTL 
import numpy as np
import time

list_range = (2**i for i in range(10,31))

mode = 'cpu_threaded'

for i in list_range:
    Ar3 = 1
    br3 = 160
    
    Ar4 = 1
    br4 = 4500
    
    pred3 = MTL.Predicate('data1',Ar3,br3)
    pred4 = MTL.Predicate('data2',Ar4,br4)

    traces = {'data1': np.ones(i,dtype=np.float32),'data2': np.ones(i,dtype=np.float32)}
    time_stamps = np.arange(1, i + 1,dtype=np.float32)
    
    root = MTL.Not(MTL.Finally(0,100,MTL.And(pred3,pred4,mode),mode),mode)
        
    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()
    print("phi_1\t","| Mode:" ,mode,'\t| Samples:', "{:,}".format(i), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
    del traces
    del time_stamps
