import sys
sys.path.insert(1, '../robustness')
import os
import MTL as MTL 
import numpy as np
import time

list_range = (2**i for i in range(10,31))

mode = 'cpu_threaded'

for i in list_range:
# phi_2 = !([]_[5,150]r3 /\ <>_[300,400] r4) 
# r3: [1 0 0; -1 0 0] x <= [250; -240]
# r4: [1 0 0; -1 0 0] x <= [240; -230]
    
    Ar3 = [[1,0,0],[-1,0,0]]
    br3 = [[250],[-240]]
    
    Ar4 = [[1,0,0],[-1,0,0]]
    br4 = [[250],[-240]]
    
    r3 = MTL.Predicate('data1',Ar3,br3)
    r4 = MTL.Predicate('data2',Ar4,br4)

    traces = {} 
    traces['data1'] = [[1,1,1]]*i
    traces['data2'] = [[1,1,1]]*i
    #traces = {'data1': [[1,1,1]]*i,dtype=np.float32),'data2': [[1,1,1]]*i,dtype=np.float32)}
    time_stamps = np.arange(1, i + 1,dtype=np.float32)
    
    root = MTL.Not(MTL.And(MTL.Global(5,150,r3,mode), MTL.Finally(300,400,r4,mode),mode),mode)
        
    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()
    print("phi_2\t","| Mode:" ,mode,'\t| Samples:', "{:,}".format(i), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
    del traces
    del time_stamps
