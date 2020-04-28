import sys
sys.path.insert(1, '../robustness')
import os
import tltk 
import numpy as np
import time

list_range = (2**i for i in range(4,20))

mode = 'cpu_threaded'

for i in list_range:
# staliro aircraft hscc specs
# phi_2 = !([]_[5,150]r3 /\ <>_[300,400] r4) 
# r3: [1 0 0; -1 0 0] x <= [250; -240]
# r4: [1 0 0; -1 0 0] x <= [240; -230]
    
    Ar3 = np.array([[1,0,0],[-1,0,0]],dtype=np.float64)
    br3 = np.array([250,-240],dtype=np.float64)
    
    Ar4 = np.array([[1,0,0],[-1,0,0]],dtype=np.float64)
    br4 = np.array([240,-230],dtype=np.float64)
    
    r3 = tltk.Predicate('data1',Ar3,br3,mode)
    r4 = tltk.Predicate('data2',Ar4,br4,mode)

    traces = {} 
    traces['data1'] = np.array([[1,1,1]]*i,dtype=np.float64)
    traces['data2'] = np.array([[1,1,1]]*i,dtype=np.float64)
    
    time_stamps = np.arange(1, i + 1,dtype=np.float32)
    #root = MTL.Not(MTL.And(MTL.Global(5,150,r3,mode), MTL.Finally(300,400,r4,mode),mode),mode)
    root = r3
    t0 = time.time()
    print(np.array(root.eval_interval(traces, time_stamps)))
    t1 = time.time()
    print("phi_2\t","| Mode:" ,mode,'\t| Samples:', "{:,}".format(i), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
    del traces
    del time_stamps
