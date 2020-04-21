import sys
sys.path.insert(1, '../robustness')
import os
import MTL as MTL 
import numpy as np
import time

list_range = (2**i for i in range(10,31))

mode = 'cpu_threaded'

for i in list_range:
# navigation benchmark
# phi_3 = !p11 U p12
    g = 4
    Ap11 = np.random.rand(g,3)
    bp11 =  np.random.rand(g)
    p11 = MTL.Predicate('data1',Ap11,bp11)    

    Ap12 = np.random.rand(g,3)
    bp12 =  np.random.rand(g)
    p12 = MTL.Predicate('data2',Ap12,bp12)

    traces = {} 
    traces['data1'] = np.array([[1,1,1]]*i,dtype=np.float64) #np.random.rand(1,3)
    traces['data2'] = np.array([[1,1,1]]*i,dtype=np.float64) #np.random.rand(1,3)
    
    time_stamps = np.arange(1, i + 1,dtype=np.float32)
    
    root = MTL.Until(0,float('inf'),MTL.Not(p11,mode),p12,mode)
            
    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()
    print("phi_2\t","| Mode:" ,mode,'\t| Samples:', "{:,}".format(i), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
    del traces
    del time_stamps
