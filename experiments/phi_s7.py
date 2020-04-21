import sys
sys.path.insert(1, '../robustness')
import os
import MTL as MTL 
import numpy as np
import time

list_range = (2**i for i in range(10,31))

mode = 'cpu_threaded'

for i in list_range:
   
    Ar7 = [[2.5 -0.6],[[0.1 -0.08]],[[-1.9 -0.4]],[1.7 0.8]]
    br7 = [[5.5],[3.8],[4.9],[7.5]]
    
    Ar8 = [[2.5 -0.6],[[0.1 -0.08]],[[-1.9 -0.4]],[1.7 0.8]]
    br8 = [[4.5],[2.8],[2.9],[2.5]]
    
    r7 = MTL.Predicate('data1',Ar3,br3)
    r8 = MTL.Predicate('data2',Ar4,br4)

    traces = {} 
    traces['data1'] = [[1,1,1]]*i
    traces['data2'] = [[1,1,1]]*i
    #traces = {'data1': [[1,1,1]]*i,dtype=np.float32),'data2': [[1,1,1]]*i,dtype=np.float32)}
    time_stamps = np.arange(1, i + 1,dtype=np.float32)
    #[]<>_[0,100](r7 /\ <>_[0,100]r8]
    root = MTL.Finally(0,float('inf'), MTL.And(r7,MTL.Global(0,float('inf'),r8,MTL.Finally(0,200,r7,mode),mode),mode),mode)
                
    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()
    print("phi_2\t","| Mode:" ,mode,'\t| Samples:', "{:,}".format(i), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
    del traces
    del time_stamps
