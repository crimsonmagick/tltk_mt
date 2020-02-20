import sys
sys.path.insert(1, '../robustness')
import os
import MTL as MTL 
import numpy as np
import time

import matplotlib.pyplot as plt

list_range = (2**i for i in range(10,29))

mode = 'cpu_threaded'

x_length = []
y_time = []

for i in list_range:
    Ar3 = 1
    br3 = 160
    
    Ar4 = 1
    br4 = 4500
    
    pred3 = MTL.Predicate('data1',Ar3,br3)
    pred4 = MTL.Predicate('data2',Ar4,br4)

    traces = {'data1': np.ones(i,dtype=np.float32)*5,'data2': np.ones(i,dtype=np.float32)*5}
    time_stamps = np.arange(1, i + 1,dtype=np.float32)
    
    root = MTL.Not(MTL.And(MTL.Finally(0,float('inf'),pred3,mode),MTL.Global(0,float('inf'),pred4,mode),mode),mode)
    
    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()
    
    x_length.append(i)
    y_time.append(t1-t0)
    
    print("phi_b3\t","| Mode:" ,mode,'\t| Samples:', "{:,}".format(i), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
    del traces
    del time_stamps

fig = plt.scatter(x_length, y_time)
plt.xlabel('Trace Length')
plt.ylabel('Computation Time')
plt.savefig('phi_1.pdf')