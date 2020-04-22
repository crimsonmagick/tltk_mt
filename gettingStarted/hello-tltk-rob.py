import sys
sys.path.insert(1, '../robustness')
import os
import MTL as MTL 
import numpy as np
import time

## Options
mode = 'cpu_threaded'

## Predicates and Formulas

Ar3 = -1
br3 = -160
r1 = MTL.Predicate('data1',Ar3,br3)
root = MTL.Not(MTL.Finally(0,float('inf'),r1,mode),mode)

## Trace and Time Stamps
traces = {'data1': np.ones(10,dtype=np.float32)*5}
time_stamps = np.arange(10, 10 + 1,dtype=np.float32)

## Evaluate Robustness    
t0 = time.time()
root.eval_interval(traces, time_stamps)
t1 = time.time()
    
print("TLTk" ,"\t | Mode:" ,mode,'\t| Time: ', '%.4f'%(t1 - t0), '    \t| Robustness:', root.robustness)