import sys
import os
import tltk_mtl as MTL
import numpy as np
import time

## Options

mode = 'cpu_threaded'

## Predicates and Formulas

Ar1 = -1
br1 = -160
r1 = MTL.Predicate('data1',Ar1,br1)
root = MTL.Not(MTL.Finally(0,float('inf'),r1))

Ar2 = -1
br2 = -4500
r2 = MTL.Predicate('data2',Ar2,br2)

# root = F(r1 /\ r2)
root = MTL.Not(MTL.Finally(0,float('inf'),MTL.And(r1,r2)))

## Trace and Time Stamps

#trace defined as a dictionary
traces = {'data1': np.ones(10,dtype=np.float32)*5, 'data2': np.ones(10,dtype=np.float32)*10} 
time_stamps = np.arange(1, 10 + 1,dtype=np.float32)

## Evaluate Robustness    

t0 = time.time()
root.eval_interval(traces, time_stamps)
t1 = time.time()
    
print("TLTk" ,"\t | Mode:" ,mode,'\t| Time: ', '%.4f'%(t1 - t0), '    \t| Robustness:', root.robustness)
