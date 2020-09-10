import sys
import os
import tltk_mtl as MTL 
import numpy as np
import time

list_range = (2**i for i in range(10,31))

mode = 'cpu_threaded'

#for i in list_range:
i = 10

Ar7 = np.array([[2.5, -0.6],[0.1, -0.08],[-1.9, -0.4],[1.7, 0.8]],dtype=np.float64)
br7 = np.array([5.5,3.8,4.9,7.5],dtype=np.float64)

Ar8 = np.array([[2.5, -0.6],[0.1, -0.08],[-1.9, -0.4],[1.7, 0.8]],dtype=np.float64)
br8 = np.array([4.5,2.8,2.9,2.5],dtype=np.float64)

r7 = MTL.Predicate('data1',Ar7,br7)
r8 = MTL.Predicate('data2',Ar8,br8)

traces = {} 
traces['data1'] = np.array([[1,1,1]]*i,dtype=np.float64)
traces['data2'] = np.array([[1,1,1]]*i,dtype=np.float64)
#traces = {'data1': [[1,1,1]]*i,dtype=np.float32),'data2': [[1,1,1]]*i,dtype=np.float32)}
time_stamps = np.arange(1, i + 1,dtype=np.float32)
#root = r8
root = MTL.Not(MTL.Global(0,float('inf'), MTL.Or(r7,MTL.Finally(0,100,r8,mode),mode)))
        
t0 = time.time()
root.eval_interval(traces, time_stamps)
t1 = time.time()

print(r8.robustness_array)
root.reset()
print(r8.robustness_array)
root.eval_interval(traces, time_stamps)
print(r8.robustness_array)

#print("phi_2\t","| Mode:" ,mode,'\t| Samples:', "{:,}".format(i), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
del traces
del time_stamps
