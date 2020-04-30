import importlib.util
spec = importlib.util.spec_from_file_location("tltk_mtl", "../robustness/tltk_mtl.cpython-38-x86_64-linux-gnu.so")
tltk_mtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tltk_mtl)
import os
import numpy as np
import time

import matplotlib.pyplot as plt


list_range = (2**i for i in range(10,31))

mode = 'cpu_threaded'

x_length = []
y_time = []

for i in list_range:
    Ar3 = -1
    br3 = -160
    
    Ar4 = 1
    br4 = 4500
    
    pred3 = tltk_mtl.Predicate('data1',Ar3,br3)
    pred4 = tltk_mtl.Predicate('data2',Ar4,br4)

    traces = {'data1': [1]*i,'data2': [1]*i}
    time_stamps = list(range(0, i))
    
    root = tltk_mtl.Not(tltk_mtl.And(tltk_mtl.Finally(0,float('inf'),pred3,mode),tltk_mtl.Global(0,float('inf'),tltk_mtl.And(pred4, tltk_mtl.Global(0,float('inf'),tltk_mtl.Finally(0,float('inf'),tltk_mtl.And(pred3,tltk_mtl.Until(0,float('inf'),pred3,pred4,mode),mode),mode),mode),mode),mode),mode),mode)
    
    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()
    
    x_length.append(i)
    y_time.append(t1-t0)

    print("TLTk" ,"| phi_b3","| Mode:" ,mode,'\t| Samples:', "{:,}".format(i), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)


    del traces
    del time_stamps

fig = plt.scatter(x_length, y_time)
plt.xlabel('Trace Length')
plt.ylabel('Computation Time')
plt.savefig('phi_1.pdf')
