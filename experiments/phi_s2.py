import importlib.util
spec = importlib.util.spec_from_file_location("tltk_mtl", "../robustness/tltk_mtl.cpython-38-x86_64-linux-gnu.so")
tltk_mtl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tltk_mtl)
import os
#import tltk_mtl
import numpy as np
import time

list_range = (2**i for i in range(4,20))

mode = 'cpu_threaded'

for i in list_range:
# staliro aircraft hscc specs
# phi_2 = !([]_[5,150]r3 /\ <>_[300,400] r4) 
# r3: [1 0 0; -1 0 0] x <= [250; -240]
# r4: [1 0 0; -1 0 0] x <= [240; -230]
    
    Ar3 = [[1,0,0],[-1,0,0]]
    br3 = [250,-240]
    
    Ar4 = [[1,0,0],[-1,0,0]]
    br4 = [240,-230]
    
    r3 = tltk_mtl.Predicate('data1',Ar3,br3,mode)
    r4 = tltk_mtl.Predicate('data2',Ar4,br4,mode)

    traces = {} 
    traces['data1'] = [[1,1,1]]*i
    traces['data2'] = [[1,1,1]]*i
    
    time_stamps = np.arange(1, i + 1,dtype=np.float32)
    #root = MTL.Not(MTL.And(MTL.Global(5,150,r3,mode), MTL.Finally(300,400,r4,mode),mode),mode)
    root = r3
    t0 = time.time()
    print(np.array(root.eval_interval(traces, time_stamps)))
    t1 = time.time()
    print("phi_2\t","| Mode:" ,mode,'\t| Samples:', "{:,}".format(i), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', root.robustness)
    del traces
    del time_stamps
