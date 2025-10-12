#import sys #uncomment if not using export statment for bash
#sys.path.insert(1, 'robustness') #uncomment if not using export for bash
import tltk_mtl as MTL
import numpy as np
import time
#predicate definition
predicate = MTL.Predicate('example data',1,1)


#Ar1 = 1
#br1 = 100
#r1 = MTL.Predicate('speed',Ar1,br1)
mode = 'cpu'
Ar1 = np.array([[1, 0],[0, 1]],dtype=np.float64)
br1 = np.array([160, 4500],dtype=np.float64)
r1 = MTL.Predicate(['speed','rpm'],Ar1,br1,process_type = 'cpu')

root = MTL.Global(0,100,r1)
        

## Trace and Time Stamps

#trace defined as a dictionary
traces = {'speed': np.array([20,20,52,63,64,65,166],dtype=np.float64), 'rpm':np.array([1020,2022,3022,2523,4024,4025,4626],dtype=np.float64)}

time_stamps = np.array([1,2,3,4,5,6.5,101],dtype=np.float32)

## Evaluate Robustness    

t0 = time.time()
print(root.eval_interval(traces, time_stamps))
t1 = time.time()
 

print("TLTk" ,"\t | Mode:" ,mode,'\t| Time: ', '%.4f'%(t1 - t0), '    \t| Robustness:', root.robustness)

