#import sys #uncomment if not using export statment for bash
#sys.path.insert(1, 'robustness') #uncomment if not using export for bash
import MTL as MTL 
import numpy as np
#predicate definition
predicate = MTL.Predicate('example data',1,1)

#signal and time stamps
signal = {'example data':np.array([95,96,97,96,95],dtype=np.float32)}
time_stamps = np.array([0,.5,.7,.8,1],dtype=np.float32)

#calculate predicate and print results 
print(predicate.eval_interval(signal,time_stamps))


