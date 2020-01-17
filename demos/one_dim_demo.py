import sys
sys.path.insert(1, '../robustness')

import MTL
import time
import numpy as np



Aspeed = 1
bspeed = 150

Arpm = -1
brpm = -4500

mode = "cpu"

i = 1000000

speed_predicate = MTL.Predicate('speed',Aspeed,bspeed)

traces = {'speed': [1]*i}
root = MTL.Not(MTL.Finally(0,100,speed_predicate,mode))

time_stamps = np.arange(1, i + 1)

times = []
t0 = time.time()
root.eval_interval(traces, time_stamps)
t1 = time.time()

print('Run time: ', t1 - t0)
print('Robustness: ', root.robustness)
    
