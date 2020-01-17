import sys
sys.path.insert(1, '../robustness')

import MTL as MTL
import time
import numpy as np




mode = "cpu"

i = 10000000
Acomb = [1,1]
bcomb = [150]

#comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
root = MTL.Predicate('comb',Acomb,bcomb)
#root = MTL.Not(MTL.Finally(0,float('inf'),comb_predicate,mode))

traces = {'comb': [[1,0]]*i}
#traces = {'comb': combData.tolist()}
time_stamps = np.arange(1, i + 1)

times = []
t0 = time.time()
root.eval_interval(traces, time_stamps)
t1 = time.time()

print('Run time: ', t1 - t0)
print('Robustness: ', root.robustness)

