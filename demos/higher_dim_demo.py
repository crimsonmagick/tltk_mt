import sys
sys.path.insert(1, '../robustness')
import os

import MTL as MTL
import time
from numpy import genfromtxt
import numpy as np
from multiprocessing import Pool, freeze_support

import matplotlib.pyplot as plt

if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()

    mode = "cpu"

    i = 10000
    Acomb = [1,0]
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
    
