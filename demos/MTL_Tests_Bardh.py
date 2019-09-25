import sys
sys.path.insert(1, '../robustness')
import os

import MTL as MTL
import time
from numpy import genfromtxt
import numpy as np
from multiprocessing import Pool, freeze_support


if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()

    Aspeed = 1
    bspeed = 120

    Arpm = -1
    brpm = -4500

    mode = "gpu"

    speed_pred = MTL.Predicate('speed', Aspeed, bspeed, mode)
    rpm_pred = MTL.Predicate('rpm', Arpm, brpm, mode)
    #root = MTL.Not(MTL.And(MTL.Finally(0, 100, speed_pred,mode), MTL.Finally(0, 100, rpm_pred,mode),mode),mode)
    #root = MTL.Finally(0,100,MTL.Predicate('rpm', Arpm, brpm),'gpu')
    #root = MTL.And(speed_pred ,speed_pred,'gpu')

    #root = Finally(1,2.2,Predicate('geese',-1,-1))

    # data = genfromtxt('data.csv', delimiter=',')
    # time_data = genfromtxt('dataTime.csv')
    # two_dim_pred = Predicate('data',[[-1.0,1.0],[1.0,1.0]],[-120.0,4500.0])
    # root = two_dim_pred

    # speedData = np.transpose(data[:, 0])

    # rpmData = np.transpose(data[:, 1])

    # timeData = np.transpose(time_data)

    # traces = {'speed': speedData, 'rpm': rpmData}
    # #traces = {'data' : data}
    # time_stamps = timeData
    i = 500000000
    traces = {'speed': np.arange(i), 'rpm': np.arange(i)}
    time_stamps = np.arange(1, i + 1)
    times = []
   
    t0 = time.time()
    root.eval_interval(traces, time_stamps)[0]
    t1 = time.time()

    print('Run time: ', t1 - t0)
    print('Robustness: ', root.robustness)
    
