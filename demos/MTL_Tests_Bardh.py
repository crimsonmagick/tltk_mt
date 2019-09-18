import sys
sys.path.insert(1, '../robustness')
import os
os.chdir('../robustness')
import MTL as MTL
import time
from numpy import genfromtxt
import numpy as np
from multiprocessing import Pool, freeze_support


if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()

    Aspeed = -1
    bspeed = -120

    Arpm = [1, 1]
    brpm = [4500, 4500]

    speed_pred = MTL.Predicate('speed', Aspeed, bspeed)
    rpm_pred = MTL.Predicate('rpm', Arpm, brpm)
    #root = MTL.Not(MTL.And(MTL.Finally(0, float('inf'), speed_pred), MTL.Finally(0, float('inf'), rpm_pred)))
    #root = MTL.Finally(0,100,MTL.Predicate('rpm', Arpm, brpm))
    root = MTL.Or(speed_pred ,speed_pred,'cpu')

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
    i = 200000000
    traces = {'speed': np.ones(i), 'rpm': np.ones(i)}
    time_stamps = np.arange(1, i + 1)
    times = []
   
    t0 = time.time()
    print(root.eval_interval(traces, time_stamps)[0])
    t1 = time.time()

    print('Run time: ', t1 - t0)
    print('Robustness: ', root.robustness)
    
