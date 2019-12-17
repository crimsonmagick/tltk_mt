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

    Aspeed = 1
    bspeed = 150

    Arpm = -1
    brpm = -4500

    mode = "cpu"

    #speed_pred = MTL.Predicate('speed', Aspeed, bspeed, mode)
    #rpm_pred = MTL.Predicate('rpm', Arpm, brpm, mode)
    #root = MTL.Not(MTL.And(MTL.Finally(0, 100, speed_pred,mode), MTL.Finally(0, 100, rpm_pred,mode),mode),mode)
    #root = MTL.Finally(0,100,MTL.Predicate('rpm', Arpm, brpm),mode)
    #root = MTL.And(speed_pred ,speed_pred,'gpu')

    #root = Finally(1,2.2,Predicate('geese',-1,-1))

    #data = genfromtxt('seqS.csv', delimiter=',')
    #time_data = genfromtxt('seqT.csv')
    #root = MTL.Predicate('speed',[[1,0],[-1,0],[0,1],[0,-1]],[3, -2, 4, -2])
    #root = two_dim_pred
    
    #root = MTL.Finally(0,float('inf'),MTL.Or(MTL.Predicate('speed',[-1, 0],[-120]),MTL.Predicate('rpm',[0 ,-1],[-4500])))
    
    #root = MTL.Predicate('speed',[[1,0],[-1,0],[0,1],[0,-1]],[3, -2, 4, -2])

    #combData = np.array(data)
    #timeData = np.array(time_data)
    
    # traces = {'speed': speedData, 'rpm': rpmData}
    # #traces = {'data' : data}
    # time_stamps = timeData
    i = 10
    #Acomb = [[1,0],[0,1]]
    #bcomb = [150, 4500]
    Acomb = [1,1]
    bcomb = [150]
    #root = MTL.Not(MTL.And(MTL.Finally(0,float('inf'),MTL.Predicate('comb',Acomb,bcomb)),MTL.Finally(0,float('inf'),MTL.Predicate('comb',Acomb,bcomb))))
    #plt.scatter(timeData, combData[1])
    #plt.show()
    #root = MTL.Not(MTL.And(MTL.Finally(0,float('inf'),MTL.Predicate('comb',Acomb,bcomb)),MTL.Finally(0,float('inf'),MTL.Predicate('comb',Acomb,bcomb))))
    traces = {'comb': [[175,54]]*i}
    root = MTL.Finally(0,100,MTL.Predicate('comb', Acomb, bcomb))
    #traces = {'comb': combData.tolist()}
    time_stamps = np.arange(1, i + 1)
    
    # Maybe use mgrid for 2d trace generation
    # X,Y = np.mgrid[0:i:1, 0:i:1]
    times = []
    t0 = time.time()
    root.eval_interval(traces, time_stamps)
    t1 = time.time()

    print('Run time: ', t1 - t0)
    print('Robustness: ', root.robustness)
    
