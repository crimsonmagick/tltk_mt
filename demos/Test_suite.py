import sys
sys.path.insert(1, '../robustness')
import os

import MTL as MTL
import time
from numpy import genfromtxt
import numpy as np


import matplotlib.pyplot as plt

modes = ["cpu_threaded", "gpu"]
for mode in modes:
    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    #for i in range(start,stop,step):
    while i<stop:
        Ar1 = [1,0]
        br1 = [150]

        Ar2 = [0,1]
        br2 = [4500]

        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)

        # phi = '[](r1 /\ r2)';
        root = MTL.And(MTL.Global(0,1000,r1,mode), MTL.Global(0,1000,r2,mode),mode)


        traces = {'comb': [[1,0]]*i}
        time_stamps = np.arange(1, i + 1)

        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        print("Phi_1_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step
    print('-----------------------------------------------------')
    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    #for i in range(start,stop,step):
    while i<stop:
        Ar1 = 1
        br1 = 150

        Ar2 = 1
        br2 = 4500

        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)

        # phi = '[](r1 /\ r2)';
        root = MTL.And(MTL.Global(0,1000,r1,mode), MTL.Global(0,1000,r2,mode),mode)


        traces = {'comb': [1]*i}
        time_stamps = np.arange(1, i + 1)

        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        print("Phi_1_one_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step

    print('-----------------------------------------------------')
    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    #for i in range(start,stop,step):
    while i<stop:

        Ar1 = [1,0]
        br1 = [150]
        
        Ar2 = [0,1]
        br2 = [4500]
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        
        # phi = '!F[0,1000](r1 /\ F[0,1000](r1))';
        root = MTL.Not(MTL.Finally(0,1000,MTL.And(r1,MTL.Finally(0,1000,r2,mode),mode),mode),mode)
        
        
        traces = {'comb': [[1,0]]*i}
        #traces = {'comb': combData.tolist()}
        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()

        print("Phi_2_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step
    print('-----------------------------------------------------')
    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    while i<stop:
    #for i in range(start,stop,step):


        Ar1 = 1
        br1 = 150
        
        Ar2 = 1
        br2 = 4500
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        
        # phi = '!F[0,1000](r1 /\ F[0,1000](r1))';
        root = MTL.Not(MTL.Finally(0,1000,MTL.And(r1,MTL.Finally(0,1000,r2,mode),mode),mode),mode)
        
        
        traces = {'comb': [1]*i}
        #traces = {'comb': combData.tolist()}
        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()

        print("Phi_2_one_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step
    print('-----------------------------------------------------')

    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    while i<stop:
    #for i in range(start,stop,step):
        Ar1 = [1,0]
        br1 = [150]
        
        Ar2 = [0,1]
        br2 = [4500]
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        
        # phi = '[](r1 /\ r2)';
        root = MTL.Global(0,float('inf'),MTL.And(r1,r2,mode))
        
        
        traces = {'comb': [[1,0]]*i}
        #traces = {'comb': combData.tolist()}
        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        
        print("Phi_5_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step
        
    print('-----------------------------------------------------')

    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    while i<stop:
    #for i in range(start,stop,step):
        Ar1 = 1
        br1 = 150
        
        Ar2 = 1
        br2 = 4500
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        
        # phi = '[](r1 /\ r2)';
        root = MTL.Global(0,float('inf'),MTL.And(r1,r2,mode))

        traces = {'comb': [1]*i}

        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        
        print("Phi_5_one_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step

    print('-----------------------------------------------------')

    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    while i<stop:
    #for i in range(start,stop,step):
        
        Ar1 = [1,0]
        br1 = [150]
        
        Ar2 = [0,1]
        br2 = [4500]
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        
        # phi = '!(<>[0,1000]r1 /\ []r2)';
        root = MTL.Not(MTL.And(MTL.Finally(0,1000,r1,mode),MTL.Finally(0,1000,r1,mode),mode),mode)
        
        
        traces = {'comb': [[1,0]]*i}
        #traces = {'comb': combData.tolist()}
        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        
        print("Phi_6_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step
        
    print('-----------------------------------------------------')

    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    while i<stop:
    #for i in range(start,stop,step):
        
        Ar1 = 1
        br1 = 150
        
        Ar2 = 1
        br2 = 4500
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        
        # phi = '!(<>[0,1000]r1 /\ []r2)';
        root = MTL.Not(MTL.And(MTL.Finally(0,1000,r1,mode),MTL.Finally(0,1000,r1,mode),mode),mode)
        
        
        traces = {'comb': [1]*i}
        #traces = {'comb': combData.tolist()}
        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        
        print("Phi_6_one_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step
    print('-----------------------------------------------------')

    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    while i<stop:
    #for i in range(start,stop,step):
        Ar1 = [1,0]
        br1 = [150]
        
        Ar2 = [0,1]
        br2 = [4500]
        
        Ar3 = [1,0]
        br3 = [150]
        
        Ar4 = [0,1]
        br4 = [4500]
        
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        r3 = MTL.Predicate('comb',Ar3,br3)
        r4 = MTL.Predicate('comb',Ar4,br4)
        
        # phi = '!(<>[0,1000]r1 /\ []r2)';
        root = MTL.And(MTL.And(MTL.Global(0,float('inf'),MTL.And(r1,r2,mode)), MTL.Finally(0,1000, r3,mode)), MTL.Finally(0,1000, r4,mode),mode) 
        
        traces = {'comb': [[4400,0]]*i}
        #traces = {'comb': combData.tolist()}
        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        print("Phi_7_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step
        
    print('-----------------------------------------------------')
        
    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    while i<stop:
    #for i in range(start,stop,step):
        Ar1 = 1
        br1 = 150
        
        Ar2 = 1
        br2 = 4500
        
        Ar3 = 1
        br3 = 150
        
        Ar4 = 1
        br4 = 4500
        
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        r3 = MTL.Predicate('comb',Ar3,br3)
        r4 = MTL.Predicate('comb',Ar4,br4)
        
        # phi = '!(<>[0,1000]r1 /\ []r2)';
        root = MTL.And(MTL.And(MTL.Global(0,float('inf'),MTL.And(r1,r2,mode)), MTL.Finally(0,1000, r3,mode)), MTL.Finally(0,1000, r4,mode),mode) 
        
        traces = {'comb': [4400]*i}
        #traces = {'comb': combData.tolist()}
        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        print("Phi_7_one_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step

    print('-----------------------------------------------------')

    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    while i<stop:
    #for i in range(start,stop,step):
        Ar1 = [1,0]
        br1 = [150]
        
        Ar2 = [0,1]
        br2 = [4500]
        
        Ar3 = [1,0]
        br3 = [150]
        
        Ar4 = [0,1]
        br4 = [4500]
        
        Ar5 = [0,1]
        br5 = [4500]
        
        Ar6 = [0,1]
        br6 = [4500]
        
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        r3 = MTL.Predicate('comb',Ar3,br3)
        r4 = MTL.Predicate('comb',Ar4,br4)
        r5 = MTL.Predicate('comb',Ar5,br5)
        r6 = MTL.Predicate('comb',Ar6,br6)
        
        # phi = '!(<>[0,1000]r1 /\ []r2)';
        root = MTL.And(MTL.And(MTL.And(MTL.And(MTL.Global(0,float('inf'),MTL.And(r1,r2,mode)),MTL.Finally(0,1000,r3,mode)),MTL.Finally(0,1000,r4,mode)),MTL.Global(0,60,r5,mode),mode),MTL.Global(0,60,r6,mode),mode)
        traces = {'comb': [[0,0]]*i}
        #traces = {'comb': combData.tolist()}
        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        
        print("Phi_8_higher_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step

    print('-----------------------------------------------------')
        
    start = 1000000
    stop = 5000001
    step = 500000
    i = start
    while i<stop:
    #for i in range(start,stop,step):
        Ar1 = 1
        br1 = 150
        
        Ar2 = 1
        br2 = 4500
        
        Ar3 = 1
        br3 = 150
        
        Ar4 = 1
        br4 = 4500
        
        Ar5 = 1
        br5 = 4500
        
        Ar6 = 1
        br6 = 4500
        
        #comb_predicate = MTL.Predicate('comb',Acomb,bcomb)
        r1 = MTL.Predicate('comb',Ar1,br1)
        r2 = MTL.Predicate('comb',Ar2,br2)
        r3 = MTL.Predicate('comb',Ar3,br3)
        r4 = MTL.Predicate('comb',Ar4,br4)
        r5 = MTL.Predicate('comb',Ar5,br5)
        r6 = MTL.Predicate('comb',Ar6,br6)
        
        phi = '!(<>[0,1000]r1 /\ []r2)';
        root = MTL.And(MTL.And(MTL.And(MTL.And(MTL.Global(0,float('inf'),MTL.And(r1,r2,mode)),MTL.Finally(0,1000,r3,mode)),MTL.Finally(0,1000,r4,mode)),MTL.Global(0,60,r5,mode),mode),MTL.Global(0,60,r6,mode),mode)
        traces = {'comb': [0]*i}
        #traces = {'comb': combData.tolist()}
        time_stamps = np.arange(1, i + 1)
        
        times = []
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        
        print("Phi_8_one_dim","| Mode:" ,mode,'| Samples:', i, ' | Time: ', t1 - t0)
        if (i==5000000):
            start=10000000
            stop=50000001
            step = 10000000
            i = start-step
        i += step
