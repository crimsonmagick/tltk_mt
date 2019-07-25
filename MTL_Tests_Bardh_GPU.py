from MTL_Concurrent_time_bounds import *
import time
from numpy import genfromtxt
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.pyplot as plt
n = 0
N = 1000000
range_list = [10,100,1000,10000,1000000]
cpu_duration = []
gpu_duration = []
Aspeed = -1
bspeed = -120

Arpm = 1
brpm = 4500
for i in range_list:
    print('C: ', i)
    speed_pred = Predicate('speed',Aspeed,bspeed,'cpu')
    rpm_pred = Predicate('rpm',Arpm,brpm,'cpu')
    root = Not(And(Finally(0,float('inf'),speed_pred), Finally(0,float('inf'),rpm_pred)))
    #root = Until(0,float('inf'),speed_pred,rpm_pred)
    #root = speed_pred
    # root = Finally(1,2.2,Predicate('geese',-1,-1))

    #traces = {'speed':speedData,'rpm':rpmData}
    traces = {'speed':np.ones(i),'rpm':np.ones(i)}
    time_stamps = np.ones(i)
    t0 = time.time()
    root.eval_interval(traces,time_stamps)
    t1 = time.time()
    cpu_duration.append(t1-t0)
plt.plot(range_list,cpu_duration,'r')
print('------------CPU Threaded--------------')
print('Robustness: ',root.robustness)
print('Time CPU: ', t1 - t0)
print('Data length: ' ,len(traces['speed']))
for i in range_list:
    print('CPU : ', i)
    speed_pred = Predicate('speed',Aspeed,bspeed,'cpu')
    rpm_pred = Predicate('rpm',Arpm,brpm,'cpu')
    root = Not(And(Finally(0,100,speed_pred), Finally(0,100,rpm_pred)))
    #root = Until(0,float('inf'),speed_pred,rpm_pred)
    # root = Finally(1,2.2,Predicate('geese',-1,-1))

    #traces = {'speed':speedData,'rpm':rpmData}
    traces = {'speed':np.ones(i),'rpm':np.ones(i)}
    time_stamps = np.ones(i)
    t0 = time.time()
    root.eval_interval(traces,time_stamps)
    t1 = time.time()
    gpu_duration.append(t1 - t0)
plt.plot( range_list, gpu_duration,'g')
plt.show()
print('------------GPU Threaded--------------')
print('Robustness: ',root.robustness)
print('Time GPU: ', t1 - t0)
print('Data length: ' ,len(traces['speed']))

