from MTL import *
import time
from numpy import genfromtxt
import matplotlib.pyplot as plt
from multiprocessing import Pool


Aspeed = 1
bspeed = 120

Arpm = 1
brpm = 4500

speed_pred = Predicate('speed',Aspeed,bspeed,False)
rpm_pred = Predicate('rpm',Arpm,brpm,False)
root = Not(And(Finally(0,100,speed_pred), Finally(0,100,rpm_pred)))
#root = Until(0,float('inf'),speed_pred,rpm_pred)
# root = Finally(1,2.2,Predicate('geese',-1,-1))

data = genfromtxt('data.csv', delimiter=',')
time_data = genfromtxt('dataTime.csv')
#two_dim_pred = Predicate('data',[[-1.0,1.0],[1.0,1.0]],[-120.0,4500.0])
#root = two_dim_pred

speedData = np.transpose(data[:,0])

rpmData = np.transpose(data[:,1])

timeData = np.transpose(time_data)

traces = {'speed':speedData,'rpm':rpmData}
#traces = {'data' : data}
time_stamps = timeData
t0 = time.time()
root.eval_interval(traces,time_stamps)
t1 = time.time()
print('Robustness: ',root.robustness)
print('Time: ', t1 - t0)
