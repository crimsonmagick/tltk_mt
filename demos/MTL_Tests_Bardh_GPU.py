from MTL_GPU import *
import time
from numpy import genfromtxt
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()

    n = 0
    N = 1000000
    range_list = [2 ** i for i in range(8, 22)]
    gpu_duration = []
    Aspeed = -1
    bspeed = -120

    Arpm = 1
    brpm = 4500

    mode = 'gpu'

    for i in range_list:
        print('------------GPU Threaded--------------')

        # predicates
        speed_pred = Predicate('speed', Aspeed, bspeed, mode)
        rpm_pred = Predicate('rpm', Arpm, brpm, mode)

        # formula
        root = Not(And(Finally(0, 100, speed_pred, mode), Finally(0, 100, rpm_pred, mode), mode), mode)
        # root = Not(Finally(0, 100, And(speed_pred, rpm_pred,'gpu'),'gpu'), 'gpu')

        traces = {'speed': np.ones(i), 'rpm': np.ones(i)}
        time_stamps = np.arange(1, i + 1)
        t0 = time.time()
        root.eval_interval(traces, time_stamps)
        t1 = time.time()
        gpu_duration.append(t1 - t0)
        print('Trajecotry Size: ', i)
        print('Robustness: ', root.robustness)
        print('Time GPU: ', t1 - t0)

    plt.plot(range_list, gpu_duration, 'g')
    plt.show()
