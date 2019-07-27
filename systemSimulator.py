from computeInputSignal import ComputeInputSignal
import numpy as np
from MTL import *
import time
from numpy import genfromtxt
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
import os
import matlab.engine


# Simulate system function
# INPUTS:
#   * eng: matlab engine instance
#   * model: simulink (for the time being) model (string name)
#   * simulationTime: user defined simulation time
# OUTPUTS:
#   * timeStamps: timestamps of simulated trajectory
#   * internalStates: internal states at the timestamps
#   * outputs: outputs at the timestamps

def simulateSystem(eng, model, simulationTime, inpSignal):
    start = 0
    end = 30
    step = 0.05
    steptime = (np.arange(start, end+step, step)).tolist()
    simopt = eng.simget(model)
    simTimeArray = matlab.double([0, simulationTime])
    signalArray = []
    for i in range(len(steptime)):
        signalArray.append([steptime[i], inpSignal[i]])

    signalArray = matlab.double(signalArray)

    # Commence simulink model simulation

    timeStamps, internalStates, outputs = eng.sim(model, simTimeArray, [], signalArray, nargout=3)

    return timeStamps, internalStates, outputs

def initEngine():
    eng = matlab.engine.start_matlab()
    return eng