from computeInputSignal import ComputeInputSignal
import numpy as np
from MTL import *
import time
from numpy import genfromtxt
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
import os
from systemSimulator import *
from stochasticOptimization import *

if __name__ == '__main__':
    if os.name == 'nt':
        freeze_support()
    Aspeed = 1
    bspeed = 120

    Arpm = 1
    brpm = 4500
    model = 'sldemo_autotrans_mod01'
    # Initialize matlab engine
    eng = initEngine()
    speed_pred = Predicate('speed', Aspeed, bspeed)
    rpm_pred = Predicate('rpm', Arpm, brpm)
    root = Not(And(Finally(0, float("inf"), speed_pred), Finally(0, float("inf"), rpm_pred)))
    simulationTime = 30
    step = 0.05
    inpRange = [0, 100]
    # Initial control point samples
    cpSamples = [
    38.3319,
    25.1618,
    61.0611,
    83.9307,
    73.3755,
    56.4311,
    91.5186]
    # Calculating the interpolated signal from the control points
    signal = ComputeInputSignal("pchip", cpSamples, inpRange, len(cpSamples), simulationTime, step)
    # Simulate the model
    timeStamps, internalStates, output = simulateSystem(eng, model, simulationTime, signal)
    outputs = [row[0] for row in output]
    speedData = np.transpose(outputs)  # signal 1
    outputs = [row[1] for row in output]
    rpmData = np.transpose(outputs)  # signal 2
    timeStamps = np.ravel(timeStamps)
    timeData = np.transpose(timeStamps)
    traces = {'speed': speedData, 'rpm': rpmData}
    # traces = {'data' : data}
    time_stamps = timeData
    root.eval_interval(traces, time_stamps)
    # How many optimization steps?
    samplesPerRound = 50
    N = 25
    nInputs = len(cpSamples)
    distrib = [[1/N for i in range(N)] for j in range(nInputs)]
    inpRanges = [inpRange] * len(cpSamples)
    subDivs = [N] * len(cpSamples)
    allSamples = []
    allRob = []
    for i in range(samplesPerRound):
        curSample = chooseBernoulliSamples(inpRanges, subDivs, distrib)
        signal = ComputeInputSignal("pchip", curSample, inpRange, len(curSample), simulationTime, step)
        timeStamps, internalStates, output = simulateSystem(eng, model, simulationTime, signal)
        outputs = [row[0] for row in output]
        speedData = np.transpose(outputs)  # signal 1
        outputs = [row[1] for row in output]
        rpmData = np.transpose(outputs)  # signal 2
        timeStamps = np.ravel(timeStamps)
        timeData = np.transpose(timeStamps)
        traces = {'speed': speedData, 'rpm': rpmData}
        time_stamps = timeData
        root.eval_interval(traces, time_stamps)
        allSamples.append(curSample)
        allRob.append(root.robustness)
        print('Robustness: ', root.robustness)
    minRobustness = min(allRob)
    indexOfMin = allRob.index(minRobustness)
    if minRobustness <= 0:
        print("FALSIFIED!")
    print("Minimum robustness: ", minRobustness, " at sample: ")
    print(allSamples[indexOfMin])
